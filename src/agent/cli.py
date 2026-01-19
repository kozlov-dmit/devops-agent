from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .config import load_chunking_config, load_index_config, load_embeddings_config, load_qdrant_config
from .embeddings_fastembed import FastEmbedProvider
from .indexer import build_chunks
from .retriever import retrieve_topk
from .store_sqlite import SQLiteStore
from .vectordb_qdrant import QdrantVectorDB
from .llm_client import LLMClient, load_llm_config
from .analyzer import ContextItem, analyze_incident_with_llm


log = logging.getLogger("agent")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def embed_with_adaptive_batch(embedder: FastEmbedProvider, texts: List[str], batch_size: int) -> List[List[float]]:
    bs = max(1, int(batch_size))
    while True:
        try:
            out: List[List[float]] = []
            for i in range(0, len(texts), bs):
                out.extend(embedder.embed_texts(texts[i:i + bs]))
            return out
        except Exception as e:
            msg = str(e)
            if "Failed to allocate memory" in msg or "onnxruntime" in msg:
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                log.warning("OOM in embeddings. Reducing embed batch to %d and retrying...", bs)
                continue
            raise


def cmd_index(repo: Path, out_dir: Path) -> int:
    chunk_cfg = load_chunking_config()
    idx_cfg = load_index_config()
    emb_cfg = load_embeddings_config()
    qcfg = load_qdrant_config()

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "payload.sqlite"

    store = SQLiteStore(db_path=db_path)
    store.init()

    log.info("Scanning repo: %s", repo)
    chunks = build_chunks(repo_root=repo, chunk_cfg=chunk_cfg, index_cfg=idx_cfg)
    log.info("Built chunks: %d", len(chunks))
    if not chunks:
        log.warning("No chunks built. Check include_prefixes/excludes and repo path.")
        return 2

    log.info("Writing payload to SQLite: %s", db_path)
    store.insert_chunks(chunks)

    embedder = FastEmbedProvider(model_name=emb_cfg.model_name, batch_size=emb_cfg.batch_size)
    dim = embedder.dim()

    vectordb = QdrantVectorDB(local_path=qcfg.local_path, collection=qcfg.collection)
    vectordb.ensure_collection(dim=dim)

    batch = idx_cfg.batch_size
    total = len(chunks)
    log.info(
        "Indexing to Qdrant LOCAL (chunks=%d, index_batch=%d, embed_batch=%d, dim=%d, model=%s, qdrant_path=%s, collection=%s)",
        total, batch, emb_cfg.batch_size, dim, emb_cfg.model_name, qcfg.local_path, qcfg.collection,
    )

    i = 0
    while i < total:
        part = chunks[i:i + batch]
        ids = [c.chunk_id for c in part]
        texts = [c.text for c in part]
        vecs = embed_with_adaptive_batch(embedder, texts, emb_cfg.batch_size)

        payloads = [
            {"path": c.path, "language": c.language, "start_line": c.start_line, "end_line": c.end_line}
            for c in part
        ]

        vectordb.upsert_batch(ids=ids, vectors=vecs, payloads=payloads)
        i += len(part)

        if i % max(batch * 20, 1) == 0:
            log.info("Upserted %d / %d chunks...", i, total)

    meta = {
        "repo_root": str(repo),
        "commit_sha": None,
        "embed_model": emb_cfg.model_name,
        "embed_batch_size": emb_cfg.batch_size,
        "chunk_max_lines": chunk_cfg.max_lines,
        "chunk_overlap": chunk_cfg.overlap,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dim": dim,
        "chunks": total,
        "qdrant_local_path": qcfg.local_path,
        "qdrant_collection": qcfg.collection,
    }
    (out_dir / "index_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote meta: %s", out_dir / "index_meta.json")
    return 0


def _make_runtime_clients(index_dir: Path):
    store = SQLiteStore(db_path=index_dir / "payload.sqlite")
    emb_cfg = load_embeddings_config()
    qcfg = load_qdrant_config()
    embedder = FastEmbedProvider(model_name=emb_cfg.model_name, batch_size=emb_cfg.batch_size)
    vectordb = QdrantVectorDB(local_path=qcfg.local_path, collection=qcfg.collection)
    return store, embedder, vectordb


def cmd_run(index_dir: Path, incident_file: Path, topk: int, prefetch: int, max_per_file: int) -> int:
    store, embedder, vectordb = _make_runtime_clients(index_dir)
    incident = json.loads(incident_file.read_text(encoding="utf-8"))

    results = retrieve_topk(
        vectordb=vectordb,
        store=store,
        embedder=embedder,
        incident=incident,
        top_k=topk,
        prefetch_k=prefetch,
        max_per_file=max_per_file,
    )

    print(f"Retrieved chunks: {len(results)}\n")
    for r in results:
        c = r.chunk
        print(
            f"[score={r.score:.4f} base={r.base_score:.4f} rr={r.rerank_score:+.2f}] "
            f"{c.path}:{c.start_line}-{c.end_line} ({c.language})"
        )
    return 0


def cmd_analyze(
    index_dir: Path,
    incident_file: Path,
    out_report: Path,
    topk: int,
    prefetch: int,
    max_per_file: int,
    max_context_chars: int,
) -> int:
    store, embedder, vectordb = _make_runtime_clients(index_dir)
    incident = json.loads(incident_file.read_text(encoding="utf-8"))

    retrieved = retrieve_topk(
        vectordb=vectordb,
        store=store,
        embedder=embedder,
        incident=incident,
        top_k=topk,
        prefetch_k=prefetch,
        max_per_file=max_per_file,
    )

    # собрать контекст с ограничением по размеру
    contexts: List[ContextItem] = []
    used = 0
    for r in retrieved:
        c = r.chunk
        text = c.text
        if used + len(text) > max_context_chars:
            # пропустим, если не помещается
            continue
        contexts.append(
            ContextItem(
                score=r.score,
                base=r.base_score,
                rr=r.rerank_score,
                path=c.path,
                start_line=c.start_line,
                end_line=c.end_line,
                language=c.language,
                text=text,
            )
        )
        used += len(text)

    llm = LLMClient(load_llm_config())
    report = analyze_incident_with_llm(llm=llm, incident=incident, contexts=contexts, temperature=0.1)

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report: {out_report}")
    return 0


def main(argv: List[str] | None = None) -> int:
    _setup_logging()

    p = argparse.ArgumentParser(prog="agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build index for a repo (SQLite payload + Qdrant local vectors)")
    p_index.add_argument("--repo", required=True, type=Path)
    p_index.add_argument("--out", required=True, type=Path)

    p_run = sub.add_parser("run", help="Run retrieval for an incident (prints hits)")
    p_run.add_argument("--index", required=True, type=Path)
    p_run.add_argument("--incident", required=True, type=Path)
    p_run.add_argument("--topk", type=int, default=12)
    p_run.add_argument("--prefetch", type=int, default=80)
    p_run.add_argument("--max-per-file", type=int, default=2)

    p_an = sub.add_parser("analyze", help="Run retrieval + LLM analysis, write JSON report")
    p_an.add_argument("--index", required=True, type=Path)
    p_an.add_argument("--incident", required=True, type=Path)
    p_an.add_argument("--out-report", required=True, type=Path)
    p_an.add_argument("--topk", type=int, default=12)
    p_an.add_argument("--prefetch", type=int, default=80)
    p_an.add_argument("--max-per-file", type=int, default=2)
    p_an.add_argument("--max-context-chars", type=int, default=120_000)

    args = p.parse_args(argv)

    if args.cmd == "index":
        return cmd_index(repo=args.repo, out_dir=args.out)
    if args.cmd == "run":
        return cmd_run(
            index_dir=args.index,
            incident_file=args.incident,
            topk=args.topk,
            prefetch=args.prefetch,
            max_per_file=args.max_per_file,
        )
    if args.cmd == "analyze":
        return cmd_analyze(
            index_dir=args.index,
            incident_file=args.incident,
            out_report=args.out_report,
            topk=args.topk,
            prefetch=args.prefetch,
            max_per_file=args.max_per_file,
            max_context_chars=args.max_context_chars,
        )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
