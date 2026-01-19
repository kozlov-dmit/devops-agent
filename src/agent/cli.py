from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .config import load_chunking_config, load_index_config, load_embeddings_config
from .indexer import build_chunks
from .store_sqlite import SQLiteStore
from .vectordb_faiss import FaissIndex
from .retriever import retrieve_topk
from .embeddings_local import LocalEmbeddingsProvider


log = logging.getLogger("agent")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def cmd_index(repo: Path, out_dir: Path) -> int:
    chunk_cfg = load_chunking_config()
    idx_cfg = load_index_config()
    emb_cfg = load_embeddings_config()

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

    embedder = LocalEmbeddingsProvider(
        model_name=emb_cfg.model_name,
        device=emb_cfg.device,
        batch_size=emb_cfg.batch_size,
        use_e5_prefix=emb_cfg.use_e5_prefix,
    )

    dim = embedder.dim()
    faiss_index = FaissIndex(dim=dim)

    batch = idx_cfg.batch_size
    log.info(
        "Building FAISS index (index_batch=%d, embed_batch=%d, dim=%d, model=%s, device=%s)...",
        batch,
        emb_cfg.batch_size,
        dim,
        emb_cfg.model_name,
        emb_cfg.device,
    )

    i = 0
    total = len(chunks)
    while i < total:
        part = chunks[i:i + batch]
        vecs = embedder.embed_texts([c.text for c in part], is_query=False)
        faiss_index.add(vecs, [c.chunk_id for c in part])
        i += len(part)

        if i % max(batch * 20, 1) == 0:
            log.info("Indexed %d / %d chunks...", i, total)

    faiss_index.save(out_dir)
    log.info("Index saved: %s", out_dir)

    meta = {
        "repo_root": str(repo),
        "commit_sha": None,
        "embed_model": emb_cfg.model_name,
        "embed_device": emb_cfg.device,
        "embed_batch_size": emb_cfg.batch_size,
        "embed_e5_prefix": emb_cfg.use_e5_prefix,
        "chunk_max_lines": chunk_cfg.max_lines,
        "chunk_overlap": chunk_cfg.overlap,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dim": dim,
        "chunks": total,
    }
    (out_dir / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Wrote meta: %s", out_dir / "index_meta.json")

    return 0


def cmd_run(index_dir: Path, incident_file: Path, topk: int) -> int:
    faiss_index = FaissIndex.load(index_dir)
    store = SQLiteStore(db_path=index_dir / "payload.sqlite")

    emb_cfg = load_embeddings_config()
    embedder = LocalEmbeddingsProvider(
        model_name=emb_cfg.model_name,
        device=emb_cfg.device,
        batch_size=emb_cfg.batch_size,
        use_e5_prefix=emb_cfg.use_e5_prefix,
    )

    incident = json.loads(incident_file.read_text(encoding="utf-8"))

    results = retrieve_topk(
        faiss_index=faiss_index,
        store=store,
        embedder=embedder,
        incident=incident,
        top_k=topk,
    )

    print(f"Retrieved chunks: {len(results)}\n")
    for r in results:
        c = r.chunk
        print(f"[score={r.score:.4f}] {c.path}:{c.start_line}-{c.end_line} ({c.language})")
        snippet = c.text
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "\n...<trimmed>..."
        # print(snippet)
        # print("\n" + ("-" * 80) + "\n")

    return 0


def main(argv: List[str] | None = None) -> int:
    _setup_logging()

    p = argparse.ArgumentParser(prog="agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build index for a repo")
    p_index.add_argument("--repo", required=True, type=Path)
    p_index.add_argument("--out", required=True, type=Path)

    p_run = sub.add_parser("run", help="Run retrieval for an incident")
    p_run.add_argument("--index", required=True, type=Path)
    p_run.add_argument("--incident", required=True, type=Path)
    p_run.add_argument("--topk", type=int, default=12)

    args = p.parse_args(argv)

    if args.cmd == "index":
        return cmd_index(repo=args.repo, out_dir=args.out)
    if args.cmd == "run":
        return cmd_run(index_dir=args.index, incident_file=args.incident, topk=args.topk)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
