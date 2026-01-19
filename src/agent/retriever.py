from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from .chunking import Chunk
from .store_sqlite import SQLiteStore
from .vectordb_qdrant import QdrantVectorDB, VectorHit
from .signals import extract_signals, score_chunk_text, path_penalty


def incident_to_query_text(incident: Dict[str, Any]) -> str:
    parts: List[str] = []

    service = incident.get("service")
    if service:
        parts.append(f"service={service}")

    symptoms = incident.get("symptoms", {})
    if isinstance(symptoms, dict):
        for k, v in symptoms.items():
            parts.append(f"{k}={v}")

    logs = incident.get("logs", [])
    if isinstance(logs, list):
        for line in logs[:40]:
            parts.append(str(line))

    traces = incident.get("traces", {})
    if isinstance(traces, dict):
        spans = traces.get("top_spans", [])
        if isinstance(spans, list):
            for s in spans[:30]:
                parts.append(str(s))

    return "\n".join(parts)


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    chunk: Chunk
    base_score: float
    rerank_score: float


class EmbeddingsProvider(Protocol):
    def dim(self) -> int: ...
    def embed_texts(self, texts: List[str]) -> List[List[float]]: ...


def _dedup_per_file(items: List[RetrievedChunk], max_per_file: int) -> List[RetrievedChunk]:
    out: List[RetrievedChunk] = []
    per_file: Dict[str, int] = {}
    for it in items:
        key = it.chunk.path
        cnt = per_file.get(key, 0)
        if cnt >= max_per_file:
            continue
        per_file[key] = cnt + 1
        out.append(it)
    return out


def retrieve_topk(
    *,
    vectordb: QdrantVectorDB,
    store: SQLiteStore,
    embedder: EmbeddingsProvider,
    incident: Dict[str, Any],
    top_k: int = 12,
    prefetch_k: int = 80,
    max_per_file: int = 2,
) -> List[RetrievedChunk]:
    query_text = incident_to_query_text(incident)
    signals = extract_signals(query_text)

    qv = embedder.embed_texts([query_text])[0]
    hits: List[VectorHit] = vectordb.search(query_vector=qv, top_k=prefetch_k)

    candidates: List[RetrievedChunk] = []
    for h in hits:
        chunk = store.get_chunk(h.chunk_id)
        if not chunk:
            continue

        rr = score_chunk_text(chunk.text, signals) + path_penalty(chunk.path)
        candidates.append(
            RetrievedChunk(
                score=float(h.score) + rr,
                base_score=float(h.score),
                rerank_score=float(rr),
                chunk=chunk,
            )
        )

    candidates.sort(key=lambda x: x.score, reverse=True)
    candidates = _dedup_per_file(candidates, max_per_file=max_per_file)
    return candidates[:top_k]
