from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from .vectordb_faiss import FaissIndex, VectorHit
from .store_sqlite import SQLiteStore
from .chunking import Chunk


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
        for line in logs[:30]:
            parts.append(str(line))

    traces = incident.get("traces", {})
    if isinstance(traces, dict):
        spans = traces.get("top_spans", [])
        if isinstance(spans, list):
            for s in spans[:20]:
                parts.append(str(s))

    return "\n".join(parts)


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    chunk: Chunk


class EmbeddingsProvider(Protocol):
    def dim(self) -> int: ...
    def embed_texts(self, texts: List[str], *, is_query: bool) -> List[List[float]]: ...


def retrieve_topk(
    *,
    faiss_index: FaissIndex,
    store: SQLiteStore,
    embedder: EmbeddingsProvider,
    incident: Dict[str, Any],
    top_k: int = 10,
) -> List[RetrievedChunk]:
    query_text = incident_to_query_text(incident)
    qv = embedder.embed_texts([query_text], is_query=True)[0]
    hits: List[VectorHit] = faiss_index.search(qv, top_k=top_k)

    results: List[RetrievedChunk] = []
    for h in hits:
        chunk = store.get_chunk(h.chunk_id)
        if not chunk:
            continue
        results.append(RetrievedChunk(score=h.score, chunk=chunk))

    return results
