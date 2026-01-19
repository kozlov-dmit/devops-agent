from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


@dataclass(frozen=True)
class VectorHit:
    score: float
    chunk_id: int
    payload: Dict[str, Any]


class QdrantVectorDB:
    """
    Qdrant local mode (no server, no docker):
      client = QdrantClient(path="path/to/db")  # persists changes to disk
      client = QdrantClient(":memory:")         # in-memory

    Поиск: в qdrant-client 1.16+ предпочтительный API — query_points(...) (Query API). :contentReference[oaicite:1]{index=1}
    """

    def __init__(self, *, local_path: str, collection: str):
        self.collection = collection

        if local_path == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            p = Path(local_path)
            p.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(p))

    def ensure_collection(self, *, dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=dim,
                distance=qm.Distance.COSINE,
            ),
        )

    def upsert_batch(self, *, ids: List[int], vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        points = [
            qm.PointStruct(id=int(i), vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, *, query_vector: List[float], top_k: int) -> List[VectorHit]:
        """
        Версионно-устойчивый поиск:
        1) если есть client.search(...) — используем
        2) иначе используем client.query_points(..., query=<vector>, limit=top_k).points :contentReference[oaicite:2]{index=2}
        """
        hits: List[VectorHit] = []

        if hasattr(self.client, "search"):
            res = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )
            for r in res:
                hits.append(
                    VectorHit(
                        score=float(r.score),
                        chunk_id=int(r.id),
                        payload=dict(r.payload or {}),
                    )
                )
            return hits

        if hasattr(self.client, "query_points"):
            # query_points возвращает объект, у которого обычно есть .points
            resp = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                with_payload=True,
                limit=top_k,
            )
            points = getattr(resp, "points", resp)
            for p in points:
                hits.append(
                    VectorHit(
                        score=float(p.score),
                        chunk_id=int(p.id),
                        payload=dict(p.payload or {}),
                    )
                )
            return hits

        # Если вдруг попалась совсем необычная сборка клиента
        raise AttributeError(
            "QdrantClient has neither 'search' nor 'query_points'. "
            "Please check qdrant-client version and API."
        )
