from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import faiss


@dataclass(frozen=True)
class VectorHit:
    score: float
    chunk_id: int


class FaissIndex:
    """
    MVP: FAISS IndexFlatIP + отдельный mapping row_id -> chunk_id.
    Вектора нормализуем, чтобы IP ~ cosine similarity.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.row_to_chunk_id: List[int] = []

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def add(self, vectors: List[List[float]], chunk_ids: List[int]) -> None:
        if len(vectors) != len(chunk_ids):
            raise ValueError("vectors and chunk_ids length mismatch")

        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise ValueError(f"Invalid vector shape {arr.shape}, expected (*, {self.dim})")

        arr = self._normalize(arr)
        self.index.add(arr)
        self.row_to_chunk_id.extend(chunk_ids)

    def search(self, query_vector: List[float], top_k: int) -> List[VectorHit]:
        q = np.asarray([query_vector], dtype="float32")
        if q.shape[1] != self.dim:
            raise ValueError(f"Query dim mismatch: got {q.shape[1]}, expected {self.dim}")

        q = self._normalize(q)
        scores, ids = self.index.search(q, top_k)

        hits: List[VectorHit] = []
        for score, row_id in zip(scores[0].tolist(), ids[0].tolist()):
            if row_id < 0:
                continue
            chunk_id = self.row_to_chunk_id[row_id]
            hits.append(VectorHit(score=float(score), chunk_id=int(chunk_id)))
        return hits

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        # mapping
        np.save(out_dir / "row_to_chunk_id.npy", np.asarray(self.row_to_chunk_id, dtype="int64"))

    @classmethod
    def load(cls, out_dir: Path) -> "FaissIndex":
        index = faiss.read_index(str(out_dir / "index.faiss"))
        row_map = np.load(out_dir / "row_to_chunk_id.npy")
        obj = cls(dim=index.d)
        obj.index = index
        obj.row_to_chunk_id = row_map.astype("int64").tolist()
        return obj
