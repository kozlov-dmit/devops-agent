from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class LocalEmbeddingsProvider:
    """
    Локальные embeddings через sentence-transformers.
    CPU-only friendly: батчинг + нормализация.
    """
    model_name: str
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True

    # E5-подсказки: "query:" и "passage:"
    use_e5_prefix: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def dim(self) -> int:
        # SentenceTransformer может вернуть dim через encode одного текста
        v = self.embed_texts(["ping"], is_query=True)[0]
        return len(v)

    def embed_texts(self, texts: List[str], *, is_query: bool) -> List[List[float]]:
        if self.use_e5_prefix:
            if is_query:
                texts = [self.query_prefix + t for t in texts]
            else:
                texts = [self.passage_prefix + t for t in texts]

        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32).tolist()
