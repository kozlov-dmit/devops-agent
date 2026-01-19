from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from fastembed import TextEmbedding


@dataclass
class FastEmbedProvider:
    """
    FastEmbed embeddings via ONNX Runtime (no PyTorch).
    """

    model_name: str
    batch_size: int = 256

    _model: Optional[TextEmbedding] = None
    _dim: Optional[int] = None

    def __post_init__(self) -> None:
        self._model = TextEmbedding(model_name=self.model_name)

    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        v = self.embed_texts(["ping"])
        self._dim = len(v[0])
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            raise RuntimeError("FastEmbed model is not initialized")

        out: List[List[float]] = []
        for vec in self._model.embed(texts, batch_size=self.batch_size):
            out.append(np.asarray(vec, dtype=np.float32).tolist())
        return out
