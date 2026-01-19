from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ChunkingConfig:
    max_lines: int = 120
    overlap: int = 20


@dataclass(frozen=True)
class IndexConfig:
    # Сколько чанков за раз отправляем на embeddings + upsert
    batch_size: int = 128
    include_prefixes: tuple[str, ...] = (
        "src/main/java/",
        "src/main/resources/",
        "src/",
        "helm/",
        "charts/",
        "k8s/",
        "deploy/",
    )


@dataclass(frozen=True)
class EmbeddingsConfig:
    model_name: str
    # FastEmbed batch for embedding inference
    batch_size: int = 256


@dataclass(frozen=True)
class QdrantConfig:
    # Local mode: path to local db folder (no docker, no server). :contentReference[oaicite:1]{index=1}
    local_path: str
    collection: str


def load_chunking_config() -> ChunkingConfig:
    return ChunkingConfig(
        max_lines=int(os.getenv("CHUNK_MAX_LINES", "80")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "15")),
    )


def load_index_config() -> IndexConfig:
    return IndexConfig(
        batch_size=int(os.getenv("INDEX_BATCH_SIZE", "64")),
    )


def load_embeddings_config() -> EmbeddingsConfig:
    return EmbeddingsConfig(
        model_name=os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code"),
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "8")),
    )


def load_qdrant_config() -> QdrantConfig:
    local_path = os.getenv("QDRANT_LOCAL_PATH", "./data/qdrant_local")
    collection = os.getenv("QDRANT_COLLECTION", "repo_chunks")
    return QdrantConfig(local_path=local_path, collection=collection)
