from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ChunkingConfig:
    max_lines: int = 120
    overlap: int = 20


@dataclass(frozen=True)
class IndexConfig:
    batch_size: int = 64
    # Ограничиваем индексируемые директории (MVP-эвристика)
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
    device: str = "cpu"
    batch_size: int = 32
    use_e5_prefix: bool = True


def load_chunking_config() -> ChunkingConfig:
    return ChunkingConfig(
        max_lines=int(os.getenv("CHUNK_MAX_LINES", "120")),
        overlap=int(os.getenv("CHUNK_OVERLAP", "20")),
    )


def load_index_config() -> IndexConfig:
    return IndexConfig(
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "64")),
    )

def load_embeddings_config() -> EmbeddingsConfig:
    return EmbeddingsConfig(
        model_name=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base"),
        device=os.getenv("EMBED_DEVICE", "cpu"),
        batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        use_e5_prefix=os.getenv("EMBED_E5_PREFIX", "true").lower() == "true",
    )
