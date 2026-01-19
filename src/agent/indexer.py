from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .chunking import Chunk, chunk_text_by_lines
from .config import ChunkingConfig, IndexConfig


CODE_EXT = {
    ".java": "java",
    ".kt": "kotlin",
    ".py": "python",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".properties": "properties",
    ".xml": "xml",
    ".json": "json",
    ".md": "markdown",
    ".sql": "sql",
    ".tf": "terraform",
    ".tpl": "helm",
}


EXCLUDED_DIRS = {".git", "target", "build", ".idea", ".gradle", ".mvn", "node_modules"}


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_DIRS for part in path.parts)


def iter_repo_files(repo_root: Path) -> Iterable[Path]:
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if _is_excluded(p):
            continue
        if p.suffix.lower() in CODE_EXT:
            yield p


def _matches_prefixes(rel_path: str, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return True
    # Нормализуем на “/”
    rel = rel_path.replace("\\", "/")
    return any(rel.startswith(pref) for pref in prefixes)


def build_chunks(
    *,
    repo_root: Path,
    chunk_cfg: ChunkingConfig,
    index_cfg: IndexConfig,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    next_id = 1

    for file_path in iter_repo_files(repo_root):
        rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
        if not _matches_prefixes(rel, index_cfg.include_prefixes):
            continue

        try:
            raw = file_path.read_bytes()
        except Exception:
            continue

        # Быстрая эвристика “текстовый файл”: отсечь нули
        if b"\x00" in raw[:4096]:
            continue

        text = raw.decode("utf-8", errors="replace")
        lang = CODE_EXT.get(file_path.suffix.lower(), "text")

        file_chunks = chunk_text_by_lines(
            text=text,
            path=rel,
            language=lang,
            chunk_id_start=next_id,
            max_lines=chunk_cfg.max_lines,
            overlap=chunk_cfg.overlap,
        )
        chunks.extend(file_chunks)
        next_id += len(file_chunks)

    return chunks
