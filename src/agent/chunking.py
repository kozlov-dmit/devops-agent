from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    path: str
    language: str
    start_line: int
    end_line: int
    text: str


def chunk_text_by_lines(
    *,
    text: str,
    path: str,
    language: str,
    chunk_id_start: int,
    max_lines: int = 120,
    overlap: int = 20,
) -> List[Chunk]:
    lines = text.splitlines()
    chunks: List[Chunk] = []

    i = 0
    chunk_id = chunk_id_start
    while i < len(lines):
        start = i
        end = min(i + max_lines, len(lines))
        chunk_lines = lines[start:end]

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                path=path,
                language=language,
                start_line=start + 1,
                end_line=end,
                text="\n".join(chunk_lines),
            )
        )
        chunk_id += 1

        if end >= len(lines):
            break

        i = end - overlap
        if i < 0:
            i = 0

    return chunks
