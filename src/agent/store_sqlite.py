from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .chunking import Chunk


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS index_meta (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  repo_root TEXT NOT NULL,
  commit_sha TEXT NULL,
  embed_model TEXT NOT NULL,
  chunk_max_lines INTEGER NOT NULL,
  chunk_overlap INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id INTEGER PRIMARY KEY,
  path TEXT NOT NULL,
  language TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL,
  text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
"""


@dataclass(frozen=True)
class SQLiteStore:
    db_path: Path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def insert_chunks(self, chunks: Iterable[Chunk]) -> None:
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks(chunk_id, path, language, start_line, end_line, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    (c.chunk_id, c.path, c.language, c.start_line, c.end_line, c.text)
                    for c in chunks
                ),
            )

    def get_chunk(self, chunk_id: int) -> Optional[Chunk]:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT chunk_id, path, language, start_line, end_line, text FROM chunks WHERE chunk_id=?",
                (chunk_id,),
            ).fetchone()
            if not row:
                return None
            return Chunk(
                chunk_id=int(row["chunk_id"]),
                path=str(row["path"]),
                language=str(row["language"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                text=str(row["text"]),
            )
