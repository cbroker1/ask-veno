"""Transcript lexical (BM25) index: SQLite FTS5 over existing Chroma docs.

A SEPARATE database (.state/lexical_search.sqlite by default) built by
reading the existing youtube_chunks collection — the ingestion registry and
Chroma store are never written to. Rebuilds are staged + atomic
(os.replace). If the file is missing, dense retrieval simply carries on.
"""

from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from pathlib import Path

from app.config import get_lexical_db_path

SCHEMA_SQL = """
CREATE TABLE lexical_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE transcript_chunks (
    chunk_id    TEXT PRIMARY KEY,
    video_id    TEXT,
    video_title TEXT,
    start_hms   TEXT,
    end_hms     TEXT,
    url         TEXT,
    text        TEXT NOT NULL
);
CREATE VIRTUAL TABLE transcript_fts USING fts5(
    text,
    chunk_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


class LexicalUnavailable(Exception):
    pass


def build_from_rows(db_path: Path, rows: list[dict], source: str) -> None:
    """Atomically (re)build the lexical DB from chunk rows:
    {chunk_id, video_id, video_title, start_hms, end_hms, url, text}."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    staging = db_path.with_suffix(db_path.suffix + ".staging")
    if staging.exists():
        staging.unlink()
    conn = sqlite3.connect(staging)
    try:
        conn.executescript(SCHEMA_SQL)
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO transcript_chunks"
                " (chunk_id, video_id, video_title, start_hms, end_hms, url, text)"
                " VALUES (?,?,?,?,?,?,?)",
                [(r["chunk_id"], r.get("video_id"), r.get("video_title"),
                  r.get("start_hms"), r.get("end_hms"), r.get("url"),
                  r["text"]) for r in rows])
            conn.executemany(
                "INSERT INTO transcript_fts (text, chunk_id) VALUES (?,?)",
                [(r["text"], r["chunk_id"]) for r in rows])
            conn.executemany(
                "INSERT INTO lexical_meta (key, value) VALUES (?,?)",
                [("source", source),
                 ("chunk_count", str(len(rows))),
                 ("built_at",
                  _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))])
    finally:
        conn.close()
    os.replace(staging, db_path)


class LexicalDB:
    """Read-only BM25 access over transcript chunks."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path else get_lexical_db_path()
        if not self.db_path.exists():
            raise LexicalUnavailable(f"lexical DB not found: {self.db_path}")
        self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def meta(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM lexical_meta").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) c FROM transcript_chunks").fetchone()["c"]

    def fts_search(self, match_query: str, limit: int = 20) -> list[dict]:
        if not match_query:
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT c.chunk_id, c.video_id, c.video_title, c.start_hms,
                       c.end_hms, c.url, c.text, bm25(transcript_fts) AS bm25_score
                FROM transcript_fts
                JOIN transcript_chunks c ON c.chunk_id = transcript_fts.chunk_id
                WHERE transcript_fts MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """, (match_query, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]
