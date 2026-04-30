#!/usr/bin/env python3
"""
Initialize the local SQLite workflow registry.

This database tracks discovered YouTube videos, processing status, transcript status,
embedding status, and errors. ChromaDB is used later for vector storage; SQLite is
used here for workflow state.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    channel_url TEXT,
    title TEXT,
    upload_date TEXT,
    duration INTEGER,
    title_filter TEXT,

    discovered_at TEXT NOT NULL,
    last_checked_at TEXT,

    ingest_status TEXT NOT NULL DEFAULT 'discovered',
    transcript_status TEXT NOT NULL DEFAULT 'not_started',
    embedding_status TEXT NOT NULL DEFAULT 'not_started',

    local_dir TEXT,
    transcript_path TEXT,

    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    last_error_type TEXT,
    last_error_message TEXT,

    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_videos_ingest_status
ON videos (ingest_status);

CREATE INDEX IF NOT EXISTS idx_videos_title_filter
ON videos (title_filter);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    details TEXT
);
"""


def get_db_path() -> Path:
    load_dotenv()
    db_path = os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def main() -> int:
    db_path = get_db_path()
    init_db(db_path)
    print(f"Initialized SQLite registry at: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
