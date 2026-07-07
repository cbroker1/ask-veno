#!/usr/bin/env python3
"""
init_db.py -- Initialize the local SQLite workflow registry.

This database tracks YouTube video discovery, audio download state,
Whisper transcription state, embedding state, and error history.

ChromaDB is used later for vector storage.
SQLite is used here for workflow state.

USAGE
-----
    python scripts/init_db.py

OPTIONS
    (none)  The script creates the database and schema, then exits.
            Set SQLITE_DB_PATH in .env or environment to change the path.

EXAMPLES
    # Initialize with default path (.state/youtube_ingest.sqlite)
    python scripts/init_db.py

    # Initialize to a custom path
    SQLITE_DB_PATH=/path/to/my.db python scripts/init_db.py

DATABASE SCHEMA
---------------
The script creates two tables:

videos
    video_id          TEXT PRIMARY KEY   - YouTube video ID
    url               TEXT NOT NULL      - Video URL
    channel_url       TEXT               - Source channel URL
    title             TEXT NOT NULL      - Video title
    upload_date       TEXT               - Upload date (YYYYMMDD)
    duration          INTEGER            - Duration in seconds
    title_filters     TEXT               - Filters that matched this video
    discovered_at     TEXT NOT NULL      - When the video was discovered
    last_checked_at   TEXT               - Last discovery check time
    ingest_status     TEXT DEFAULT 'queued'     - queued | audio_ready | transcript_ready | transcript_clean_ready | complete | failed_*
    audio_status      TEXT DEFAULT 'not_started' - not_started | downloaded | failed
    whisper_status    TEXT DEFAULT 'not_started' - not_started | transcribed | failed
    embedding_status  TEXT DEFAULT 'not_started' - not_started | embedded | failed
    local_dir         TEXT               - Local folder path for this video
    audio_path        TEXT               - Path to the audio file
    transcript_path   TEXT               - Path to the raw Whisper transcript JSON
    clean_transcript_status TEXT DEFAULT 'not_started' - not_started | cleaned | failed
    clean_transcript_path TEXT           - Path to the cleaned transcript JSON
    chroma_collection TEXT               - ChromaDB collection name
    chunk_count       INTEGER DEFAULT 0  - Number of chunks in ChromaDB
    embedded_at       TEXT               - When embedding completed
    attempt_count     INTEGER DEFAULT 0  - Total attempt count
    last_attempt_at   TEXT               - Last attempt timestamp
    last_success_at   TEXT               - Last success timestamp
    last_error_type   TEXT               - Type of last error
    last_error_message TEXT              - Error message (truncated to 2000 chars)
    completed_at      TEXT               - When the full pipeline completed

runs
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT
    started_at  TEXT NOT NULL
    completed_at TEXT
    run_type    TEXT NOT NULL     - discovery | download | whisper | clean | embed
    status      TEXT DEFAULT 'running' - running | complete | failed
    details     TEXT              - JSON details about the run

INDEXES
    idx_videos_ingest_status
    idx_videos_audio_status
    idx_videos_whisper_status
    idx_videos_embedding_status
    idx_videos_upload_date

ENVIRONMENT VARIABLES
    SQLITE_DB_PATH    Database path (default: .state/youtube_ingest.sqlite)

EXIT CODES
    0  Database initialized successfully.
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
    title TEXT NOT NULL,
    upload_date TEXT,
    duration INTEGER,
    title_filters TEXT,

    discovered_at TEXT NOT NULL,
    last_checked_at TEXT,

    ingest_status TEXT NOT NULL DEFAULT 'queued',
    audio_status TEXT NOT NULL DEFAULT 'not_started',
    whisper_status TEXT NOT NULL DEFAULT 'not_started',
    embedding_status TEXT NOT NULL DEFAULT 'not_started',

    local_dir TEXT,
    audio_path TEXT,
    transcript_path TEXT,
    clean_transcript_status TEXT NOT NULL DEFAULT 'not_started',
    clean_transcript_path TEXT,
    chroma_collection TEXT,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    embedded_at TEXT,

    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_attempt_at TEXT,
    last_success_at TEXT,
    last_error_type TEXT,
    last_error_message TEXT,

    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_videos_ingest_status
ON videos (ingest_status);

CREATE INDEX IF NOT EXISTS idx_videos_audio_status
ON videos (audio_status);

CREATE INDEX IF NOT EXISTS idx_videos_whisper_status
ON videos (whisper_status);

CREATE INDEX IF NOT EXISTS idx_videos_embedding_status
ON videos (embedding_status);

CREATE INDEX IF NOT EXISTS idx_videos_upload_date
ON videos (upload_date);

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
