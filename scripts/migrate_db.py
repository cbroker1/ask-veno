#!/usr/bin/env python3
"""
migrate_db.py -- Apply lightweight SQLite schema migrations.

This keeps an existing local .state/youtube_ingest.sqlite database compatible
as the repo evolves. It adds missing columns using ALTER TABLE ADD COLUMN,
skipping columns that already exist.

USAGE
-----
    python scripts/migrate_db.py

OPTIONS
    (none)  The script applies all known migrations and exits.
            Set SQLITE_DB_PATH in .env or environment to change the path.

EXAMPLES
    # Migrate the default database
    python scripts/migrate_db.py

    # Migrate a custom database
    SQLITE_DB_PATH=/path/to/db.sqlite python scripts/migrate_db.py

MIGRATIONS APPLIED
------------------
The following columns are added to the videos table if missing:

    clean_transcript_status   TEXT NOT NULL DEFAULT 'not_started'
    clean_transcript_path     TEXT
    chunk_count               INTEGER NOT NULL DEFAULT 0
    embedded_at               TEXT

ENVIRONMENT VARIABLES
    SQLITE_DB_PATH    Database path (default: .state/youtube_ingest.sqlite)

EXIT CODES
    0  Migration completed successfully.
    1  Database does not exist yet (run init_db.py first).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"


def get_db_path() -> Path:
    load_dotenv()
    return Path(os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH))


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    if column_exists(conn, table, column):
        print(f"Column already exists: {table}.{column}")
        return

    sql = f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
    print(f"Applying: {sql}")
    conn.execute(sql)


def main() -> int:
    db_path = get_db_path()

    if not db_path.exists():
        print(f"DB does not exist yet: {db_path}")
        print("Run: python scripts/init_db.py")
        return 1

    with sqlite3.connect(db_path) as conn:
        add_column_if_missing(
            conn,
            "videos",
            "clean_transcript_status",
            "TEXT NOT NULL DEFAULT 'not_started'",
        )
        add_column_if_missing(
            conn,
            "videos",
            "clean_transcript_path",
            "TEXT",
        )
        add_column_if_missing(
            conn,
            "videos",
            "chunk_count",
            "INTEGER NOT NULL DEFAULT 0",
        )
        add_column_if_missing(
            conn,
            "videos",
            "embedded_at",
            "TEXT",
        )
        conn.commit()

    print(f"Migration complete: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
