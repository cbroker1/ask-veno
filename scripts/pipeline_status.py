#!/usr/bin/env python3
"""
pipeline_status.py -- Show the current YouTube RAG pipeline status from SQLite.

Displays:
- Counts by ingest_status, audio_status, whisper_status, clean_transcript_status,
  embedding_status (grouped status tables).
- Recent videos (last N by discovered_at) with all status columns.
- Failed videos (any with last_error_type set or ingest_status starting with 'failed').

USAGE
-----
    python scripts/pipeline_status.py

OPTIONS
    --db-path PATH        SQLite database path
                          (default: .state/youtube_ingest.sqlite)
    --limit N             Number of recent videos to show (default: 20)

EXAMPLES
    # Show default status overview
    python scripts/pipeline_status.py

    # Show last 50 recent videos
    python scripts/pipeline_status.py --limit 50

    # Use a custom database
    python scripts/pipeline_status.py --db-path /path/to/db.sqlite

WHAT IT DISPLAYS
----------------
1. Counts by status -- for each of the 5 status columns, shows how many
   videos are in each state (e.g., how many are 'queued', 'downloaded', etc.).
2. Recent videos -- the N most recently discovered videos with all status
   columns and chunk count.
3. Failures -- all videos with errors, showing the error type and message.

ENVIRONMENT VARIABLES
    SQLITE_DB_PATH    Database path (default: .state/youtube_ingest.sqlite)

EXIT CODES
    0  Status displayed successfully.
    2  Startup error (config or DB failure).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


console = Console()

DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"


def load_db_path(args: argparse.Namespace) -> Path:
    load_dotenv(dotenv_path=Path(".env"))
    return Path(args.db_path or os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH))


def connect_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found at {db_path}. Run python scripts/init_db.py first.")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def print_status_counts(conn: sqlite3.Connection, column: str) -> None:
    rows = conn.execute(
        f"""
        SELECT {column} AS status, COUNT(*) AS count
        FROM videos
        GROUP BY {column}
        ORDER BY count DESC, status ASC
        """
    ).fetchall()

    table = Table(title=f"Counts by {column}")
    table.add_column("Status")
    table.add_column("Count", justify="right")

    for row in rows:
        table.add_row(str(row["status"]), str(row["count"]))

    console.print(table)


def print_recent_videos(conn: sqlite3.Connection, limit: int) -> None:
    rows = conn.execute(
        """
        SELECT
            video_id,
            ingest_status,
            audio_status,
            whisper_status,
            clean_transcript_status,
            embedding_status,
            chunk_count,
            title
        FROM videos
        ORDER BY discovered_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    table = Table(title=f"Recent videos, limit {limit}")
    table.add_column("Video ID", no_wrap=True)
    table.add_column("Ingest")
    table.add_column("Audio")
    table.add_column("Whisper")
    table.add_column("Clean")
    table.add_column("Embed")
    table.add_column("Chunks", justify="right")
    table.add_column("Title")

    for row in rows:
        table.add_row(
            str(row["video_id"]),
            str(row["ingest_status"]),
            str(row["audio_status"]),
            str(row["whisper_status"]),
            str(row["clean_transcript_status"]),
            str(row["embedding_status"]),
            str(row["chunk_count"]),
            str(row["title"])[:80],
        )

    console.print(table)


def print_failures(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT
            video_id,
            ingest_status,
            last_error_type,
            last_error_message,
            title
        FROM videos
        WHERE last_error_type IS NOT NULL
           OR ingest_status LIKE 'failed%'
        ORDER BY last_attempt_at DESC
        """
    ).fetchall()

    if not rows:
        console.print("[green]No failed videos found.[/green]")
        return

    table = Table(title="Failures")
    table.add_column("Video ID", no_wrap=True)
    table.add_column("Ingest")
    table.add_column("Error Type")
    table.add_column("Error Message")
    table.add_column("Title")

    for row in rows:
        table.add_row(
            str(row["video_id"]),
            str(row["ingest_status"]),
            str(row["last_error_type"]),
            str(row["last_error_message"] or "")[:120],
            str(row["title"])[:80],
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        db_path = load_db_path(args)
        conn = connect_db(db_path)
    except Exception as exc:
        console.print(f"[red]Startup failed:[/red] {exc}")
        return 2

    console.print(f"[bold]Pipeline DB:[/bold] {db_path}")
    console.print()

    for column in [
        "ingest_status",
        "audio_status",
        "whisper_status",
        "clean_transcript_status",
        "embedding_status",
    ]:
        print_status_counts(conn, column)
        console.print()

    print_recent_videos(conn, args.limit)
    console.print()
    print_failures(conn)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
