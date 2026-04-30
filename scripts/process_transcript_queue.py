#!/usr/bin/env python3
"""
Clean raw Whisper transcript JSON into a simplified transcript_clean.json file.

This script is based on the existing transcript post-processing notebook.

Input:
  <folder_name> transcript.json

Output:
  <folder_name> transcript_clean.json

Each cleaned segment keeps:
  - start
  - end
  - text

The script is SQLite-state driven:
  transcript_ready + transcribed + not cleaned
    -> transcript_clean_ready + cleaned
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


console = Console()

DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"


@dataclass
class Config:
    db_path: Path
    max_clean_videos: int
    dry_run: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config(args: argparse.Namespace) -> Config:
    load_dotenv()

    max_clean = args.max_clean_videos
    if max_clean is None:
        max_clean = int(os.getenv("MAX_CLEAN_TRANSCRIPTS", "1"))

    return Config(
        db_path=Path(args.db_path or os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)),
        max_clean_videos=max_clean,
        dry_run=args.dry_run,
    )


def connect_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite DB not found at {db_path}. Run python scripts/init_db.py first."
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_transcript_ready_videos(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT *
        FROM videos
        WHERE ingest_status = 'transcript_ready'
          AND whisper_status = 'transcribed'
          AND transcript_path IS NOT NULL
          AND clean_transcript_status != 'cleaned'
        ORDER BY last_success_at ASC, discovered_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return list(rows)


def clean_text(value: str) -> str:
    return " ".join(value.replace("\n", " ").replace("\r", " ").split()).strip()


def clean_segments(raw_transcript: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = raw_transcript.get("segments", [])
    cleaned: list[dict[str, Any]] = []

    for segment in raw_segments:
        text = clean_text(str(segment.get("text", "")))
        if not text:
            continue

        start = segment.get("start")
        end = segment.get("end")

        if start is None or end is None:
            continue

        cleaned.append(
            {
                "start": float(start),
                "end": float(end),
                "text": text,
            }
        )

    cleaned.sort(key=lambda item: item["start"])
    return cleaned


def clean_output_path(raw_transcript_path: Path) -> Path:
    folder = raw_transcript_path.parent
    folder_name = folder.name
    return folder / f"{folder_name} transcript_clean.json"


def mark_cleaned(
    conn: sqlite3.Connection,
    video_id: str,
    clean_path: Path,
) -> None:
    conn.execute(
        """
        UPDATE videos
        SET
            ingest_status = 'transcript_clean_ready',
            clean_transcript_status = 'cleaned',
            clean_transcript_path = ?,
            last_success_at = ?,
            last_error_type = NULL,
            last_error_message = NULL
        WHERE video_id = ?
        """,
        (
            str(clean_path),
            utc_now(),
            video_id,
        ),
    )
    conn.commit()


def mark_failed(
    conn: sqlite3.Connection,
    video_id: str,
    error_message: str,
) -> None:
    conn.execute(
        """
        UPDATE videos
        SET
            ingest_status = 'failed_transcript_clean',
            clean_transcript_status = 'failed',
            attempt_count = attempt_count + 1,
            last_attempt_at = ?,
            last_error_type = 'transcript_clean',
            last_error_message = ?
        WHERE video_id = ?
        """,
        (
            utc_now(),
            error_message[:2000],
            video_id,
        ),
    )
    conn.commit()


def print_queue(rows: list[sqlite3.Row]) -> None:
    table = Table(title=f"Transcripts to clean: {len(rows)}")
    table.add_column("Video ID", no_wrap=True)
    table.add_column("Ingest")
    table.add_column("Whisper")
    table.add_column("Clean")
    table.add_column("Transcript Path")

    for row in rows:
        table.add_row(
            row["video_id"],
            row["ingest_status"],
            row["whisper_status"],
            row["clean_transcript_status"],
            str(row["transcript_path"])[:100],
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path")
    parser.add_argument("--max-clean-videos", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        config = load_config(parse_args())
        conn = connect_db(config.db_path)
    except Exception as exc:
        console.print(f"[red]Startup failed:[/red] {exc}")
        return 2

    console.print("[bold]Transcript cleanup processor config[/bold]")
    console.print(f"DB path: {config.db_path}")
    console.print(f"Max clean videos: {config.max_clean_videos}")
    console.print(f"Dry run: {config.dry_run}")
    console.print()

    rows = get_transcript_ready_videos(conn, config.max_clean_videos)
    print_queue(rows)

    if not rows:
        console.print("[green]No transcript_ready videos to clean.[/green]")
        return 0

    if config.dry_run:
        console.print("[yellow]Dry run enabled. No cleanup performed.[/yellow]")
        return 0

    for row in rows:
        video_id = row["video_id"]
        raw_path = Path(row["transcript_path"])
        clean_path = clean_output_path(raw_path)

        console.rule(f"Cleaning transcript {video_id}")

        try:
            if not raw_path.exists():
                raise FileNotFoundError(f"Raw transcript does not exist: {raw_path}")

            raw_transcript = json.loads(raw_path.read_text(encoding="utf-8"))
            cleaned_segments = clean_segments(raw_transcript)

            if not cleaned_segments:
                raise RuntimeError("No cleaned transcript segments were produced.")

            clean_path.write_text(
                json.dumps(cleaned_segments, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )

            mark_cleaned(conn, video_id, clean_path)

            console.print(f"[bold green]Clean transcript written:[/bold green] {clean_path}")
            console.print(f"Segments: {len(cleaned_segments)}")

        except Exception as exc:
            message = str(exc)
            mark_failed(conn, video_id, message)
            console.print(f"[red]Failed {video_id}:[/red] {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
