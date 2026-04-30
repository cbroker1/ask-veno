#!/usr/bin/env python3
"""
Process audio files through Whisper.

This script is based on the original notebook workflow:

- Load Whisper once
- Use model.transcribe(..., word_timestamps=True)
- Write transcript JSON into the same folder as the MP3
- Name the transcript after the folder:
    <folder_name> transcript.json

Unlike the notebook, this script is SQLite-state driven:

- Select audio_ready videos with audio_status=downloaded
- Skip videos already transcribed
- Update whisper_status and transcript_path
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

import whisper
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


console = Console()

DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"


@dataclass
class Config:
    db_path: Path
    max_transcribe_videos: int
    whisper_model: str
    whisper_device: str
    word_timestamps: bool
    dry_run: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config(args: argparse.Namespace) -> Config:
    load_dotenv()

    max_transcribe = args.max_transcribe_videos
    if max_transcribe is None:
        max_transcribe = int(os.getenv("MAX_TRANSCRIBE_VIDEOS", "1"))

    return Config(
        db_path=Path(args.db_path or os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)),
        max_transcribe_videos=max_transcribe,
        whisper_model=args.whisper_model or os.getenv("WHISPER_MODEL", "large"),
        whisper_device=args.whisper_device or os.getenv("WHISPER_DEVICE", "cuda:0"),
        word_timestamps=parse_bool(
            args.word_timestamps
            if args.word_timestamps is not None
            else os.getenv("WHISPER_WORD_TIMESTAMPS"),
            default=True,
        ),
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


def get_audio_ready_videos(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT *
        FROM videos
        WHERE ingest_status = 'audio_ready'
          AND audio_status = 'downloaded'
          AND whisper_status != 'transcribed'
          AND audio_path IS NOT NULL
        ORDER BY last_success_at ASC, discovered_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return list(rows)


def transcript_output_path(audio_path: Path) -> Path:
    folder_path = audio_path.parent
    folder_name = folder_path.name
    return folder_path / f"{folder_name} transcript.json"


def print_queue(rows: list[sqlite3.Row]) -> None:
    table = Table(title=f"Audio files to transcribe: {len(rows)}")
    table.add_column("Video ID", no_wrap=True)
    table.add_column("Whisper")
    table.add_column("Audio Path")

    for row in rows:
        table.add_row(
            row["video_id"],
            row["whisper_status"],
            str(row["audio_path"])[:120],
        )

    console.print(table)


def mark_transcribed(
    conn: sqlite3.Connection,
    video_id: str,
    transcript_path: Path,
) -> None:
    conn.execute(
        """
        UPDATE videos
        SET
            ingest_status = 'transcript_ready',
            whisper_status = 'transcribed',
            transcript_path = ?,
            last_success_at = ?,
            last_error_type = NULL,
            last_error_message = NULL
        WHERE video_id = ?
        """,
        (
            str(transcript_path),
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
            ingest_status = 'failed_whisper',
            whisper_status = 'failed',
            attempt_count = attempt_count + 1,
            last_attempt_at = ?,
            last_error_type = 'whisper',
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


def transcribe_audio(audio_path: Path, model: Any, word_timestamps: bool) -> dict[str, Any]:
    result = model.transcribe(str(audio_path), word_timestamps=word_timestamps)
    if not isinstance(result, dict):
        raise RuntimeError("Whisper returned unexpected non-dict result.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path")
    parser.add_argument("--max-transcribe-videos", type=int)
    parser.add_argument("--whisper-model")
    parser.add_argument("--whisper-device")
    parser.add_argument("--word-timestamps", choices=["true", "false"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        config = load_config(parse_args())
        conn = connect_db(config.db_path)
    except Exception as exc:
        console.print(f"[red]Startup failed:[/red] {exc}")
        return 2

    console.print("[bold]Whisper queue processor config[/bold]")
    console.print(f"DB path: {config.db_path}")
    console.print(f"Max transcribe videos: {config.max_transcribe_videos}")
    console.print(f"Whisper model: {config.whisper_model}")
    console.print(f"Whisper device: {config.whisper_device}")
    console.print(f"Word timestamps: {config.word_timestamps}")
    console.print(f"Dry run: {config.dry_run}")
    console.print()

    rows = get_audio_ready_videos(conn, config.max_transcribe_videos)
    print_queue(rows)

    if not rows:
        console.print("[green]No audio_ready videos to transcribe.[/green]")
        return 0

    if config.dry_run:
        console.print("[yellow]Dry run enabled. No transcription performed.[/yellow]")
        return 0

    console.print("[bold]Loading Whisper model...[/bold]")
    model = whisper.load_model(config.whisper_model, device=config.whisper_device)
    console.print("[green]Whisper model loaded.[/green]")

    for row in rows:
        video_id = row["video_id"]
        audio_path = Path(row["audio_path"])
        transcript_path = transcript_output_path(audio_path)

        console.rule(f"Transcribing {video_id}")

        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

            if transcript_path.exists():
                console.print(f"[yellow]Transcript already exists, marking transcribed:[/yellow] {transcript_path}")
                mark_transcribed(conn, video_id, transcript_path)
                continue

            console.print(f"Audio: {audio_path}")
            console.print(f"Transcript output: {transcript_path}")

            result = transcribe_audio(audio_path, model, config.word_timestamps)

            transcript_path.write_text(
                json.dumps(result, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )

            mark_transcribed(conn, video_id, transcript_path)
            console.print(f"[bold green]Complete:[/bold green] {transcript_path}")

        except Exception as exc:
            message = str(exc)
            mark_failed(conn, video_id, message)
            console.print(f"[red]Failed {video_id}:[/red] {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
