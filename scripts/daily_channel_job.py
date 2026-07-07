#!/usr/bin/env python3
"""
daily_channel_job.py -- Daily YouTube RAG channel job.

Behavior:
1. Run exhaustive YouTube discovery.
2. Detect whether brand-new videos were inserted into SQLite.
3. If no new videos were found, exit quietly (exit code 0).
4. If new videos were found, process the full queue:
   - download audio
   - Whisper transcription
   - transcript cleanup
   - ChromaDB embedding
   - print final pipeline status

This is intended to be run once per day by a systemd user timer.

USAGE
-----
    python scripts/daily_channel_job.py

OPTIONS
    (none)  All configuration comes from environment variables or .env.

EXAMPLES
    # Run the daily job
    python scripts/daily_channel_job.py

    # Run with custom discovery limit
    DAILY_DISCOVERY_MAX_VIDEOS=100 python scripts/daily_channel_job.py

WHAT IT DOES
------------
1. Acquires a file lock (.state/daily_job.lock) to prevent concurrent runs.
2. Records the start time and current video count.
3. Runs discover_audio_candidates.py with exhaustive mode
   (--max-discovery-vectors from DAILY_DISCOVERY_MAX_VIDEOS, default 0 = unlimited).
4. Queries for new videos discovered since the start time.
5. If no new videos: exits silently (exit code 0).
6. If new videos found:
   - Logs each new video (ID, upload date, title).
   - Runs the full queue pipeline (download, whisper, clean, embed).
   - Prints the final pipeline status.
7. Releases the file lock on exit.

ENVIRONMENT VARIABLES
    SQLITE_DB_PATH                Database path (default: .state/youtube_ingest.sqlite)
    DAILY_DISCOVERY_MAX_VIDEOS    Max videos for discovery (default: 0 = unlimited)
    All variables accepted by the individual queue processor scripts
    (discover_audio_candidates.py, process_audio_queue.py, etc.)

EXIT CODES
    0  No new videos or job completed successfully.
    1  Discovery failed or a pipeline step failed.

SYSTEMD TIMER EXAMPLE
---------------------
Create /etc/systemd/user/daily-rag.timer:

    [Unit]
    Description=Daily YouTube RAG job

    [Timer]
    OnCalendar=daily
    Persistent=true

    [Install]
    WantedBy=timers.target

Create /etc/systemd/user/daily-rag.service:

    [Unit]
    Description=Daily YouTube RAG job

    [Service]
    Type=oneshot
    WorkingDirectory=/path/to/ask-veno
    ExecStart=/path/to/.venv/bin/python scripts/daily_channel_job.py

Enable with:
    systemctl --user enable --now daily-rag.timer

"""

from __future__ import annotations

import fcntl
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def run_step(name: str, command: list[str]) -> int:
    log("=" * 100)
    log(f"STEP: {name}")
    log("$ " + " ".join(command))
    log("=" * 100)

    result = subprocess.run(command, cwd=ROOT, check=False)

    if result.returncode == 0:
        log(f"OK: {name}")
    else:
        log(f"FAILED: {name} exit_code={result.returncode}")

    return result.returncode


def db_path_from_env() -> Path:
    db_path = Path(os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH))
    if not db_path.is_absolute():
        db_path = ROOT / db_path
    return db_path


def connect_db() -> sqlite3.Connection:
    db_path = db_path_from_env()
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def count_videos(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM videos").fetchone()
    return int(row["count"])


def get_new_videos(conn: sqlite3.Connection, start_time: str) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT video_id, title, upload_date, ingest_status
            FROM videos
            WHERE discovered_at >= ?
            ORDER BY discovered_at ASC
            """,
            (start_time,),
        ).fetchall()
    )


def main() -> int:
    load_dotenv(dotenv_path=ROOT / ".env")

    lock_path = ROOT / ".state" / "daily_job.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("Another daily job is already running. Exiting.")
            return 0

        start_time = utc_now()
        log(f"Daily job started at {start_time}")
        log(f"Repo root: {ROOT}")

        py = sys.executable
        max_discovery = os.getenv("DAILY_DISCOVERY_MAX_VIDEOS", "0")

        conn = connect_db()
        before_count = count_videos(conn)
        log(f"Video count before discovery: {before_count}")

        code = run_step(
            "Exhaustive discovery",
            [
                py,
                "scripts/discover_audio_candidates.py",
                "--max-discovery-videos",
                str(max_discovery),
                "--use-browser-cookies",
                "false",
            ],
        )

        if code != 0:
            log("Discovery failed. Stopping daily job.")
            return code

        conn = connect_db()
        after_count = count_videos(conn)
        new_videos = get_new_videos(conn, start_time)

        log(f"Video count after discovery: {after_count}")
        log(f"New videos discovered this run: {len(new_videos)}")

        if not new_videos:
            log("No new videos discovered. Exiting quietly.")
            return 0

        for row in new_videos:
            log(f"NEW: {row['video_id']} | {row['upload_date']} | {row['title']}")

        steps = [
            (
                "Download queued audio",
                [
                    py,
                    "scripts/process_audio_queue.py",
                    "--max-new-videos",
                    "999",
                    "--use-browser-cookies",
                    "false",
                ],
            ),
            (
                "Whisper transcription",
                [
                    py,
                    "scripts/process_whisper_queue.py",
                    "--max-transcribe-videos",
                    "999",
                ],
            ),
            (
                "Transcript cleanup",
                [
                    py,
                    "scripts/process_transcript_queue.py",
                    "--max-clean-videos",
                    "999",
                ],
            ),
            (
                "ChromaDB embedding",
                [
                    py,
                    "scripts/process_chromadb_queue.py",
                    "--max-embed-videos",
                    "999",
                ],
            ),
            (
                "Final pipeline status",
                [
                    py,
                    "scripts/pipeline_status.py",
                ],
            ),
        ]

        for name, command in steps:
            code = run_step(name, command)
            if code != 0:
                log(f"Stopping after failed step: {name}")
                return code

        log("Daily job completed successfully.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
