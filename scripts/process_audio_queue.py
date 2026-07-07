#!/usr/bin/env python3
"""
process_audio_queue.py -- Download audio from queued YouTube videos.

This script mirrors the useful parts of the original notebook:

- Select videos already discovered into SQLite.
- Fetch full metadata only for selected queued videos (fast, avoids pulling
  metadata for every channel item).
- Create a local folder using upload_date + title + video_id.
- Download audio as MP3 (or configured format).
- Save youtube_url_<video_id>.txt with the video URL.
- Save video_metadata_<video_id>.json with enriched metadata.
- Update SQLite with local_dir, audio_path, upload_date, and status.

It does not run Whisper.
It does not create ChromaDB embeddings.

USAGE
-----
    python scripts/process_audio_queue.py

OPTIONS
    --db-path PATH                    SQLite database path
                                      (default: .state/youtube_ingest.sqlite)
    --audio-root PATH                 Root directory for downloaded audio
                                      (default: data/raw/youtube)
    --max-new-videos N                Number of videos to process per run
                                      (default: 1)
    --audio-format FORMAT             Audio format: mp3, m4a, etc.
                                      (default: mp3)
    --browser BROWSER                 Browser for cookies (chrome, firefox, etc.)
                                      (default: chrome)
    --cookies-txt-path PATH           Path to cookies.txt file
    --use-browser-cookies true|false  Use browser cookies for authentication
                                      (default: false)
    --use-cookies-txt-fallback true|false  Fall back to cookies.txt if browser
                                           cookies fail (default: false)
    --dry-run                         Print queue and exit without downloading

EXAMPLES
    # Download 1 queued video with defaults
    python scripts/process_audio_queue.py

    # Download 3 videos using m4a format and browser cookies
    python scripts/process_audio_queue.py \
        --max-new-videos 3 \
        --audio-format m4a \
        --use-browser-cookies true

    # Dry run to see what would be downloaded
    python scripts/process_audio_queue.py --dry-run

WHAT IT DOES
------------
1. Queries the database for videos with ingest_status in
   ('queued', 'failed_ytdlp', 'failed_network', 'failed_auth') and
   audio_status != 'downloaded'.
2. For each video:
   - Fetches full metadata via yt-dlp (tries cookie modes: none, browser, cookies_txt).
   - Creates a local folder: <audio_root>/<upload_date>_<title>_<video_id>.
   - Downloads audio using yt-dlp (tries cookie modes sequentially).
   - Writes sidecar files: youtube_url_<id>.txt, video_metadata_<id>.json.
   - Updates the video row: ingest_status='audio_ready', audio_status='downloaded'.
3. Classifies errors (auth, network, ffmpeg) for targeted retry logic.
4. Handles failures by recording error type and message.

ENVIRONMENT VARIABLES
    SQLITE_DB_PATH                Database path (default: .state/youtube_ingest.sqlite)
    YOUTUBE_AUDIO_ROOT            Audio root directory (default: data/raw/youtube)
    MAX_NEW_VIDEOS                Max videos per run (default: 1)
    AUDIO_FORMAT                  Audio format (default: mp3)
    USE_BROWSER_COOKIES           Use browser cookies (default: false)
    YTDLP_BROWSER                 Browser for cookies (default: chrome)
    USE_COOKIES_TXT_FALLBACK      Use cookies.txt fallback (default: false)
    COOKIES_TXT_PATH              Path to cookies.txt (default: cookies.txt)
    YTDLP_USE_DENO                Use Deno for yt-dlp JS runtime (default: false)
    YTDLP_DENO_PATH               Path to Deno binary

EXIT CODES
    0  Download completed (all or no videos).
    1  One or more videos failed download.
    2  Startup error (config or DB failure).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yt_dlp
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


console = Console()

DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"
DEFAULT_AUDIO_ROOT = "data/raw/youtube"


@dataclass
class Config:
    db_path: Path
    audio_root: Path
    max_new_videos: int
    audio_format: str
    use_browser_cookies: bool
    browser: str
    use_cookies_txt_fallback: bool
    cookies_txt_path: Path
    use_deno: bool
    deno_path: str | None
    dry_run: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_filename(value: str, max_len: int = 140) -> str:
    value = value.strip()
    value = re.sub(r'[<>:"/\\|?*\n\r\t]', " ", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" .")
    if len(value) > max_len:
        value = value[:max_len].rstrip(" .")
    return value or "untitled"


def load_config(args: argparse.Namespace) -> Config:
    load_dotenv()

    max_new_videos = args.max_new_videos
    if max_new_videos is None:
        max_new_videos = int(os.getenv("MAX_NEW_VIDEOS", "1"))

    use_browser_cookies = parse_bool(
        args.use_browser_cookies
        if args.use_browser_cookies is not None
        else os.getenv("USE_BROWSER_COOKIES"),
        default=False,
    )

    use_cookies_txt_fallback = parse_bool(
        args.use_cookies_txt_fallback
        if args.use_cookies_txt_fallback is not None
        else os.getenv("USE_COOKIES_TXT_FALLBACK"),
        default=False,
    )

    return Config(
        db_path=Path(args.db_path or os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)),
        audio_root=Path(args.audio_root or os.getenv("YOUTUBE_AUDIO_ROOT", DEFAULT_AUDIO_ROOT)),
        max_new_videos=max_new_videos,
        audio_format=args.audio_format or os.getenv("AUDIO_FORMAT", "mp3"),
        use_browser_cookies=use_browser_cookies,
        browser=args.browser or os.getenv("YTDLP_BROWSER", "chrome"),
        use_cookies_txt_fallback=use_cookies_txt_fallback,
        cookies_txt_path=Path(args.cookies_txt_path or os.getenv("COOKIES_TXT_PATH", "cookies.txt")),
        use_deno=parse_bool(os.getenv("YTDLP_USE_DENO"), default=False),
        deno_path=os.getenv("YTDLP_DENO_PATH") or None,
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


def get_queued_videos(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    rows = conn.execute(
        """
        SELECT *
        FROM videos
        WHERE ingest_status IN ('queued', 'failed_ytdlp', 'failed_network', 'failed_auth')
          AND audio_status != 'downloaded'
        ORDER BY discovered_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return list(rows)


def classify_error(message: str) -> tuple[str, str]:
    lower = message.lower()

    if any(token in lower for token in ["sign in", "cookies", "bot", "confirm you", "authentication"]):
        return "failed_auth", "auth_or_cookie"

    if any(token in lower for token in ["network", "timed out", "temporary failure", "connection reset"]):
        return "failed_network", "network"

    if any(token in lower for token in ["ffmpeg", "postprocessing"]):
        return "failed_ytdlp", "ffmpeg_or_postprocess"

    return "failed_ytdlp", "yt_dlp"


def update_video_error(
    conn: sqlite3.Connection,
    video_id: str,
    ingest_status: str,
    error_type: str,
    error_message: str,
) -> None:
    conn.execute(
        """
        UPDATE videos
        SET
            ingest_status = ?,
            audio_status = 'failed',
            attempt_count = attempt_count + 1,
            last_attempt_at = ?,
            last_error_type = ?,
            last_error_message = ?
        WHERE video_id = ?
        """,
        (
            ingest_status,
            utc_now(),
            error_type,
            error_message[:2000],
            video_id,
        ),
    )
    conn.commit()


def ydl_metadata_opts(config: Config, cookie_mode: str) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "skip_download": True,
        "ignoreerrors": False,
    }

    if cookie_mode == "browser":
        opts["cookiesfrombrowser"] = (config.browser,)
    elif cookie_mode == "cookies_txt":
        opts["cookiefile"] = str(config.cookies_txt_path)

    return opts


def fetch_metadata(video_url: str, config: Config) -> dict[str, Any]:
    attempts: list[tuple[str, dict[str, Any]]] = [
        ("none", ydl_metadata_opts(config, "none"))
    ]

    if config.use_browser_cookies:
        attempts.append(("browser", ydl_metadata_opts(config, "browser")))

    if config.use_cookies_txt_fallback and config.cookies_txt_path.exists():
        attempts.append(("cookies_txt", ydl_metadata_opts(config, "cookies_txt")))

    last_error: Exception | None = None

    for mode, opts in attempts:
        try:
            console.print(f"Fetching full metadata using cookie mode: [cyan]{mode}[/cyan]")
            with yt_dlp.YoutubeDL(opts) as ydl:
                data = ydl.extract_info(video_url, download=False)
                if not isinstance(data, dict):
                    raise RuntimeError("yt-dlp returned unexpected metadata response.")
                return data
        except Exception as exc:
            last_error = exc
            console.print(f"[yellow]Metadata attempt failed with mode {mode}: {exc}[/yellow]")

    raise RuntimeError(f"All metadata attempts failed. Last error: {last_error}")


def build_local_dir(video: sqlite3.Row, metadata: dict[str, Any], config: Config) -> Path:
    video_id = str(video["video_id"])
    title = safe_filename(str(metadata.get("title") or video["title"]))
    upload_date = str(metadata.get("upload_date") or "unknown_date")
    folder_name = safe_filename(f"{upload_date} {title} {video_id}", max_len=180)
    return config.audio_root / folder_name


def build_ytdlp_command(video_url: str, output_template: Path, config: Config, cookie_mode: str) -> list[str]:
    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        config.audio_format,
        "--no-playlist",
        "-o",
        str(output_template),
    ]

    if cookie_mode == "browser":
        cmd.extend(["--cookies-from-browser", config.browser])
    elif cookie_mode == "cookies_txt":
        cmd.extend(["--cookies", str(config.cookies_txt_path)])

    if config.use_deno and config.deno_path:
        cmd.extend(
            [
                "--js-runtimes",
                f"deno:{config.deno_path}",
                "--remote-components",
                "ejs:github",
            ]
        )

    cmd.append(video_url)
    return cmd


def run_download(video_url: str, local_dir: Path, config: Config) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)

    output_template = local_dir / "%(upload_date)s %(title).160B %(id)s.%(ext)s"

    attempts: list[str] = ["none"]

    if config.use_browser_cookies:
        attempts.append("browser")

    if config.use_cookies_txt_fallback and config.cookies_txt_path.exists():
        attempts.append("cookies_txt")

    last_error = ""

    for mode in attempts:
        cmd = build_ytdlp_command(video_url, output_template, config, mode)

        console.print(f"Running audio download using cookie mode: [cyan]{mode}[/cyan]")
        console.print(" ".join(cmd))

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            audio_files = sorted(local_dir.glob(f"*.{config.audio_format}"))
            if not audio_files:
                raise RuntimeError("yt-dlp succeeded but no audio file was found.")
            return audio_files[-1]

        last_error = (result.stderr or result.stdout or "").strip()
        console.print(f"[yellow]Download failed with mode {mode}:[/yellow]")
        console.print(last_error[-2000:])

    raise RuntimeError(last_error or "Audio download failed for unknown reason.")


def write_sidecar_files(
    local_dir: Path,
    video_id: str,
    video_url: str,
    metadata: dict[str, Any],
    audio_path: Path,
) -> Path:
    url_path = local_dir / f"youtube_url_{video_id}.txt"
    url_path.write_text(video_url + "\n", encoding="utf-8")

    safe_metadata = {
        "video_id": video_id,
        "title": metadata.get("title"),
        "url": video_url,
        "webpage_url": metadata.get("webpage_url"),
        "upload_date": metadata.get("upload_date"),
        "duration": metadata.get("duration"),
        "uploader": metadata.get("uploader"),
        "channel": metadata.get("channel"),
        "channel_id": metadata.get("channel_id"),
        "channel_url": metadata.get("channel_url"),
        "audio_path": str(audio_path),
        "local_dir": str(local_dir),
        "metadata_captured_at": utc_now(),
    }

    metadata_path = local_dir / f"video_metadata_{video_id}.json"
    metadata_path.write_text(
        json.dumps(safe_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata_path


def mark_audio_downloaded(
    conn: sqlite3.Connection,
    video_id: str,
    metadata: dict[str, Any],
    local_dir: Path,
    audio_path: Path,
) -> None:
    conn.execute(
        """
        UPDATE videos
        SET
            title = COALESCE(?, title),
            upload_date = COALESCE(?, upload_date),
            duration = COALESCE(?, duration),
            ingest_status = 'audio_ready',
            audio_status = 'downloaded',
            local_dir = ?,
            audio_path = ?,
            attempt_count = attempt_count + 1,
            last_attempt_at = ?,
            last_success_at = ?,
            last_error_type = NULL,
            last_error_message = NULL
        WHERE video_id = ?
        """,
        (
            metadata.get("title"),
            metadata.get("upload_date"),
            metadata.get("duration"),
            str(local_dir),
            str(audio_path),
            utc_now(),
            utc_now(),
            video_id,
        ),
    )
    conn.commit()


def print_queue(rows: list[sqlite3.Row]) -> None:
    table = Table(title=f"Queued videos to process: {len(rows)}")
    table.add_column("Video ID", no_wrap=True)
    table.add_column("Status")
    table.add_column("Audio")
    table.add_column("Title")

    for row in rows:
        table.add_row(
            row["video_id"],
            row["ingest_status"],
            row["audio_status"],
            row["title"][:100],
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path")
    parser.add_argument("--audio-root")
    parser.add_argument("--max-new-videos", type=int)
    parser.add_argument("--audio-format")
    parser.add_argument("--browser")
    parser.add_argument("--cookies-txt-path")
    parser.add_argument("--use-browser-cookies", choices=["true", "false"])
    parser.add_argument("--use-cookies-txt-fallback", choices=["true", "false"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        config = load_config(parse_args())
        config.audio_root.mkdir(parents=True, exist_ok=True)
        conn = connect_db(config.db_path)
    except Exception as exc:
        console.print(f"[red]Startup failed:[/red] {exc}")
        return 2

    console.print("[bold]Audio queue processor config[/bold]")
    console.print(f"DB path: {config.db_path}")
    console.print(f"Audio root: {config.audio_root}")
    console.print(f"Max new videos: {config.max_new_videos}")
    console.print(f"Audio format: {config.audio_format}")
    console.print(f"Use browser cookies: {config.use_browser_cookies}")
    console.print(f"Browser: {config.browser}")
    console.print(f"Cookies.txt fallback: {config.use_cookies_txt_fallback}")
    console.print(f"Use Deno: {config.use_deno}")
    console.print(f"Dry run: {config.dry_run}")
    console.print()

    rows = get_queued_videos(conn, config.max_new_videos)
    print_queue(rows)

    if not rows:
        console.print("[green]No queued videos to process.[/green]")
        return 0

    if config.dry_run:
        console.print("[yellow]Dry run enabled. No downloads performed.[/yellow]")
        return 0

    for row in rows:
        video_id = row["video_id"]
        video_url = row["url"]

        console.rule(f"Processing {video_id}")

        try:
            metadata = fetch_metadata(video_url, config)
            local_dir = build_local_dir(row, metadata, config)

            console.print(f"Local dir: {local_dir}")

            audio_path = run_download(video_url, local_dir, config)
            write_sidecar_files(local_dir, video_id, video_url, metadata, audio_path)
            mark_audio_downloaded(conn, video_id, metadata, local_dir, audio_path)

            console.print(f"[bold green]Downloaded audio:[/bold green] {audio_path}")

        except Exception as exc:
            message = str(exc)
            ingest_status, error_type = classify_error(message)
            update_video_error(conn, video_id, ingest_status, error_type, message)
            console.print(f"[red]Failed {video_id}:[/red] {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
