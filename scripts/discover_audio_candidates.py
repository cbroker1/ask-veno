#!/usr/bin/env python3
"""
Discover YouTube audio candidates and persist them into SQLite.

Workflow:

- Use yt-dlp with extract_flat=True for fast channel / streams discovery
- Scan a channel / streams URL
- Match videos by title substring
- Enrich only the matched videos with full metadata so fields like upload_date are populated
- Save video_id/title/url/upload_date/duration into a persistent SQLite registry

It does NOT download audio.
It does NOT run Whisper.
It does NOT embed anything into ChromaDB.
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

import yt_dlp
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


console = Console()

DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"


@dataclass
class Config:
    channel_url: str
    title_filters: list[str]
    db_path: Path
    max_discovery_videos: int | None
    use_browser_cookies: bool
    browser: str
    use_cookies_txt_fallback: bool
    cookies_txt_path: Path
    enrich_metadata: bool
    dry_run: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_bool(value: str | bool | None, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def split_filters(raw: str) -> list[str]:
    filters = [item.strip() for item in raw.split(",") if item.strip()]
    if not filters:
        raise ValueError("At least one title filter is required.")
    return filters


def load_config(args: argparse.Namespace) -> Config:
    load_dotenv()

    channel_url = args.channel_url or os.getenv("YOUTUBE_CHANNEL_URL", "").strip()
    if not channel_url:
        raise ValueError("Missing YOUTUBE_CHANNEL_URL. Add it to .env or pass --channel-url.")

    raw_filters = args.title_filters or os.getenv("TITLE_FILTERS", "").strip()
    if not raw_filters:
        raise ValueError("Missing TITLE_FILTERS. Add it to .env or pass --title-filters.")

    db_path = Path(args.db_path or os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH))

    max_raw = args.max_discovery_videos
    if max_raw is None:
        env_max = os.getenv("MAX_DISCOVERY_VIDEOS", "").strip()
        max_raw = int(env_max) if env_max else None

    use_browser_cookies = parse_bool(
        args.use_browser_cookies
        if args.use_browser_cookies is not None
        else os.getenv("USE_BROWSER_COOKIES"),
        default=True,
    )

    browser = args.browser or os.getenv("YTDLP_BROWSER", "chrome").strip() or "chrome"

    use_cookies_txt_fallback = parse_bool(
        args.use_cookies_txt_fallback
        if args.use_cookies_txt_fallback is not None
        else os.getenv("USE_COOKIES_TXT_FALLBACK"),
        default=False,
    )

    cookies_txt_path = Path(args.cookies_txt_path or os.getenv("COOKIES_TXT_PATH", "cookies.txt"))

    enrich_metadata = parse_bool(
        args.enrich_metadata
        if args.enrich_metadata is not None
        else os.getenv("DISCOVERY_ENRICH_METADATA"),
        default=True,
    )

    return Config(
        channel_url=channel_url,
        title_filters=split_filters(raw_filters),
        db_path=db_path,
        max_discovery_videos=max_raw,
        use_browser_cookies=use_browser_cookies,
        browser=browser,
        use_cookies_txt_fallback=use_cookies_txt_fallback,
        cookies_txt_path=cookies_txt_path,
        enrich_metadata=enrich_metadata,
        dry_run=args.dry_run,
    )


def build_ydl_opts(config: Config, cookie_mode: str, *, extract_flat: bool) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "extract_flat": extract_flat,
        "ignoreerrors": True,
        "skip_download": True,
    }

    if extract_flat and config.max_discovery_videos:
        opts["playlistend"] = config.max_discovery_videos

    if cookie_mode == "browser":
        opts["cookiesfrombrowser"] = (config.browser,)
    elif cookie_mode == "cookies_txt":
        opts["cookiefile"] = str(config.cookies_txt_path)

    return opts


def cookie_attempts(config: Config, *, extract_flat: bool) -> list[tuple[str, dict[str, Any]]]:
    attempts: list[tuple[str, dict[str, Any]]] = []

    if config.use_browser_cookies:
        attempts.append(
            (
                "browser",
                build_ydl_opts(config, "browser", extract_flat=extract_flat),
            )
        )
    else:
        attempts.append(
            (
                "none",
                build_ydl_opts(config, "none", extract_flat=extract_flat),
            )
        )

    if config.use_cookies_txt_fallback and config.cookies_txt_path.exists():
        attempts.append(
            (
                "cookies_txt",
                build_ydl_opts(config, "cookies_txt", extract_flat=extract_flat),
            )
        )

    return attempts


def discover_raw(config: Config) -> dict[str, Any]:
    last_error: Exception | None = None

    for mode, opts in cookie_attempts(config, extract_flat=True):
        try:
            console.print(
                f"[bold]Running yt-dlp discovery[/bold] using cookie mode: [cyan]{mode}[/cyan]"
            )
            with yt_dlp.YoutubeDL(opts) as ydl:
                data = ydl.extract_info(config.channel_url, download=False)

            if not isinstance(data, dict):
                raise RuntimeError("yt-dlp returned unexpected non-dict response.")

            return data

        except Exception as exc:
            last_error = exc
            console.print(f"[yellow]Discovery attempt failed with mode {mode}: {exc}[/yellow]")

    raise RuntimeError(f"All discovery attempts failed. Last error: {last_error}")


def video_url_from_entry(entry: dict[str, Any]) -> str | None:
    if entry.get("webpage_url"):
        return str(entry["webpage_url"])

    video_id = entry.get("id")
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"

    url = entry.get("url")
    if isinstance(url, str) and url.startswith("http"):
        return url

    return None


def normalize_upload_date_from_info(info: dict[str, Any]) -> str | None:
    """
    yt-dlp usually returns upload_date as YYYYMMDD.

    For livestreams or unusual YouTube entries, upload_date may be absent.
    In that case, try timestamp/release_timestamp and convert it to YYYYMMDD.
    """
    upload_date = info.get("upload_date")
    if upload_date:
        return str(upload_date)

    timestamp = info.get("timestamp") or info.get("release_timestamp")
    if timestamp:
        try:
            return datetime.fromtimestamp(int(timestamp), timezone.utc).strftime("%Y%m%d")
        except Exception:
            return None

    return None


def extract_matches(data: dict[str, Any], title_filters: list[str]) -> list[dict[str, Any]]:
    entries = data.get("entries") or []
    upper_filters = [f.upper() for f in title_filters]

    matches: list[dict[str, Any]] = []

    for entry in entries:
        if not entry:
            continue

        title = str(entry.get("title") or "").strip()
        if not title:
            continue

        title_upper = title.upper()
        if not any(phrase in title_upper for phrase in upper_filters):
            continue

        video_id = entry.get("id")
        url = video_url_from_entry(entry)

        if not video_id or not url:
            continue

        matches.append(
            {
                "video_id": str(video_id),
                "title": title,
                "url": url,
                "duration": entry.get("duration"),
                "upload_date": entry.get("upload_date"),
            }
        )

    return matches


def enrich_matches_with_full_metadata(
    config: Config,
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Flat channel extraction is fast but often misses metadata like upload_date.

    This enriches only the already-matched videos with full per-video metadata,
    avoiding a full metadata pull for every channel item.
    """
    if not matches:
        return matches

    if not config.enrich_metadata:
        console.print("[yellow]Metadata enrichment disabled.[/yellow]")
        return matches

    console.print("[bold]Enriching matched videos with full metadata...[/bold]")

    total = len(matches)
    enriched = 0
    failed = 0

    # Try the same cookie modes as discovery.
    # Usually only the first working mode is needed.
    attempts = cookie_attempts(config, extract_flat=False)

    for idx, video in enumerate(matches, start=1):
        video_id = video.get("video_id")
        video_url = video.get("url")

        if not video_url:
            continue

        # Skip if we already got everything useful from flat mode.
        if video.get("upload_date") and video.get("duration"):
            continue

        console.print(f"Enriching {idx}/{total}: {video_id}")

        last_error: Exception | None = None
        info: dict[str, Any] | None = None

        for mode, opts in attempts:
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    result = ydl.extract_info(str(video_url), download=False)

                if isinstance(result, dict):
                    info = result
                    break

            except Exception as exc:
                last_error = exc
                console.print(
                    f"[yellow]Metadata enrich attempt failed for {video_id} "
                    f"with mode {mode}: {exc}[/yellow]"
                )

        if not info:
            failed += 1
            console.print(
                f"[yellow]Could not enrich metadata for {video_id}. "
                f"Last error: {last_error}[/yellow]"
            )
            continue

        upload_date = normalize_upload_date_from_info(info)
        duration = info.get("duration")

        if upload_date:
            video["upload_date"] = upload_date

        if duration and not video.get("duration"):
            video["duration"] = duration

        # Keep title/url fresh if full metadata has better values.
        if info.get("title"):
            video["title"] = str(info["title"])

        if info.get("webpage_url"):
            video["url"] = str(info["webpage_url"])

        enriched += 1

    console.print(
        f"[green]Metadata enrichment complete:[/green] "
        f"{enriched} enriched, {failed} failed"
    )

    return matches


def ensure_db_exists(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite DB not found at {db_path}. Run: python scripts/init_db.py"
        )


def start_run(conn: sqlite3.Connection, run_type: str, details: dict[str, Any]) -> int:
    cur = conn.execute(
        """
        INSERT INTO runs (started_at, run_type, status, details)
        VALUES (?, ?, ?, ?)
        """,
        (utc_now(), run_type, "running", json.dumps(details)),
    )
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, status: str, details: dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE runs
        SET completed_at = ?, status = ?, details = ?
        WHERE run_id = ?
        """,
        (utc_now(), status, json.dumps(details), run_id),
    )


def upsert_matches(config: Config, matches: list[dict[str, Any]]) -> dict[str, int]:
    ensure_db_exists(config.db_path)

    now = utc_now()
    inserted = 0
    updated = 0
    already_complete = 0

    with sqlite3.connect(config.db_path) as conn:
        conn.row_factory = sqlite3.Row

        run_id = start_run(
            conn,
            "discovery",
            {
                "channel_url": config.channel_url,
                "title_filters": config.title_filters,
                "max_discovery_videos": config.max_discovery_videos,
                "enrich_metadata": config.enrich_metadata,
            },
        )

        try:
            for video in matches:
                existing = conn.execute(
                    "SELECT video_id, ingest_status FROM videos WHERE video_id = ?",
                    (video["video_id"],),
                ).fetchone()

                if existing is None:
                    conn.execute(
                        """
                        INSERT INTO videos (
                            video_id,
                            url,
                            channel_url,
                            title,
                            upload_date,
                            duration,
                            title_filters,
                            discovered_at,
                            last_checked_at,
                            ingest_status,
                            audio_status,
                            whisper_status,
                            embedding_status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            video["video_id"],
                            video["url"],
                            config.channel_url,
                            video["title"],
                            video.get("upload_date"),
                            video.get("duration"),
                            ",".join(config.title_filters),
                            now,
                            now,
                            "queued",
                            "not_started",
                            "not_started",
                            "not_started",
                        ),
                    )
                    inserted += 1
                else:
                    conn.execute(
                        """
                        UPDATE videos
                        SET
                            url = ?,
                            channel_url = ?,
                            title = ?,
                            upload_date = COALESCE(?, upload_date),
                            duration = COALESCE(?, duration),
                            title_filters = ?,
                            last_checked_at = ?
                        WHERE video_id = ?
                        """,
                        (
                            video["url"],
                            config.channel_url,
                            video["title"],
                            video.get("upload_date"),
                            video.get("duration"),
                            ",".join(config.title_filters),
                            now,
                            video["video_id"],
                        ),
                    )

                    if existing["ingest_status"] == "complete":
                        already_complete += 1
                    else:
                        updated += 1

            summary = {
                "matched": len(matches),
                "inserted": inserted,
                "updated": updated,
                "already_complete": already_complete,
            }

            finish_run(conn, run_id, "complete", summary)
            conn.commit()
            return summary

        except Exception as exc:
            finish_run(conn, run_id, "failed", {"error": str(exc)})
            conn.commit()
            raise


def print_matches(matches: list[dict[str, Any]], limit: int = 20) -> None:
    table = Table(title=f"Discovered matching videos; showing first {min(limit, len(matches))}")
    table.add_column("Video ID", no_wrap=True)
    table.add_column("Title")
    table.add_column("Upload Date")
    table.add_column("Duration")

    for video in matches[:limit]:
        table.add_row(
            video["video_id"],
            video["title"][:100],
            str(video.get("upload_date") or ""),
            str(video.get("duration") or ""),
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--channel-url")
    parser.add_argument("--title-filters", help='Comma-separated filters, e.g. "ONE LIFE,1 LIFE"')
    parser.add_argument("--db-path")
    parser.add_argument("--max-discovery-videos", type=int)

    parser.add_argument("--browser", help="Browser for yt-dlp cookies-from-browser. Default: chrome")
    parser.add_argument("--cookies-txt-path")
    parser.add_argument("--use-browser-cookies", choices=["true", "false"])
    parser.add_argument("--use-cookies-txt-fallback", choices=["true", "false"])

    parser.add_argument(
        "--enrich-metadata",
        choices=["true", "false"],
        help="Fetch full metadata for matched videos so upload_date is populated. Default: true",
    )

    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> int:
    try:
        config = load_config(parse_args())
    except Exception as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        return 2

    console.print("[bold]Discovery config[/bold]")
    console.print(f"Channel URL: {config.channel_url}")
    console.print(f"Title filters: {config.title_filters}")
    console.print(f"DB path: {config.db_path}")
    console.print(f"Max discovery videos: {config.max_discovery_videos or 'unlimited'}")
    console.print(f"Use browser cookies: {config.use_browser_cookies}")
    console.print(f"Browser: {config.browser}")
    console.print(f"Cookies.txt fallback: {config.use_cookies_txt_fallback}")
    console.print(f"Enrich metadata: {config.enrich_metadata}")
    console.print(f"Dry run: {config.dry_run}")
    console.print()

    try:
        data = discover_raw(config)
        matches = extract_matches(data, config.title_filters)
        console.print(f"[green]Matched videos before enrichment:[/green] {len(matches)}")
        matches = enrich_matches_with_full_metadata(config, matches)
    except Exception as exc:
        console.print(f"[red]Discovery failed:[/red] {exc}")
        return 1

    console.print(f"[green]Matched videos:[/green] {len(matches)}")
    print_matches(matches)

    if config.dry_run:
        console.print("[yellow]Dry run enabled. No database writes performed.[/yellow]")
        return 0

    try:
        summary = upsert_matches(config, matches)
    except Exception as exc:
        console.print(f"[red]Database update failed:[/red] {exc}")
        return 1

    console.print("[bold green]Discovery complete.[/bold green]")
    console.print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())