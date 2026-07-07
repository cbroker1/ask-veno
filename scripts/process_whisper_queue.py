#!/usr/bin/env python3
"""
process_whisper_queue.py -- Transcribe downloaded audio files using faster-whisper.

This script is based on the original notebook workflow. It loads the Whisper
model once, then processes queued audio files in order.

- Load Whisper once (supports batched inference for throughput).
- Use word_timestamps=True for per-word timing.
- Write transcript JSON into the same folder as the MP3.
- Name the transcript after the folder: <folder_name> transcript.json.

SQLite-state driven:

- Select videos with ingest_status='audio_ready' and audio_status='downloaded'.
- Skip videos already transcribed (whisper_status='transcribed').
- Update whisper_status and transcript_path on success.

USAGE
-----
    python scripts/process_whisper_queue.py

OPTIONS
    --db-path PATH                    SQLite database path
                                      (default: .state/youtube_ingest.sqlite)
    --max-transcribe-videos N         Number of videos to transcribe per run
                                      (default: 1)
    --whisper-model NAME              Whisper model name (default: large-v3)
    --whisper-device DEVICE           Device: cuda:0, cuda, or cpu
                                      (default: cuda:0)
    --whisper-compute-type TYPE       float16, float32, int8, int8_float16, etc.
                                      (default: float16)
    --whisper-batch-size N            Batch size for batched inference (default: 8)
    --whisper-beam-size N             Beam search width (default: 5)
    --whisper-vad-filter true|false   Enable voice activity detection filter
                                      (default: true)
    --word-timestamps true|false      Enable per-word timestamps (default: true)
    --dry-run                         Print queue and exit without transcribing

EXAMPLES
    # Transcribe 1 video with defaults
    python scripts/process_whisper_queue.py

    # Transcribe 5 videos using a smaller model on CPU
    python scripts/process_whisper_queue.py \
        --max-transcribe-videos 5 \
        --whisper-model base.en \
        --whisper-device cpu

    # Dry run to see what would be transcribed
    python scripts/process_whisper_queue.py --dry-run

WHAT IT DOES
------------
1. Queries the database for audio_ready videos with audio_status='downloaded'
   and whisper_status != 'transcribed'.
2. Loads the faster-whisper model (single instance, reused for all videos).
3. For each video:
   - Checks if the audio file exists.
   - Skips if a transcript JSON already exists (idempotent).
   - Runs transcribe() with beam search and optional VAD filter.
   - Writes transcript JSON (segments, text, language info, config).
   - Marks the video as transcribed in the database.
4. Handles failures by recording error type and message.

ENVIRONMENT VARIABLES
    SQLITE_DB_PATH                Database path (default: .state/youtube_ingest.sqlite)
    MAX_TRANSCRIBE_VIDEOS         Max videos per run (default: 1)
    WHISPER_MODEL                 Whisper model (default: large-v3)
    WHISPER_DEVICE                Device string (default: cuda:0)
    WHISPER_COMPUTE_TYPE          Compute type (default: float16)
    WHISPER_BATCH_SIZE            Batch size (default: 8)
    WHISPER_BEAM_SIZE             Beam size (default: 5)
    WHISPER_VAD_FILTER            VAD filter (default: true)
    WHISPER_WORD_TIMESTAMPS       Word timestamps (default: true)

EXIT CODES
    0  Transcription completed (all or no videos).
    1  One or more videos failed transcription.
    2  Startup error (config, DB, or model loading failure).
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
from faster_whisper import BatchedInferencePipeline, WhisperModel
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
    whisper_compute_type: str
    whisper_batch_size: int
    whisper_beam_size: int
    whisper_vad_filter: bool
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

    whisper_batch_size = args.whisper_batch_size
    if whisper_batch_size is None:
        whisper_batch_size = int(os.getenv("WHISPER_BATCH_SIZE", "8"))

    whisper_beam_size = args.whisper_beam_size
    if whisper_beam_size is None:
        whisper_beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "5"))

    return Config(
        db_path=Path(args.db_path or os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)),
        max_transcribe_videos=max_transcribe,
        whisper_model=args.whisper_model or os.getenv("WHISPER_MODEL", "large-v3"),
        whisper_device=args.whisper_device or os.getenv("WHISPER_DEVICE", "cuda:0"),
        whisper_compute_type=args.whisper_compute_type
        or os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
        whisper_batch_size=whisper_batch_size,
        whisper_beam_size=whisper_beam_size,
        whisper_vad_filter=parse_bool(
            args.whisper_vad_filter
            if args.whisper_vad_filter is not None
            else os.getenv("WHISPER_VAD_FILTER"),
            default=True,
        ),
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
          AND (whisper_status IS NULL OR whisper_status != 'transcribed')
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
            row["whisper_status"] or "",
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


def parse_faster_whisper_device(device: str) -> tuple[str, int]:
    """
    Convert OpenAI-style device strings into faster-whisper args.

    Examples:
        cuda:0 -> ("cuda", 0)
        cuda   -> ("cuda", 0)
        cpu    -> ("cpu", 0)
    """
    normalized = device.strip().lower()

    if normalized.startswith("cuda:"):
        _, index = normalized.split(":", 1)
        return "cuda", int(index)

    if normalized == "cuda":
        return "cuda", 0

    if normalized == "cpu":
        return "cpu", 0

    raise ValueError(
        f"Unsupported WHISPER_DEVICE value: {device}. Use cuda:0, cuda, or cpu."
    )


def load_transcription_model(config: Config) -> Any:
    device, device_index = parse_faster_whisper_device(config.whisper_device)

    console.print("[bold]Loading faster-whisper model...[/bold]")
    console.print(f"Model: {config.whisper_model}")
    console.print(f"Device: {device}")
    console.print(f"Device index: {device_index}")
    console.print(f"Compute type: {config.whisper_compute_type}")
    console.print(f"Batch size: {config.whisper_batch_size}")

    base_model = WhisperModel(
        config.whisper_model,
        device=device,
        device_index=device_index,
        compute_type=config.whisper_compute_type,
    )

    if config.whisper_batch_size > 1:
        console.print("[bold]Using faster-whisper batched inference.[/bold]")
        return BatchedInferencePipeline(model=base_model)

    return base_model


def get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def json_safe(value: Any) -> Any:
    """
    Make faster-whisper metadata safe for json.dumps.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, list):
        return [json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [json_safe(item) for item in value]

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    # Handle namedtuples / small objects from faster-whisper metadata.
    if hasattr(value, "_asdict"):
        return json_safe(value._asdict())

    return str(value)


def word_to_dict(word: Any) -> dict[str, Any]:
    return {
        "word": get_attr(word, "word"),
        "start": get_attr(word, "start"),
        "end": get_attr(word, "end"),
        "probability": get_attr(word, "probability"),
    }


def segment_to_dict(segment: Any) -> dict[str, Any]:
    words = get_attr(segment, "words", None)

    return {
        "id": get_attr(segment, "id"),
        "seek": get_attr(segment, "seek"),
        "start": get_attr(segment, "start"),
        "end": get_attr(segment, "end"),
        "text": get_attr(segment, "text", ""),
        "tokens": json_safe(get_attr(segment, "tokens", [])),
        "temperature": get_attr(segment, "temperature"),
        "avg_logprob": get_attr(segment, "avg_logprob"),
        "compression_ratio": get_attr(segment, "compression_ratio"),
        "no_speech_prob": get_attr(segment, "no_speech_prob"),
        "words": [word_to_dict(word) for word in words] if words else [],
    }


def info_to_dict(info: Any) -> dict[str, Any]:
    return {
        "language": get_attr(info, "language"),
        "language_probability": get_attr(info, "language_probability"),
        "duration": get_attr(info, "duration"),
        "duration_after_vad": get_attr(info, "duration_after_vad"),
        "all_language_probs": json_safe(get_attr(info, "all_language_probs")),
    }


def transcribe_audio(audio_path: Path, model: Any, config: Config) -> dict[str, Any]:
    """
    faster-whisper returns:

        segments, info = model.transcribe(...)

    segments is lazy. Actual transcription happens while iterating over segments.
    This function consumes the iterator and normalizes the result into JSON.
    """
    kwargs: dict[str, Any] = {
        "beam_size": config.whisper_beam_size,
        "word_timestamps": config.word_timestamps,
        "vad_filter": config.whisper_vad_filter,
    }

    if config.whisper_batch_size > 1:
        kwargs["batch_size"] = config.whisper_batch_size

    segments_iter, info = model.transcribe(str(audio_path), **kwargs)

    segment_list: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    for idx, segment in enumerate(segments_iter, start=1):
        segment_dict = segment_to_dict(segment)
        segment_list.append(segment_dict)
        full_text_parts.append(segment_dict.get("text") or "")

        if idx % 25 == 0:
            start = segment_dict.get("start")
            end = segment_dict.get("end")

            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                console.print(
                    f"Processed {idx} segments; latest segment "
                    f"{start:.2f}s -> {end:.2f}s"
                )
            else:
                console.print(f"Processed {idx} segments.")

    info_dict = info_to_dict(info)

    return {
        "text": "".join(full_text_parts).strip(),
        "segments": segment_list,
        "language": info_dict.get("language"),
        "language_probability": info_dict.get("language_probability"),
        "duration": info_dict.get("duration"),
        "duration_after_vad": info_dict.get("duration_after_vad"),
        "all_language_probs": info_dict.get("all_language_probs"),
        "faster_whisper": {
            "model": config.whisper_model,
            "device": config.whisper_device,
            "compute_type": config.whisper_compute_type,
            "batch_size": config.whisper_batch_size,
            "beam_size": config.whisper_beam_size,
            "vad_filter": config.whisper_vad_filter,
            "word_timestamps": config.word_timestamps,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--db-path")
    parser.add_argument("--max-transcribe-videos", type=int)

    parser.add_argument("--whisper-model")
    parser.add_argument("--whisper-device")
    parser.add_argument("--whisper-compute-type")
    parser.add_argument("--whisper-batch-size", type=int)
    parser.add_argument("--whisper-beam-size", type=int)
    parser.add_argument("--whisper-vad-filter", choices=["true", "false"])
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

    console.print("[bold]faster-whisper queue processor config[/bold]")
    console.print(f"DB path: {config.db_path}")
    console.print(f"Max transcribe videos: {config.max_transcribe_videos}")
    console.print(f"Whisper model: {config.whisper_model}")
    console.print(f"Whisper device: {config.whisper_device}")
    console.print(f"Whisper compute type: {config.whisper_compute_type}")
    console.print(f"Whisper batch size: {config.whisper_batch_size}")
    console.print(f"Whisper beam size: {config.whisper_beam_size}")
    console.print(f"Whisper VAD filter: {config.whisper_vad_filter}")
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

    try:
        model = load_transcription_model(config)
    except Exception as exc:
        console.print(f"[red]Failed to load faster-whisper model:[/red] {exc}")
        return 2

    console.print("[green]faster-whisper model loaded.[/green]")

    for row in rows:
        video_id = row["video_id"]
        audio_path = Path(row["audio_path"])
        transcript_path = transcript_output_path(audio_path)

        console.rule(f"Transcribing {video_id}")

        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

            if transcript_path.exists():
                console.print(
                    "[yellow]Transcript already exists, marking transcribed:[/yellow] "
                    f"{transcript_path}"
                )
                mark_transcribed(conn, video_id, transcript_path)
                continue

            console.print(f"Audio: {audio_path}")
            console.print(f"Transcript output: {transcript_path}")

            result = transcribe_audio(audio_path, model, config)

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