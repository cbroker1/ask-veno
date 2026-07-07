#!/usr/bin/env python3
"""
run_pipeline_once.py -- Run one pass of the local YouTube RAG pipeline.

This advances each queue stage by a small number of items, in order:

1. Discover audio candidates (scan channel for new matching videos)
2. Download audio (yt-dlp audio extraction)
3. Transcribe audio with Whisper (faster-whisper)
4. Clean transcript (strip raw Whisper artifacts)
5. Embed into ChromaDB (vector store)

This does not query Ollama. Use scripts/query_chromadb.py for inference testing.

USAGE
-----
    python scripts/run_pipeline_once.py

OPTIONS
    --max-discovery-videos N    Max videos to discover per run (default: 25)
    --max-new-videos N          Max videos to download per run (default: 1)
    --max-transcribe-videos N   Max videos to transcribe per run (default: 1)
    --max-clean-videos N        Max transcripts to clean per run (default: 1)
    --max-embed-videos N        Max videos to embed per run (default: 1)
    --skip-discovery            Skip the discovery step
    --skip-download             Skip the audio download step
    --skip-whisper              Skip the Whisper transcription step
    --skip-clean                Skip the transcript cleanup step
    --skip-embed                Skip the ChromaDB embedding step
    --use-browser-cookies true|false  Use browser cookies for audio download
                                      (default: false)

EXAMPLES
    # Run all steps with defaults
    python scripts/run_pipeline_once.py

    # Run only discovery and download
    python scripts/run_pipeline_once.py --skip-whisper --skip-clean --skip-embed

    # Run all steps but process 5 videos per stage
    python scripts/run_pipeline_once.py \
        --max-discovery-videos 50 \
        --max-new-videos 5 \
        --max-transcribe-videos 5 \
        --max-clean-videos 5 \
        --max-embed-videos 5

    # Run only Whisper transcription
    python scripts/run_pipeline_once.py \
        --skip-discovery --skip-download --skip-clean --skip-embed

WHAT IT DOES
------------
This script is a pipeline orchestrator. It runs each step sequentially,
passing control to the corresponding queue processor script:

    discover_audio_candidates.py  -> download_audio_candidates.py
    process_audio_queue.py        -> process_audio_queue.py
    process_whisper_queue.py      -> process_whisper_queue.py
    process_transcript_queue.py   -> process_transcript_queue.py
    process_chromadb_queue.py     -> process_chromadb_queue.py

If any required step fails, the pipeline stops and returns exit code 1.
Optional steps (marked with --skip-*) are skipped entirely.

ENVIRONMENT VARIABLES
    All environment variables accepted by the individual queue processor
    scripts are also valid here (e.g., SQLITE_DB_PATH, WHISPER_MODEL, etc.).

EXIT CODES
    0  All selected steps completed successfully.
    1  A required step failed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Step:
    name: str
    command: list[str]
    optional: bool = False


def run_step(step: Step) -> bool:
    print("\n" + "=" * 100)
    print(f"STEP: {step.name}")
    print("=" * 100)
    print("$ " + " ".join(step.command))
    print()

    result = subprocess.run(step.command, check=False)

    if result.returncode == 0:
        print(f"\n✅ {step.name} complete")
        return True

    print(f"\n❌ {step.name} failed with exit code {result.returncode}")

    if step.optional:
        print("Continuing because this step is optional.")
        return True

    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-discovery-videos", type=int, default=25)
    parser.add_argument("--max-new-videos", type=int, default=1)
    parser.add_argument("--max-transcribe-videos", type=int, default=1)
    parser.add_argument("--max-clean-videos", type=int, default=1)
    parser.add_argument("--max-embed-videos", type=int, default=1)

    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-whisper", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")

    parser.add_argument(
        "--use-browser-cookies",
        choices=["true", "false"],
        default="false",
        help="Use browser cookies for audio download. Default false.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    py = sys.executable

    steps: list[Step] = []

    if not args.skip_discovery:
        steps.append(
            Step(
                name="Discover audio candidates",
                command=[
                    py,
                    "scripts/discover_audio_candidates.py",
                    "--max-discovery-videos",
                    str(args.max_discovery_videos),
                    "--use-browser-cookies",
                    "false",
                ],
            )
        )

    if not args.skip_download:
        steps.append(
            Step(
                name="Download queued audio",
                command=[
                    py,
                    "scripts/process_audio_queue.py",
                    "--max-new-videos",
                    str(args.max_new_videos),
                    "--use-browser-cookies",
                    args.use_browser_cookies,
                ],
            )
        )

    if not args.skip_whisper:
        steps.append(
            Step(
                name="Transcribe audio with Whisper",
                command=[
                    py,
                    "scripts/process_whisper_queue.py",
                    "--max-transcribe-videos",
                    str(args.max_transcribe_videos),
                ],
            )
        )

    if not args.skip_clean:
        steps.append(
            Step(
                name="Clean raw Whisper transcript",
                command=[
                    py,
                    "scripts/process_transcript_queue.py",
                    "--max-clean-videos",
                    str(args.max_clean_videos),
                ],
            )
        )

    if not args.skip_embed:
        steps.append(
            Step(
                name="Embed clean transcript into ChromaDB",
                command=[
                    py,
                    "scripts/process_chromadb_queue.py",
                    "--max-embed-videos",
                    str(args.max_embed_videos),
                ],
            )
        )

    if not steps:
        print("No steps selected.")
        return 0

    for step in steps:
        ok = run_step(step)
        if not ok:
            print("\nPipeline stopped because a required step failed.")
            return 1

    print("\n" + "=" * 100)
    print("Pipeline pass complete.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
