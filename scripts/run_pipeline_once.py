#!/usr/bin/env python3
"""
Run one pass of the local YouTube RAG pipeline.

This advances each queue stage by a small number of items:

1. Discover audio candidates
2. Download audio
3. Transcribe audio with Whisper
4. Clean transcript
5. Embed into ChromaDB

This does not query Ollama. Use scripts/query_chromadb.py for inference testing.
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
