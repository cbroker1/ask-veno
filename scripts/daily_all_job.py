#!/usr/bin/env python3
"""
daily_all_job.py -- All-corpus daily wrapper.

Runs, in order, each as its own subprocess with its own locking:

1. daily_gamma_job.py     GAMMA 0.9.5 sync + conditional rebuild
2. daily_channel_job.py   YouTube discovery/ingestion, which always ends with
                          the transcript BM25 sync (no-op when the chunk
                          count is unchanged)

A failing GAMMA step does not block the YouTube job (the corpora are
independent); the exit code reflects the worst step so the timer unit shows
failures.

To point the existing systemd user timer here, change ExecStart of the
daily service to:

    ExecStart=/path/to/conda-env/bin/python scripts/daily_all_job.py

(See README — this script never modifies systemd itself.)

USAGE
-----
    python scripts/daily_all_job.py

EXIT CODES
    0 all steps succeeded;  1 at least one step failed.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("GAMMA 0.9.5 daily job", ["scripts/daily_gamma_job.py"]),
    ("YouTube channel daily job", ["scripts/daily_channel_job.py"]),
]


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def main() -> int:
    worst = 0
    for name, argv in STEPS:
        log("=" * 78)
        log(f"STEP: {name}")
        log("=" * 78)
        code = subprocess.run([sys.executable, *argv], cwd=ROOT, check=False).returncode
        if code == 0:
            log(f"OK: {name}")
        else:
            log(f"FAILED: {name} exit_code={code}")
            worst = 1
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
