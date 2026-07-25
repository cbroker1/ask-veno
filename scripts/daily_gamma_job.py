#!/usr/bin/env python3
"""
daily_gamma_job.py -- Daily GAMMA 0.9.5 corpus maintenance.

Designed for a systemd user timer (or scripts/daily_all_job.py):

1. Acquire .state/daily_gamma_job.lock (non-blocking; exit 0 if held).
2. Synchronize the upstream sparse checkout (fast-forward only; offline
   keeps the current corpus and exits 0).
3. Compare the upstream manifest SHA-256 with the active gamma_meta.
4. Unchanged -> print "unchanged", exit 0. No embeddings are rebuilt.
5. Changed -> staged rebuild (SQLite + versioned Chroma collection),
   validation, atomic activation. Failure keeps the previous corpus and
   exits non-zero.

USAGE
-----
    python scripts/daily_gamma_job.py

EXIT CODES
    0 unchanged / rebuilt successfully / another instance running / offline
    1 sync failed with no usable local dataset, or rebuild failed
"""

from __future__ import annotations

import fcntl
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import get_chroma_path, get_gamma_db_path, get_gamma_repo_path  # noqa: E402

import build_gamma_095  # noqa: E402
import sync_gamma_095  # noqa: E402


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def main() -> int:
    lock_path = ROOT / ".state" / "daily_gamma_job.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("Another daily GAMMA job is already running. Exiting.")
            return 0

        repo_path = get_gamma_repo_path()
        db_path = get_gamma_db_path()
        chroma_path = get_chroma_path()

        log(f"Sync upstream ({repo_path}) …")
        try:
            sync_result = sync_gamma_095.sync(repo_path)
        except (ValueError, RuntimeError) as exc:
            log(f"Sync failed: {exc}")
            return 1
        if sync_result["status"] == "error":
            log(f"Sync failed: {sync_result.get('error')}")
            return 1
        log(f"Sync {sync_result['status']}: commit "
            f"{sync_result.get('upstream_commit', '?')[:12]} "
            f"manifest {sync_result.get('manifest_sha256', '?')[:12]}")

        manifest_sha = sync_result.get("manifest_sha256")
        active_sha = build_gamma_095.active_manifest_sha(db_path)
        active_col = build_gamma_095.active_collection_name(db_path)
        if (manifest_sha and manifest_sha == active_sha
                and build_gamma_095.collection_intact(chroma_path, active_col)):
            log("unchanged — active corpus already matches upstream manifest")
            return 0

        log("Manifest changed (or corpus incomplete) — staged rebuild …")
        try:
            result = build_gamma_095.run_build(repo_path, db_path, chroma_path)
        except Exception as exc:
            log(f"Rebuild failed, previous corpus retained: {exc}")
            return 1
        if result["status"] not in ("built", "unchanged"):
            log(f"Rebuild failed, previous corpus retained: "
                f"{result.get('problems')}")
            return 1
        log(f"Rebuild {result['status']}: items={result.get('item_count')} "
            f"facets={result.get('facet_count')} "
            f"collection={result.get('active_collection')}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
