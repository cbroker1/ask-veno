#!/usr/bin/env python3
"""
sync_gamma_095.py -- Synchronize the pinned GAMMA 0.9.5 dataset from upstream.

Maintains a sparse sibling checkout of simonwdev/stalker-anomaly-gamma-db
(AGPL-3.0) OUTSIDE the ask-veno repository, containing only the generated
gamma-0.9.5 data tree:

    <repo>/site/public/data/gamma-0.9.5

The pack id is hard-pinned in app/config.py (GAMMA_PACK_ID). No other GAMMA
version is ever fetched or accepted by this script.

Behavior
--------
- Repo missing        -> blob-filtered sparse clone of `main`, dataset tree only.
- Repo present        -> verify remote URL, refuse to touch a dirty worktree,
                         fetch origin, fast-forward only. Never resets.
- Network unreachable -> if a valid local dataset exists, exit 0 with
                         status "offline_stale"; the active corpus is never
                         damaged by a failed fetch.
- Already up to date  -> exit 0, status "ok", synced_new_commit false.

USAGE
-----
    python scripts/sync_gamma_095.py [--check-only] [--json] [--repo-path PATH]

OPTIONS
    --check-only    Validate the existing local checkout only. No clone, no
                    fetch, no network.
    --json          Print a machine-readable JSON result on stdout.
    --repo-path P   Override the sibling checkout location
                    (default: GAMMA_REPO_PATH env or ../stalker-anomaly-gamma-db-upstream).

EXIT CODES
    0  Local dataset valid (fresh, already current, or offline-stale).
    1  Sync/validation failed and no valid local dataset is available,
       or the existing checkout is dirty / points at the wrong remote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    GAMMA_DATA_SUBDIR,
    GAMMA_PACK_ID,
    GAMMA_UPSTREAM_URL,
    get_gamma_repo_path,
)

UPSTREAM_BRANCH = "main"


def _git(repo: Path, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=timeout, check=False,
    )


def manifest_sha256(manifest_path: Path) -> str:
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """manifest.json is a flat {"<file>.json": "<8-hex content hash>"} map
    (verified against upstream gamma-0.9.5). Reject anything else so a future
    upstream format change fails loudly instead of being half-parsed.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (isinstance(data, dict) and data
            and all(isinstance(k, str) and isinstance(v, str)
                    for k, v in data.items())):
        return data
    raise ValueError(f"Unrecognized manifest.json structure in {manifest_path}")


def manifest_file_count(manifest_path: Path) -> int:
    return len(load_manifest(manifest_path))


def validate_dataset(repo_path: Path) -> dict:
    """Validate the local gamma-0.9.5 tree; raise ValueError when invalid."""
    data_dir = repo_path / GAMMA_DATA_SUBDIR
    if not data_dir.is_dir():
        raise ValueError(f"Dataset directory missing: {data_dir}")
    manifest = data_dir / "manifest.json"
    index = data_dir / "index.json"
    for required in (manifest, index):
        if not required.is_file():
            raise ValueError(f"Required dataset file missing: {required}")

    head = _git(repo_path, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise ValueError(f"Cannot resolve HEAD in {repo_path}: {head.stderr.strip()}")

    return {
        "pack_id": GAMMA_PACK_ID,
        "repo_path": str(repo_path),
        "data_dir": str(data_dir),
        "upstream_url": GAMMA_UPSTREAM_URL,
        "upstream_commit": head.stdout.strip(),
        "manifest_sha256": manifest_sha256(manifest),
        "manifest_file_count": manifest_file_count(manifest),
    }


def _remote_url(repo_path: Path) -> str:
    res = _git(repo_path, "remote", "get-url", "origin")
    return res.stdout.strip() if res.returncode == 0 else ""


def _worktree_dirty(repo_path: Path) -> bool:
    res = _git(repo_path, "status", "--porcelain=v1")
    return bool(res.stdout.strip())


def clone(repo_path: Path) -> None:
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout",
         "--branch", UPSTREAM_BRANCH, GAMMA_UPSTREAM_URL, str(repo_path)],
        check=True, timeout=600,
    )
    res = _git(repo_path, "sparse-checkout", "set", GAMMA_DATA_SUBDIR)
    if res.returncode != 0:
        raise RuntimeError(f"sparse-checkout set failed: {res.stderr.strip()}")
    res = _git(repo_path, "checkout", UPSTREAM_BRANCH, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"checkout failed: {res.stderr.strip()}")


def update(repo_path: Path) -> dict:
    """Fetch + fast-forward the existing checkout. Never destructive."""
    url = _remote_url(repo_path)
    expected = GAMMA_UPSTREAM_URL.removesuffix(".git")
    if url.removesuffix(".git") != expected:
        raise ValueError(
            f"{repo_path} tracks unexpected remote {url!r} "
            f"(expected {GAMMA_UPSTREAM_URL}); refusing to sync."
        )
    if _worktree_dirty(repo_path):
        raise ValueError(
            f"{repo_path} has local modifications; refusing destructive sync. "
            "Inspect and clean it manually."
        )

    before = _git(repo_path, "rev-parse", "HEAD").stdout.strip()
    fetch = _git(repo_path, "fetch", "origin", UPSTREAM_BRANCH, timeout=600)
    if fetch.returncode != 0:
        return {"fetched": False, "error": fetch.stderr.strip(), "before": before}

    ff = _git(repo_path, "merge", "--ff-only", f"origin/{UPSTREAM_BRANCH}")
    if ff.returncode != 0:
        raise ValueError(f"Fast-forward failed (diverged history?): {ff.stderr.strip()}")
    after = _git(repo_path, "rev-parse", "HEAD").stdout.strip()
    return {"fetched": True, "before": before, "after": after,
            "synced_new_commit": before != after}


def sync(repo_path: Path, check_only: bool = False) -> dict:
    result: dict = {"pack_id": GAMMA_PACK_ID, "status": "error"}

    if check_only:
        info = validate_dataset(repo_path)
        return {**info, "status": "ok", "checked_only": True,
                "synced_new_commit": False}

    if not (repo_path / ".git").exists():
        try:
            clone(repo_path)
        except Exception as exc:
            result["error"] = f"clone failed: {exc}"
            return result
        info = validate_dataset(repo_path)
        return {**info, "status": "ok", "cloned": True, "synced_new_commit": True}

    upd = update(repo_path)
    info = validate_dataset(repo_path)
    if not upd.get("fetched"):
        # Offline: keep serving the existing valid checkout.
        return {**info, "status": "offline_stale",
                "fetch_error": upd.get("error", ""), "synced_new_commit": False}
    return {**info, "status": "ok",
            "synced_new_commit": bool(upd.get("synced_new_commit"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--repo-path", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_path = args.repo_path or get_gamma_repo_path()

    try:
        result = sync(repo_path, check_only=args.check_only)
    except (ValueError, RuntimeError) as exc:
        result = {"pack_id": GAMMA_PACK_ID, "status": "error", "error": str(exc)}

    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if result["status"] in ("ok", "offline_stale"):
            print(f"[sync_gamma_095] {result['status']}: {GAMMA_PACK_ID} "
                  f"@ {result.get('upstream_commit', '?')[:12]} "
                  f"manifest={result.get('manifest_sha256', '?')[:12]} "
                  f"files={result.get('manifest_file_count', '?')} "
                  f"new_commit={result.get('synced_new_commit')}")
        else:
            print(f"[sync_gamma_095] ERROR: {result.get('error')}", file=sys.stderr)

    return 0 if result["status"] in ("ok", "offline_stale") else 1


if __name__ == "__main__":
    raise SystemExit(main())
