#!/usr/bin/env python3
"""
build_lexical_index.py -- Build/refresh the transcript BM25 index.

Reads every document + metadata from the EXISTING youtube_chunks Chroma
collection (read-only — nothing is re-transcribed or re-embedded) and
rebuilds .state/lexical_search.sqlite atomically. Idempotent: when the
collection chunk count matches the index and --force is absent, exits 0
without rebuilding. The full rebuild of ~29k chunks takes only a few
seconds, so daily synchronization simply re-runs this script.

USAGE
-----
    python scripts/build_lexical_index.py [--force] [--db-path P] [--json]

EXIT CODES
    0 built or already current;  1 failure (existing index left untouched).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import CHROMA_COLLECTION, get_chroma_path, get_lexical_db_path  # noqa: E402
from storage.lexical import LexicalDB, LexicalUnavailable, build_from_rows  # noqa: E402

BATCH = 2000


def fetch_all_chunks(chroma_path: Path, collection_name: str) -> list[dict]:
    import chromadb
    client = chromadb.PersistentClient(path=str(chroma_path))
    col = client.get_collection(collection_name)
    total = col.count()
    rows: list[dict] = []
    offset = 0
    while offset < total:
        got = col.get(include=["documents", "metadatas"], limit=BATCH, offset=offset)
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        for i, chunk_id in enumerate(ids):
            md = metas[i] or {}
            rows.append({
                "chunk_id": chunk_id,
                "video_id": md.get("video_id"),
                "video_title": md.get("video_title"),
                "start_hms": md.get("start_hms"),
                "end_hms": md.get("end_hms"),
                "url": md.get("youtube_time_url"),
                "text": docs[i] or "",
            })
        offset += len(ids)
        if not ids:
            break
    rows.sort(key=lambda r: r["chunk_id"])   # deterministic
    return rows


def current_index_count(db_path: Path) -> int | None:
    try:
        db = LexicalDB(db_path)
    except LexicalUnavailable:
        return None
    try:
        return db.count()
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the transcript BM25 index")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    db_path = args.db_path or get_lexical_db_path()
    chroma_path = get_chroma_path()
    result: dict

    try:
        if not chroma_path.exists():
            raise RuntimeError(f"Chroma path missing: {chroma_path}")
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.get_collection(CHROMA_COLLECTION)
        collection_count = col.count()

        if not args.force and current_index_count(db_path) == collection_count:
            result = {"status": "unchanged", "chunks": collection_count,
                      "db_path": str(db_path)}
        else:
            t0 = time.perf_counter()
            rows = fetch_all_chunks(chroma_path, CHROMA_COLLECTION)
            build_from_rows(db_path, rows,
                            source=f"chroma:{CHROMA_COLLECTION}")
            result = {"status": "built", "chunks": len(rows),
                      "db_path": str(db_path),
                      "build_seconds": round(time.perf_counter() - t0, 2)}
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"[build_lexical_index] {result}")
    return 0 if result["status"] in ("built", "unchanged") else 1


if __name__ == "__main__":
    raise SystemExit(main())
