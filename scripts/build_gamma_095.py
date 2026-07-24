#!/usr/bin/env python3
"""
build_gamma_095.py -- Build the canonical GAMMA 0.9.5 corpus (SQLite + Chroma).

Staged, atomic, idempotent:

1. Load + validate the synchronized upstream dataset (version-pinned).
2. If the active database already matches the manifest SHA-256 (and its
   Chroma collection is intact), exit 0 without rebuilding.
3. Canonicalize into a STAGING SQLite file (never the active one).
4. Run the completeness audit; hard failures abort with the active corpus
   untouched.
5. Embed facets into a NEW versioned Chroma collection
   (gamma_items_095_<manifest8>_<utc-stamp>) — the active collection is never
   modified in place.
6. Validate counts/ids + a smoke dense query.
7. Record the new collection in the staging DB, then os.replace() it onto the
   active path (atomic on POSIX).
8. Clean up stale gamma collections, keeping the new active and the previous.

USAGE
-----
    python scripts/build_gamma_095.py [options]

OPTIONS
    --repo-path P    Upstream checkout (default: config/GAMMA_REPO_PATH).
    --db-path P      Active DB path (default: .state/gamma_095.sqlite).
    --chroma-path P  Chroma directory (default: data/chroma).
    --force          Rebuild even when the manifest is unchanged.
    --no-embed       Skip Chroma embedding (structured + BM25 corpus only).
    --audit-only     Run canonicalization + audit, write the audit report,
                     change nothing on disk otherwise.
    --json           Machine-readable result on stdout.

EXIT CODES
    0 built / already current;  1 failed (active corpus left untouched).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    GAMMA_CHROMA_COLLECTION_PREFIX,
    GAMMA_PACK_ID,
    get_chroma_path,
    get_gamma_db_path,
    get_gamma_repo_path,
)
from indexing.gamma import builder as builder_mod  # noqa: E402
from indexing.gamma.loader import GammaLoadError, load_dataset  # noqa: E402

AUDIT_PATH = ROOT / ".state" / "gamma_095_build_audit.json"


def _upstream_commit(repo_path: Path) -> str:
    import subprocess
    res = subprocess.run(["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    return res.stdout.strip() if res.returncode == 0 else "unknown"


def active_manifest_sha(db_path: Path) -> str | None:
    if not db_path.exists():
        return None
    import sqlite3
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM gamma_meta WHERE key='manifest_sha256'").fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def active_collection_name(db_path: Path) -> str | None:
    if not db_path.exists():
        return None
    import sqlite3
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT value FROM gamma_meta WHERE key='active_chroma_collection'").fetchone()
            return row[0] if row and row[0] else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def collection_intact(chroma_path: Path, name: str | None, expected: int | None = None) -> bool:
    if not name or not chroma_path.exists():
        return False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.get_collection(name)
        return expected is None or col.count() == expected
    except Exception:
        return False


def embed_facets(chroma_path: Path, collection_name: str, facets, manifest_sha: str,
                 items_by_id) -> dict:
    """Embed all facets into a brand-new collection; returns timing info."""
    import chromadb

    from rag.embeddings import get_embedding_model

    model = get_embedding_model()
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        client.delete_collection(collection_name)   # stale partial staging only
    except Exception:
        pass
    col = client.create_collection(
        name=collection_name,
        metadata={"pack_id": GAMMA_PACK_ID, "manifest_sha256": manifest_sha,
                  "corpus": "gamma"})

    t0 = time.perf_counter()
    BATCH = 64
    for start in range(0, len(facets), BATCH):
        batch = facets[start:start + BATCH]
        texts = [f"passage: {f.text}" for f in batch]
        embs = model.encode(texts, convert_to_numpy=True, batch_size=32,
                            show_progress_bar=False)
        metadatas = []
        for f in batch:
            item = items_by_id[f.item_id]
            metadatas.append({
                "corpus": "gamma",
                "game_version": "0.9.5",
                "item_id": f.item_id,
                "item_name": item.name,
                "category": item.category,
                "facet_type": f.facet_type,
                "manifest_sha256": manifest_sha,
                "source_files": ";".join(f.source_files),
                "unobtainable": int(item.unobtainable),
            })
        col.add(ids=[f.facet_id for f in batch],
                documents=[f.text for f in batch],
                embeddings=[e.tolist() for e in embs],
                metadatas=metadatas)
        done = min(start + BATCH, len(facets))
        print(f"  embedded {done}/{len(facets)} facets", flush=True)
    embed_s = time.perf_counter() - t0

    # validation: exact count + first/last ids + smoke dense query
    if col.count() != len(facets):
        raise builder_mod.BuildError(
            f"collection count {col.count()} != facet count {len(facets)}")
    probe_ids = [facets[0].facet_id, facets[-1].facet_id]
    got = col.get(ids=probe_ids)
    if sorted(got.get("ids", [])) != sorted(probe_ids):
        raise builder_mod.BuildError("collection id probe failed")
    q = model.encode(["query: AK-74 rifle stats"], convert_to_numpy=True)[0].tolist()
    smoke = col.query(query_embeddings=[q], n_results=3, include=["metadatas"])
    if not smoke.get("ids") or not smoke["ids"][0]:
        raise builder_mod.BuildError("smoke dense query returned nothing")
    return {"embed_seconds": round(embed_s, 2)}


def cleanup_stale_collections(chroma_path: Path, keep: set[str]) -> list[str]:
    removed = []
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(chroma_path))
        for col in client.list_collections():
            name = getattr(col, "name", None) or str(col)
            if name.startswith(GAMMA_CHROMA_COLLECTION_PREFIX) and name not in keep:
                client.delete_collection(name)
                removed.append(name)
    except Exception as exc:
        print(f"  (stale-collection cleanup skipped: {exc})", flush=True)
    return removed


def run_build(repo_path: Path, db_path: Path, chroma_path: Path, *,
              force: bool = False, no_embed: bool = False,
              audit_only: bool = False) -> dict:
    t_start = time.perf_counter()
    ds = load_dataset(repo_path)
    upstream_commit = _upstream_commit(repo_path)

    prev_collection = active_collection_name(db_path)
    if not force and not audit_only:
        current = active_manifest_sha(db_path)
        if current == ds.manifest_sha256:
            embedded_ok = no_embed or collection_intact(chroma_path, prev_collection)
            if embedded_ok:
                return {"status": "unchanged", "manifest_sha256": ds.manifest_sha256,
                        "db_path": str(db_path), "active_collection": prev_collection}

    res = builder_mod.build(ds)
    problems = builder_mod.check_audit(res.audit)
    res.audit["upstream_commit"] = upstream_commit
    res.audit["built_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(res.audit, indent=2, ensure_ascii=False))

    if problems:
        return {"status": "failed", "problems": problems,
                "audit_path": str(AUDIT_PATH)}
    if audit_only:
        return {"status": "audit_ok", "audit": res.audit,
                "audit_path": str(AUDIT_PATH)}

    staging = db_path.with_suffix(db_path.suffix + ".staging")
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        staging.unlink()

    timing = {}
    collection_name = None
    try:
        if not no_embed:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d%H%M%S")
            collection_name = (f"{GAMMA_CHROMA_COLLECTION_PREFIX}"
                               f"{ds.manifest_sha256[:8]}_{stamp}")
            items_by_id = {i.item_id: i for i in res.items}
            timing.update(embed_facets(chroma_path, collection_name, res.facets,
                                       ds.manifest_sha256, items_by_id))

        builder_mod.write_database(staging, ds, res, upstream_commit, collection_name)
        os.replace(staging, db_path)   # atomic activation
    except Exception:
        if staging.exists():
            staging.unlink()
        if collection_name:  # remove the incomplete staging collection
            cleanup_stale_collections(chroma_path,
                                      keep={prev_collection} if prev_collection else set())
        raise

    removed = []
    if collection_name:
        keep = {collection_name}
        if prev_collection:
            keep.add(prev_collection)   # retain immediate predecessor
        removed = cleanup_stale_collections(chroma_path, keep)

    timing["build_seconds"] = round(time.perf_counter() - t_start, 2)
    return {"status": "built", "db_path": str(db_path),
            "active_collection": collection_name,
            "previous_collection": prev_collection,
            "removed_collections": removed,
            "manifest_sha256": ds.manifest_sha256,
            "upstream_commit": upstream_commit,
            "item_count": len(res.items), "facet_count": len(res.facets),
            "audit_path": str(AUDIT_PATH), **timing}


def _print_summary(result: dict, audit: dict | None) -> None:
    print(f"[build_gamma_095] status={result['status']}")
    for key in ("db_path", "active_collection", "manifest_sha256", "item_count",
                "facet_count", "embed_seconds", "build_seconds"):
        if key in result and result[key] is not None:
            print(f"  {key}: {result[key]}")
    if audit:
        print(f"  index items: {audit['index_item_count']}  canonical: "
              f"{audit['canonical_item_count']}  aliases: {audit['alias_count']}  "
              f"properties: {audit['property_count']}  relations: {audit['relation_count']}  "
              f"facets: {audit['facet_count']}")
        print(f"  files parsed: {len(audit['files_parsed'])}  skipped: "
              f"{len(audit['files_skipped'])}  missing translations: "
              f"{audit['missing_translation_count']}")
        unresolved = audit.get("unresolved_references", {})
        total_unres = sum(v["count"] for v in unresolved.values())
        print(f"  unresolved references: {total_unres} across {len(unresolved)} files")
    if result.get("problems"):
        for p in result["problems"]:
            print(f"  FATAL: {p}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the canonical GAMMA 0.9.5 corpus")
    parser.add_argument("--repo-path", type=Path, default=None)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument("--chroma-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    repo_path = args.repo_path or get_gamma_repo_path()
    db_path = args.db_path or get_gamma_db_path()
    chroma_path = args.chroma_path or get_chroma_path()

    try:
        result = run_build(repo_path, db_path, chroma_path, force=args.force,
                           no_embed=args.no_embed, audit_only=args.audit_only)
    except (GammaLoadError, builder_mod.BuildError) as exc:
        result = {"status": "failed", "problems": [str(exc)]}
    except Exception as exc:                     # keep active corpus usable
        result = {"status": "failed", "problems": [f"{type(exc).__name__}: {exc}"]}

    audit = result.pop("audit", None)
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_summary(result, audit)
    return 0 if result["status"] in ("built", "unchanged", "audit_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
