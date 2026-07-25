"""GAMMA 0.9.5 retrieval: exact item lookup first, then hybrid search.

Layer order (per the Phase 2 design):
  1. exact normalized alias/name lookup (n-gram variants of the query)
  2. structured facts for exactly-resolved items (their canonical facets)
  3. BM25 lexical search over facet text (SQLite FTS5)
  4. dense semantic search (versioned Chroma collection, E5)
  5. reciprocal-rank fusion of 3+4
  6. optional cross-encoder reranking of the fused tail (exact facets are
     protected and never reranked away)

Every layer degrades gracefully: no database -> GammaUnavailable (caller
reports corpus unavailable); no Chroma collection -> lexical/structured only;
FTS failure -> dense only.
"""

from __future__ import annotations

import re
import time

from app.config import (
    FINAL_EVIDENCE_K,
    HYBRID_BM25_K,
    HYBRID_DENSE_K,
    get_chroma_path,
)
from indexing.gamma.normalize import compact_name, norm_name
from rag import evidence as ev_mod
from rag.fusion import build_match_query, rrf_fuse
from storage.gamma import GammaDB, GammaUnavailable

__all__ = ["retrieve", "resolve_exact_items", "GammaUnavailable"]

MAX_EXACT_ITEMS = 2        # distinct items resolved per query
MAX_EXACT_FACETS = 3       # protected facets contributed by exact items
_FACET_PRIORITY = ["stats", "overview", "compatibility", "crafting",
                   "disassembly", "obtaining", "upgrades", "parts"]


_QUERY_TOKEN_RE = re.compile(r"[0-9a-zа-яё]+(?:[.\-][0-9a-zа-яё]+)*")


def _query_variants(query: str, max_n: int = 6) -> list[tuple[str, str]]:
    """Deterministic (norm, compact) candidate fragments: the whole query
    plus every contiguous token n-gram, longest first. Tokens are stripped of
    surrounding punctuation ("SVD?" -> "svd") while inner dots/hyphens stay
    ("ak-74", "5.45x39")."""
    norm = norm_name(query)
    tokens = _QUERY_TOKEN_RE.findall(norm)
    variants: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(fragment: str):
        n, c = fragment, compact_name(fragment)
        if len(c) < 3 and not c.isdigit():
            return
        if n not in seen:
            seen.add(n)
            variants.append((n, c))

    add(norm)
    for n in range(min(len(tokens), max_n), 0, -1):
        for i in range(len(tokens) - n + 1):
            add(" ".join(tokens[i:i + n]))
    return variants


def resolve_exact_items(db: GammaDB, query: str,
                        limit: int = MAX_EXACT_ITEMS) -> list[dict]:
    """Resolve items named in the query via exact alias match, falling back
    to compact-prefix match on the longest fragments. Deterministic order."""
    variants = _query_variants(query)
    hits = db.lookup_exact(variants, limit=limit * 4)
    if not hits:
        # Prefix fallback exists to catch model numbers ("AK-74" -> AK-74N,
        # "tak-7" -> TAK-74), so it only fires on digit-bearing fragments —
        # plain words like "join" or "ammo" must never prefix-match item ids.
        for _, compact in variants:
            if len(compact) < 4 or not any(ch.isdigit() for ch in compact):
                continue
            hits = db.lookup_prefix(compact, limit=limit * 4)
            if hits:
                break
    out, seen = [], set()
    for h in hits:
        if h["item_id"] in seen:
            continue
        seen.add(h["item_id"])
        out.append(h)
        if len(out) >= limit:
            break
    return out


def _exact_facet_evidence(db: GammaDB, items: list[dict], query: str) -> list:
    """Protected evidence from exactly-resolved items: their facets ranked by
    query-token overlap, then by canonical facet priority."""
    q_tokens = {t for t in norm_name(query).split() if len(t) > 2}
    scored = []
    for item in items:
        for facet in db.facets(item["item_id"]):
            text_l = facet["text"].lower()
            overlap = sum(1 for t in q_tokens if t in text_l)
            prio = (_FACET_PRIORITY.index(facet["facet_type"])
                    if facet["facet_type"] in _FACET_PRIORITY else 99)
            scored.append((-overlap, prio, facet["facet_id"], facet, item))
    scored.sort(key=lambda s: s[:3])
    out = []
    for _, _, _, facet, item in scored[:MAX_EXACT_FACETS]:
        out.append(ev_mod.from_gamma_facet(facet, item, score=1.0,
                                           source_type="gamma_exact"))
    return out


def _dense_search(db: GammaDB, query: str, k: int, debug: dict) -> list[dict]:
    collection_name = db.active_collection()
    if not collection_name:
        debug["dense_skipped"] = "no active collection"
        return []
    chroma_dir = get_chroma_path()
    if not chroma_dir.exists():
        debug["dense_skipped"] = "chroma dir missing"
        return []
    try:
        import chromadb

        from rag.embeddings import get_embedding_model
        emb = get_embedding_model().encode([f"query: {query}"],
                                           convert_to_numpy=True)[0].tolist()
        client = chromadb.PersistentClient(path=str(chroma_dir))
        col = client.get_collection(collection_name)
        res = col.query(query_embeddings=[emb], n_results=k,
                        include=["metadatas", "documents", "distances"])
    except Exception as exc:
        debug["dense_skipped"] = f"{type(exc).__name__}: {exc}"
        return []
    out = []
    if res.get("ids") and res["ids"][0]:
        for i, fid in enumerate(res["ids"][0]):
            dist = (res.get("distances") or [[None]])[0][i]
            sim = max(0.0, 1 - dist) if dist is not None else 0.0
            out.append({"facet_id": fid,
                        "text": (res.get("documents") or [[""]])[0][i],
                        "metadata": (res.get("metadatas") or [[{}]])[0][i],
                        "sim": sim})
    return out


def retrieve(query: str, top_k: int = FINAL_EVIDENCE_K, *,
             dense_k: int = HYBRID_DENSE_K, bm25_k: int = HYBRID_BM25_K,
             db_path=None, rerank_fn=None) -> dict:
    """Full GAMMA retrieval. Raises GammaUnavailable when the canonical
    database is missing. Returns {"evidence", "exact_items", "debug"}."""
    debug: dict = {}
    t0 = time.perf_counter()
    db = GammaDB(db_path)
    try:
        exact_items = resolve_exact_items(db, query)
        debug["exact_items"] = [i["item_id"] for i in exact_items]
        protected = _exact_facet_evidence(db, exact_items, query)
        t_exact = time.perf_counter()

        bm25_rows = (db.fts_search(build_match_query(query), limit=bm25_k)
                     if bm25_k > 0 else [])
        t_bm25 = time.perf_counter()
        dense_rows = (_dense_search(db, query, dense_k, debug)
                      if dense_k > 0 else [])
        t_dense = time.perf_counter()

        fused = rrf_fuse([[r["facet_id"] for r in dense_rows],
                          [r["facet_id"] for r in bm25_rows]])
        by_id: dict[str, dict] = {}
        for r in bm25_rows:
            by_id.setdefault(r["facet_id"], {})["bm25"] = r
        for r in dense_rows:
            by_id.setdefault(r["facet_id"], {})["dense"] = r

        protected_ids = {e.id for e in protected}
        candidates = []
        for fid, fscore in fused:
            if fid in protected_ids:
                continue
            facet = db.facet(fid)
            if not facet:
                continue
            item = db.item(facet["item_id"]) or {}
            e = ev_mod.from_gamma_facet(facet, item, score=fscore)
            dense_info = by_id.get(fid, {}).get("dense")
            if dense_info:
                e.metadata["sim_pct"] = round(dense_info["sim"] * 100, 1)
            if by_id.get(fid, {}).get("bm25") is not None:
                e.metadata["bm25"] = True
            candidates.append(e)

        if rerank_fn is not None and candidates:
            candidates = rerank_fn(query, candidates)
        t_fuse = time.perf_counter()

        evidence = (protected + candidates)[:max(top_k, len(protected))]
        debug.update({
            "dense_candidates": len(dense_rows),
            "bm25_candidates": len(bm25_rows),
            "fused_candidates": len(fused),
            "protected_exact_facets": len(protected),
            "final": len(evidence),
            "timings_ms": {
                "exact": round((t_exact - t0) * 1000, 1),
                "bm25": round((t_bm25 - t_exact) * 1000, 1),
                "dense": round((t_dense - t_bm25) * 1000, 1),
                "fuse_rerank": round((t_fuse - t_dense) * 1000, 1),
            },
        })
        return {"evidence": evidence, "exact_items": exact_items, "debug": debug}
    finally:
        db.close()
