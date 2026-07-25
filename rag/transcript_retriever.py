"""ChromaDB transcript retrieval.

Encodes the query with the E5 `query: ` prefix, queries the persistent
collection, and maps raw Chroma results into the chunk dictionaries the rest
of the app consumes. Raises the same HTTPExceptions as the original code so
route error behavior (JSON detail for /api/search, fault-card text for
/search) is unchanged.
"""

from __future__ import annotations

import chromadb
from fastapi import HTTPException

from app.config import CHROMA_COLLECTION, get_chroma_path
from rag.embeddings import get_embedding_model


def retrieve_chunks(query, top_k=5):
    cb = get_chroma_path()
    if not cb.exists(): raise HTTPException(500, "ChromaDB not found")
    model = get_embedding_model()
    emb = model.encode([f"query: {query}"], convert_to_numpy=True)[0].tolist()
    client = chromadb.PersistentClient(path=str(cb))
    try: col = client.get_collection(CHROMA_COLLECTION)
    except Exception as e: raise HTTPException(500, f"Collection not found: {e}")
    res = col.query(query_embeddings=[emb], n_results=top_k,
                     include=["metadatas", "documents", "distances"])
    raw_chunks = []
    if res["ids"] and res["ids"][0]:
        docs = res.get("documents")
        if docs and docs[0]:
            for i in range(len(res["ids"][0])):
                md = res["metadatas"][0][i]
                dist = res.get("distances", [[None]])[0][i]
                sim = max(0, 1 - dist) if dist else 0
                raw_chunks.append({"id": res["ids"][0][i], "distance": dist,
                    "sim_pct": round(sim*100,1),
                    "vt": md.get("video_title","Unknown")[:100],
                    "vid": md.get("video_id","Unknown"),
                    "s": md.get("start_hms","N/A"),
                    "e": md.get("end_hms","N/A"),
                    "url": md.get("youtube_time_url","N/A"),
                    "txt": docs[0][i][:1200]})
    return raw_chunks


def _lexical_row_to_chunk(row: dict) -> dict:
    """Map a lexical-index row into the legacy chunk shape. BM25-only hits
    have no dense distance, so distance is None and sim_pct 0 (the existing
    convention for unknown similarity)."""
    return {"id": row["chunk_id"], "distance": None, "sim_pct": 0,
            "vt": (row.get("video_title") or "Unknown")[:100],
            "vid": row.get("video_id") or "Unknown",
            "s": row.get("start_hms") or "N/A",
            "e": row.get("end_hms") or "N/A",
            "url": row.get("url") or "N/A",
            "txt": (row.get("text") or "")[:1200]}


def retrieve_chunks_hybrid(query, top_k=5, dense_k=20, bm25_k=20):
    """Hybrid transcript retrieval: dense + BM25 fused with RRF.

    Used by combined/adaptive modes only — the plain retrieve_chunks() dense
    path above stays byte-identical for corpus="transcripts". Falls back to
    dense-only when the lexical index is missing or the FTS query fails.
    """
    from rag.fusion import build_match_query, rrf_fuse
    from storage.lexical import LexicalDB, LexicalUnavailable

    dense = retrieve_chunks(query, dense_k)
    try:
        lex = LexicalDB()
        try:
            lex_rows = lex.fts_search(build_match_query(query), limit=bm25_k)
        finally:
            lex.close()
    except LexicalUnavailable:
        lex_rows = []
    if not lex_rows:
        return dense[:top_k]

    by_id = {c["id"]: c for c in dense}
    for row in lex_rows:
        by_id.setdefault(row["chunk_id"], _lexical_row_to_chunk(row))
    fused = rrf_fuse([[c["id"] for c in dense],
                      [r["chunk_id"] for r in lex_rows]])
    return [by_id[cid] for cid, _ in fused[:top_k] if cid in by_id]
