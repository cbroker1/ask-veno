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
