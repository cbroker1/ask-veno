"""RAG pipeline: retrieve transcript chunks, then synthesize a summary.

Coordinates the current single-corpus flow (transcript retrieval + local LLM
answer) and returns the same {"chunks": [...], "summary": "..."} structure
the routes have always consumed. This module is the extension point for
future corpora and routing.
"""

from __future__ import annotations

from rag import generator
from rag.transcript_retriever import retrieve_chunks


def search(query, top_k=5):
    raw_chunks = retrieve_chunks(query, top_k)
    summary = ""
    if raw_chunks and generator.OLLAMA_AVAILABLE:
        try:
            summary = generator.llm_answer(query, raw_chunks)
        except Exception:
            summary = ""
    return {"chunks": raw_chunks, "summary": summary}
