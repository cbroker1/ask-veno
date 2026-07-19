#!/usr/bin/env python3
"""
Stalker Gamma YouTube RAG Dashboard — Stalker-themed web UI.

Compatibility entrypoint. The implementation lives in:
    app/           FastAPI instance, configuration, routes
    rag/           embeddings, Chroma retrieval, Ollama generation, pipeline
    storage/       SQLite archive reads
    presentation/  formatters, template engine, renderer, page template

Both historical launch modes keep working:
    python web_app.py
    uvicorn web_app:app
"""

from app.config import CHROMA_COLLECTION, EMBED_MODEL_NAME, get_chroma_path, get_db_path
from app.main import app
from rag.generator import OLLAMA_AVAILABLE

__all__ = ["app"]


if __name__ == "__main__":
    import uvicorn
    print("=" * 62)
    print("  ⌈ ASK VENO ⌋ — Zone Archive Terminal (S.T.A.L.K.E.R. GAMMA)")
    print("=" * 62)
    print(f"  DB       : {get_db_path()}")
    print(f"  ChromaDB : {get_chroma_path()}  [{CHROMA_COLLECTION}]")
    print(f"  Embedder : {EMBED_MODEL_NAME}")
    print(f"  Synth    : {'ONLINE' if OLLAMA_AVAILABLE else 'OFFLINE'} (local Ollama)")
    print(f"  Terminal : http://localhost:9876")
    print("=" * 62)
    uvicorn.run(app, host="0.0.0.0", port=9876, log_level="info")
