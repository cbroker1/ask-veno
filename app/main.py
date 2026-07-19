"""FastAPI application instance for the Ask Veno dashboard.

Importing this module wires up the routes but never loads the embedding
model — that stays lazy inside rag.embeddings.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.routes import router

app = FastAPI(title="⌈ GAMMA PDA — YouTube RAG ⌋")
app.include_router(router)
