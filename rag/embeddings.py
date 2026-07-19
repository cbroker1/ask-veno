"""Embedding-model lifecycle: lazy singleton, CPU-only.

The model is only instantiated on first use (first search), never at import
time, so importing the app stays fast and unit tests never pull the weights.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.config import EMBED_MODEL_NAME

embedding_model: SentenceTransformer | None = None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device='cpu')
    return embedding_model
