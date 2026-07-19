"""Environment configuration for the Ask Veno web app.

Defaults and environment-variable names are unchanged from the original
web_app.py. SQLITE_DB_PATH and CHROMA_PATH are re-read from the environment
on every call (matching the original per-request behavior), while
CHROMA_COLLECTION and EMBED_MODEL_NAME are fixed once at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = ".state/youtube_ingest.sqlite"
DEFAULT_CHROMA_PATH = "data/chroma"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "youtube_chunks")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-large")

# Local Ollama generation settings. These have always been fixed constants in
# the web app — the OLLAMA_*/QWEN_* variables in .env belong to the ingestion
# scripts, not to this application.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "qwen3:0.6b"
OLLAMA_PROBE_TIMEOUT_S = 2
OLLAMA_GENERATE_TIMEOUT_S = 120
OLLAMA_NUM_PREDICT = 1024
OLLAMA_TEMPERATURE = 0.2


def get_db_path() -> Path:
    return Path(os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH))


def get_chroma_path() -> Path:
    return Path(os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH))
