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


# ── GAMMA 0.9.5 corpus (Phase 2) ─────────────────────────────────────────────
# The pack id is a hard pin: ingestion refuses any other GAMMA version. It is
# intentionally NOT environment-overridable.
GAMMA_PACK_ID = "gamma-0.9.5"
GAMMA_UPSTREAM_URL = "https://github.com/simonwdev/stalker-anomaly-gamma-db"
GAMMA_UPSTREAM_LICENSE = "AGPL-3.0"
# Generated dataset tree inside the upstream repository.
GAMMA_DATA_SUBDIR = f"site/public/data/{GAMMA_PACK_ID}"

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Sibling checkout next to the ask-veno repository, never inside it.
DEFAULT_GAMMA_REPO_PATH = _REPO_ROOT.parent / "stalker-anomaly-gamma-db-upstream"
DEFAULT_GAMMA_DB_PATH = ".state/gamma_095.sqlite"
DEFAULT_LEXICAL_DB_PATH = ".state/lexical_search.sqlite"
GAMMA_CHROMA_COLLECTION_PREFIX = "gamma_items_095_"

# Hybrid retrieval tuning (evaluated defaults; see evaluation/).
HYBRID_DENSE_K = 20
HYBRID_BM25_K = 20
RRF_K = 60
FINAL_EVIDENCE_K = 6

# Optional cross-encoder reranker (lazy-loaded, CPU).
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "1") not in ("0", "false", "no")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20

# One controlled corrective-RAG retry (rewrite + re-retrieve), never more.
CORRECTIVE_RAG_ENABLED = True
CORRECTIVE_MAX_RETRIES = 1


def get_gamma_repo_path() -> Path:
    return Path(os.getenv("GAMMA_REPO_PATH", str(DEFAULT_GAMMA_REPO_PATH)))


def get_gamma_db_path() -> Path:
    return Path(os.getenv("GAMMA_DB_PATH", DEFAULT_GAMMA_DB_PATH))


def get_lexical_db_path() -> Path:
    return Path(os.getenv("LEXICAL_DB_PATH", DEFAULT_LEXICAL_DB_PATH))
