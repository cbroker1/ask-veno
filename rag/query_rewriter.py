"""One-shot retrieval-oriented query rewriting via local Ollama.

Called ONLY when initial retrieval is judged weak (see pipeline). Strict
contract: the model must return {"query": "..."} — one rewrite, no answer.
Any failure (Ollama down, bad JSON, empty/identical output) returns None and
the caller keeps the original query. Temperature 0 for determinism.
"""

from __future__ import annotations

import json
import re

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_PROBE_TIMEOUT_S,
)

REWRITE_TIMEOUT_S = 30
MAX_REWRITE_CHARS = 300

SYSTEM = (
    "You rewrite search queries for a STALKER GAMMA retrieval system. "
    "Return ONLY a JSON object {\"query\": \"...\"} with ONE better retrieval query. "
    "Keep every item name, weapon model, caliber, location, and the word Venoxium "
    "exactly as given. Expand abbreviations and add likely game terminology. "
    "Never answer the question. Never invent game facts."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def rewrite_query(query: str) -> str | None:
    """Return a single rewritten query, or None to keep the original."""
    try:
        import requests as _req
        resp = _req.post(f"{OLLAMA_BASE_URL}/api/generate", json={
            "model": OLLAMA_MODEL,
            "prompt": f"Rewrite this retrieval query: {query}",
            "system": SYSTEM,
            "stream": False,
            "format": "json",
            "options": {"num_predict": 200, "temperature": 0.0},
        }, timeout=REWRITE_TIMEOUT_S)
        if resp.status_code != 200:
            return None
        raw = _THINK_RE.sub("", resp.json().get("response", "")).strip()
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    rewritten = data.get("query")
    if not isinstance(rewritten, str):
        return None
    rewritten = rewritten.strip()[:MAX_REWRITE_CHARS]
    if not rewritten or rewritten.lower() == query.strip().lower():
        return None
    return rewritten


def ollama_reachable() -> bool:
    try:
        import requests as _req
        _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=OLLAMA_PROBE_TIMEOUT_S)
        return True
    except Exception:
        return False
