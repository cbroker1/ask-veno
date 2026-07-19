"""Local Ollama generation: availability probe and answer synthesis.

OLLAMA_AVAILABLE is probed once at import time, exactly as the original
web_app.py did — the topbar SYNTH LED and the generation gate both key off
this process-lifetime flag. `requests` is imported inside try blocks so a
missing/broken requests install degrades to "synth offline" instead of
crashing the app, matching the original behavior.
"""

from __future__ import annotations

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_GENERATE_TIMEOUT_S,
    OLLAMA_MODEL,
    OLLAMA_NUM_PREDICT,
    OLLAMA_PROBE_TIMEOUT_S,
    OLLAMA_TEMPERATURE,
)

OLLAMA_AVAILABLE = False
try:
    import requests as _req
    _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=OLLAMA_PROBE_TIMEOUT_S)
    OLLAMA_AVAILABLE = True
    del _req
except Exception:
    pass


def llm_answer(query: str, chunks: list) -> str:
    """Ask ollama to answer from the retrieved chunks."""
    try:
        import requests as _req
        resp = _req.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=OLLAMA_PROBE_TIMEOUT_S)
        models = resp.json().get("models", [])
        if not any(OLLAMA_MODEL in m["name"] for m in models):
            return ""
        available_name = OLLAMA_MODEL
    except Exception:
        return ""
    ctx_lines = []
    for c in chunks:
        ctx_lines.append(f"[{c['s']}-{c['e']}] {c['vt']} ({c['vid']}): {c['txt']}")
    context = "\n\n---\n\n".join(ctx_lines)
    system = (
        "You are a knowledge base assistant answering questions from YouTube video transcripts. "
        "Answer directly and concisely in 3-5 sentences. Use only the provided chunks. "
        "DO NOT just list videos — summarize the answer to the question."
    )
    prompt = f"Question: {query}\n\nSource chunks:\n{context}\n\nAnswer:"
    try:
        import requests as _req2
        resp = _req2.post(f"{OLLAMA_BASE_URL}/api/generate", json={
            "model": available_name,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"num_predict": OLLAMA_NUM_PREDICT, "temperature": OLLAMA_TEMPERATURE}
        }, timeout=OLLAMA_GENERATE_TIMEOUT_S)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception:
        return ""
    return ""
