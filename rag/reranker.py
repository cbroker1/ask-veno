"""Optional local cross-encoder reranking (lazy, CPU, fail-open).

The model is only loaded on first use; if sentence-transformers or the
weights are unavailable the reranker reports unavailable and callers keep
the fused order. Exact GAMMA facts are protected upstream (they are never
passed through the reranker), and the candidate set is bounded by
RERANK_CANDIDATES.
"""

from __future__ import annotations

import time

from app.config import RERANK_CANDIDATES, RERANKER_ENABLED, RERANKER_MODEL

_model = None
_load_failed = False
last_latency_ms: float | None = None


def _get_model():
    global _model, _load_failed
    if _model is None and not _load_failed:
        try:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(RERANKER_MODEL, device="cpu")
        except Exception:
            _load_failed = True
    return _model


def available() -> bool:
    return RERANKER_ENABLED and _get_model() is not None


def rerank(query: str, evidence: list, top_n: int | None = None,
           force: bool = False) -> list:
    """Reorder Evidence objects by cross-encoder relevance. Returns the input
    unchanged (fused order) when disabled (unless force=True, used by the
    evaluation ablations) or on any failure."""
    global last_latency_ms
    last_latency_ms = None
    if (not RERANKER_ENABLED and not force) or not evidence:
        return evidence
    model = _get_model()
    if model is None:
        return evidence
    bounded = evidence[:RERANK_CANDIDATES]
    rest = evidence[RERANK_CANDIDATES:]
    try:
        t0 = time.perf_counter()
        scores = model.predict([(query, e.text[:1000]) for e in bounded])
        last_latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    except Exception:
        return evidence
    order = sorted(range(len(bounded)),
                   key=lambda i: (-float(scores[i]), bounded[i].id))
    reranked = []
    for rank, i in enumerate(order):
        e = bounded[i]
        e.metadata["rerank_score"] = round(float(scores[i]), 4)
        e.metadata["rerank_pos"] = rank
        reranked.append(e)
    out = reranked + rest
    return out[:top_n] if top_n else out
