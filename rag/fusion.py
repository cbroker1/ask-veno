"""Reciprocal Rank Fusion and FTS query sanitization.

RRF works on ranks, never on raw scores, so BM25 and cosine lists fuse
without score normalization: score(d) = Σ weight_i / (rrf_k + rank_i(d)).
"""

from __future__ import annotations

import re

from app.config import RRF_K

_TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ]+(?:\.[0-9]+)?")


def build_match_query(text: str, max_tokens: int = 12) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression: quoted
    tokens joined by OR. Returns "" when nothing usable remains."""
    tokens = []
    for tok in _TOKEN_RE.findall(text or ""):
        tok = tok.lower()
        if len(tok) == 1 and not tok.isdigit():
            continue
        if tok not in tokens:
            tokens.append(tok)
        if len(tokens) >= max_tokens:
            break
    return " OR ".join(f'"{t}"' for t in tokens)


def rrf_fuse(ranked_lists: list[list[str]], weights: list[float] | None = None,
             rrf_k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked id lists (best first, deduplicated within each list).

    Returns [(id, fused_score)] sorted best-first; ties break on id for
    determinism.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists")
    scores: dict[str, float] = {}
    for weight, ids in zip(weights, ranked_lists):
        seen: set[str] = set()
        rank = 0
        for id_ in ids:
            if id_ in seen:
                continue
            seen.add(id_)
            rank += 1
            scores[id_] = scores.get(id_, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
