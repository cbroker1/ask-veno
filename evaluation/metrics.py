"""Deterministic retrieval/routing metrics for the Phase 2 evaluation.

All metrics operate on plain dicts so both live pipeline output and stored
results can be scored. No LLM judging here — see judges.py for the
deterministic fact/route checks these aggregate.
"""

from __future__ import annotations

import statistics


def recall_any(expected: list[str], got: list[str]) -> float | None:
    """1.0 when any expected id appears in got; None when nothing was
    expected (excluded from averages)."""
    if not expected:
        return None
    got_set = set(got)
    # prefix-tolerant for item families (B05: 'AK-74' -> any wpn_ak74*)
    for want in expected:
        if want in got_set or any(g.startswith(want) for g in got_set):
            return 1.0
    return 0.0


def reciprocal_rank(expected: list[str], got: list[str]) -> float | None:
    if not expected:
        return None
    for rank, g in enumerate(got, 1):
        if g in expected or any(g.startswith(w) for w in expected):
            return 1.0 / rank
    return 0.0


def fact_coverage(required_facts: list[str], text: str) -> float | None:
    """Fraction of required facts present (case-insensitive substring)."""
    if not required_facts:
        return None
    text_l = (text or "").lower()
    hits = sum(1 for f in required_facts if f.lower() in text_l)
    return hits / len(required_facts)


def _clean(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def aggregate(rows: list[dict], keys: list[str]) -> dict:
    """Mean of each metric key over rows, ignoring None; adds n_scored."""
    out = {}
    for key in keys:
        vals = _clean([r.get(key) for r in rows])
        out[key] = round(statistics.mean(vals), 4) if vals else None
        out[f"{key}_n"] = len(vals)
    return out


def latency_summary(values_ms: list[float]) -> dict:
    if not values_ms:
        return {}
    ordered = sorted(values_ms)
    return {
        "mean_ms": round(statistics.mean(ordered), 1),
        "p50_ms": round(ordered[len(ordered) // 2], 1),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
        "max_ms": round(ordered[-1], 1),
    }
