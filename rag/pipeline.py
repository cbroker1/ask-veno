"""Multi-corpus RAG pipeline: routing, retrieval, one corrective retry,
generation.

Public contract (backwards compatible):

    search(query, top_k=5, corpus="auto") -> {
        "chunks":   [...],   # legacy transcript chunk dicts (may be empty)
        "summary":  "...",
        "evidence": [...],   # serialized Evidence from all corpora
        "route":    "transcripts" | "gamma" | "combined",
        "route_reason": "...",
        "corpus_requested": "...",
        "query_used": "...",
        "rewritten": bool,
        "retrieval_debug": {...},
    }

corpus="transcripts" preserves the pre-Phase-2 behavior exactly: legacy
dense retrieval, legacy generation, no hybrid, no rewriting, no reranking.
Corrective RAG is a single controlled retry — the flow is strictly
retrieve -> (maybe rewrite once -> retrieve once) -> generate; no loops.
"""

from __future__ import annotations

import re

from fastapi import HTTPException

from app.config import CORRECTIVE_RAG_ENABLED, FINAL_EVIDENCE_K
from rag import evidence as ev_mod
from rag import generator, reranker, router
from rag.transcript_retriever import retrieve_chunks, retrieve_chunks_hybrid

CORPUS_ALIASES = {
    "auto": "auto", "": "auto", None: "auto",
    "transcripts": "transcripts", "veno": "transcripts", "venoxium": "transcripts",
    "gamma": "gamma",
    "combined": "combined", "both": "combined",
}

WEAK_TRANSCRIPT_SIM = 55.0        # top sim_pct below this = weak evidence
_MODELISH_RE = re.compile(r"\b[a-z]{1,6}-?\d{1,4}\b|\b\d+x\d+\b", re.I)

# Combined-mode budget: GAMMA evidence must not be drowned by the much larger
# transcript corpus.
COMBINED_GAMMA_K = 3


def normalize_corpus(corpus: str | None) -> str:
    c = CORPUS_ALIASES.get((corpus or "").strip().lower())
    if c is None:
        raise HTTPException(422, f"Unknown corpus '{corpus}' "
                            "(use auto, veno, gamma, or combined)")
    return c


def _gamma_retrieve(query: str, top_k: int) -> dict:
    from rag import gamma_retriever
    rerank_fn = reranker.rerank if reranker.RERANKER_ENABLED else None
    return gamma_retriever.retrieve(query, top_k, rerank_fn=rerank_fn)


def _retrieve_for_route(route_used: str, query: str, top_k: int) -> dict:
    """One retrieval pass. Returns {chunks, evidence, exact_items, debug}."""
    if route_used == "transcripts":
        chunks = retrieve_chunks(query, top_k)
        return {"chunks": chunks,
                "evidence": [ev_mod.from_chunk(c) for c in chunks],
                "exact_items": [], "debug": {"mode": "legacy_dense"}}

    if route_used == "gamma":
        g = _gamma_retrieve(query, top_k)
        return {"chunks": [], "evidence": g["evidence"],
                "exact_items": g["exact_items"], "debug": g["debug"]}

    # combined: retrieve the corpora independently, keep both sides
    transcript_k = max(2, top_k - 2)
    debug: dict = {}
    try:
        g = _gamma_retrieve(query, COMBINED_GAMMA_K)
        gamma_ev, exact_items = g["evidence"], g["exact_items"]
        debug["gamma"] = g["debug"]
    except Exception as exc:               # GAMMA side degraded, keep Veno side
        gamma_ev, exact_items = [], []
        debug["gamma"] = {"unavailable": str(exc)}
    chunks = retrieve_chunks_hybrid(query, transcript_k)
    t_ev = [ev_mod.from_chunk(c) for c in chunks]
    if reranker.RERANKER_ENABLED and t_ev:
        t_ev = reranker.rerank(query, t_ev)
        chunks = [ev_mod.to_chunk(e) for e in t_ev if ev_mod.to_chunk(e)]
    debug["transcript_candidates"] = len(chunks)
    return {"chunks": chunks, "evidence": gamma_ev + t_ev,
            "exact_items": exact_items, "debug": debug}


def _strength(route_used: str, result: dict) -> tuple:
    """Deterministic evidence-strength ordering (bigger tuple = stronger)."""
    chunks = result["chunks"]
    evidence = result["evidence"]
    if route_used == "transcripts":
        top = max((c.get("sim_pct") or 0) for c in chunks) if chunks else 0
        return (1 if chunks else 0, top)
    if route_used == "gamma":
        return (1 if evidence else 0, len(result["exact_items"]), len(evidence))
    gamma_side = sum(1 for e in evidence if e.corpus == "gamma")
    veno_side = sum(1 for e in evidence if e.corpus == "transcripts")
    return ((1 if gamma_side else 0) + (1 if veno_side else 0),
            len(result["exact_items"]), len(evidence))


def _is_weak(route_used: str, result: dict, query: str) -> str | None:
    """Return a reason string when evidence is too weak, else None."""
    chunks, evidence = result["chunks"], result["evidence"]
    if route_used == "transcripts":
        if not chunks:
            return "no transcript evidence"
        top = max((c.get("sim_pct") or 0) for c in chunks)
        if top < WEAK_TRANSCRIPT_SIM:
            return f"weak dense similarity (top {top}%)"
        return None
    if route_used == "gamma":
        if not evidence:
            return "no GAMMA evidence"
        if _MODELISH_RE.search(query) and not result["exact_items"]:
            return "query names a model-like term but no exact item matched"
        return None
    gamma_side = any(e.corpus == "gamma" for e in evidence)
    veno_side = any(e.corpus == "transcripts" for e in evidence)
    if not gamma_side and not veno_side:
        return "no evidence from either corpus"
    if not gamma_side:
        return "combined query but no GAMMA evidence"
    if not veno_side:
        return "combined query but no transcript evidence"
    return None


def search(query, top_k=5, corpus="auto"):
    corpus_requested = normalize_corpus(corpus)
    route_reason = f"manual {corpus_requested} mode"
    exact_hint: list = []

    if corpus_requested == "auto":
        decision = router.route(query)
        route_used = decision.corpus
        route_reason = decision.reason
        exact_hint = decision.exact_items
    else:
        route_used = corpus_requested

    debug: dict = {"router_exact_items": [i["item_id"] for i in exact_hint]}

    # Manual transcripts mode is the frozen legacy path: retrieve + summarize
    # exactly as pre-Phase-2 (no corrective, no hybrid, no reranking).
    if corpus_requested == "transcripts":
        chunks = retrieve_chunks(query, top_k)
        summary = ""
        if chunks and generator.OLLAMA_AVAILABLE:
            try:
                summary = generator.llm_answer(query, chunks)
            except Exception:
                summary = ""
        return {"chunks": chunks, "summary": summary,
                "evidence": [ev_mod.from_chunk(c).to_dict() for c in chunks],
                "route": "transcripts", "route_reason": route_reason,
                "corpus_requested": corpus_requested,
                "query_used": query, "rewritten": False,
                "retrieval_debug": {"mode": "legacy_dense"}}

    # GAMMA availability degradation
    if route_used in ("gamma", "combined"):
        from storage.gamma import GammaUnavailable
        try:
            result = _retrieve_for_route(route_used, query, top_k)
        except GammaUnavailable as exc:
            if corpus_requested == "gamma":
                raise HTTPException(503, "GAMMA 0.9.5 corpus unavailable — "
                                    "run scripts/build_gamma_095.py "
                                    f"({exc})")
            if corpus_requested == "combined":
                route_used = "combined"
                chunks = retrieve_chunks_hybrid(query, top_k)
                result = {"chunks": chunks,
                          "evidence": [ev_mod.from_chunk(c) for c in chunks],
                          "exact_items": [],
                          "debug": {"gamma": {"unavailable": str(exc)}}}
            else:  # auto route degraded
                route_used = "transcripts"
                route_reason += " (GAMMA corpus unavailable — fell back)"
                chunks = retrieve_chunks(query, top_k)
                result = {"chunks": chunks,
                          "evidence": [ev_mod.from_chunk(c) for c in chunks],
                          "exact_items": [], "debug": {"mode": "legacy_dense"}}
    else:
        result = _retrieve_for_route(route_used, query, top_k)

    # ── one controlled corrective retry ─────────────────────────────────
    query_used, rewritten = query, False
    weak = _is_weak(route_used, result, query)
    if weak and CORRECTIVE_RAG_ENABLED:
        from rag.query_rewriter import rewrite_query
        candidate = rewrite_query(query)
        debug["weak_reason"] = weak
        if candidate:
            debug["rewrite_candidate"] = candidate
            try:
                retry = _retrieve_for_route(route_used, candidate, top_k)
            except Exception:
                retry = None
            if retry and _strength(route_used, retry) > _strength(route_used, result):
                result = retry
                query_used, rewritten = candidate, True

    evidence = result["evidence"]
    chunks = result["chunks"]
    debug.update(result["debug"])
    debug["final_evidence"] = len(evidence)
    debug["corpus_counts"] = {
        "gamma": sum(1 for e in evidence if e.corpus == "gamma"),
        "transcripts": sum(1 for e in evidence if e.corpus == "transcripts"),
    }
    if reranker.last_latency_ms is not None:
        debug["rerank_ms"] = reranker.last_latency_ms

    summary = ""
    if evidence and generator.OLLAMA_AVAILABLE:
        try:
            if route_used == "transcripts":
                summary = generator.llm_answer(query_used, chunks)
            else:
                summary = generator.llm_answer_multicorpus(query_used, evidence)
        except Exception:
            summary = ""

    return {"chunks": chunks, "summary": summary,
            "evidence": [e.to_dict() for e in evidence],
            "route": route_used, "route_reason": route_reason,
            "corpus_requested": corpus_requested,
            "query_used": query_used, "rewritten": rewritten,
            "retrieval_debug": debug}
