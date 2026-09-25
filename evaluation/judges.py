"""Deterministic per-record judges (no LLM).

Each judge takes the gold record plus observed pipeline output and returns
scores used by run_evaluation.py. Kept separate from metrics.py so the
"what counts as correct" rules live in one reviewable place.
"""

from __future__ import annotations

from evaluation.metrics import fact_coverage, recall_any, reciprocal_rank


def judge_route(record: dict, route_used: str) -> float:
    return 1.0 if route_used == record["expected_route"] else 0.0


def gamma_item_ids(evidence: list[dict]) -> list[str]:
    """Ordered distinct GAMMA item ids from serialized evidence."""
    out: list[str] = []
    for e in evidence:
        if e.get("corpus") != "gamma":
            continue
        iid = (e.get("metadata") or {}).get("item_id")
        if iid and iid not in out:
            out.append(iid)
    return out


def transcript_video_ids(result: dict) -> list[str]:
    out: list[str] = []
    for c in result.get("chunks", []):
        vid = (c.get("vid") or "").removeprefix("youtube_id:")
        if vid and vid not in out:
            out.append(vid)
    return out


def evidence_text(result: dict) -> str:
    parts = [e.get("text") or "" for e in result.get("evidence", [])]
    parts += [c.get("txt") or "" for c in result.get("chunks", [])]
    return "\n".join(parts)


def judge_record(record: dict, result: dict) -> dict:
    """Full deterministic scoring of one pipeline result against gold."""
    items = gamma_item_ids(result.get("evidence", []))
    vids = transcript_video_ids(result)
    ev_text = evidence_text(result)
    corpus_gamma = any(e.get("corpus") == "gamma"
                       for e in result.get("evidence", []))
    corpus_transcripts = bool(result.get("chunks")) or any(
        e.get("corpus") == "transcripts" for e in result.get("evidence", []))

    scores = {
        "route_correct": judge_route(record, result.get("route", "")),
        "item_recall": recall_any(record.get("expected_item_ids", []), items),
        "item_mrr": reciprocal_rank(record.get("expected_item_ids", []), items),
        "video_recall": recall_any(record.get("expected_video_ids", []), vids),
        "video_mrr": reciprocal_rank(record.get("expected_video_ids", []), vids),
        "fact_in_evidence": fact_coverage(record.get("required_facts", []), ev_text),
        "fact_in_answer": fact_coverage(record.get("required_facts", []),
                                        result.get("summary", ""))
        if result.get("summary") else None,
        "empty_result": 1.0 if not result.get("evidence") and not result.get("chunks")
        else 0.0,
        "rewritten": 1.0 if result.get("rewritten") else 0.0,
    }
    # combined-coverage: did we honour both requires_* flags?
    if record.get("requires_gamma") and record.get("requires_transcript"):
        scores["combined_coverage"] = 1.0 if (corpus_gamma and corpus_transcripts) else 0.0
    else:
        scores["combined_coverage"] = None
    return scores
