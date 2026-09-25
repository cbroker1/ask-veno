#!/usr/bin/env python3
"""
run_evaluation.py -- Phase 2 retrieval/routing evaluation harness.

Runs the gold dataset (evaluation/dataset.jsonl — every record grounded in
actual corpus data) against the live local corpora and reports deterministic
metrics. This is an integration harness: it needs the built GAMMA database,
the transcript Chroma collection, the lexical index, and (for dense search)
the E5 model. Generation is optional.

MODES
    --mode routing              router-only accuracy (fast, no models)
    --mode gamma-ablation       A-D records through GAMMA retrieval variants:
                                dense / bm25 / hybrid / hybrid+rerank / +exact
    --mode transcript-ablation  E records: legacy dense vs hybrid (BM25+RRF)
    --mode full                 every record through pipeline.search(auto)
    --mode all                  all of the above (default)

OPTIONS
    --with-generation           include Ollama answers in `full` mode
    --k N                       evidence size (default 6)
    --out PATH                  results JSON (default evaluation/results/eval_<ts>.json)

USAGE
    python evaluation/run_evaluation.py --mode routing
    python evaluation/run_evaluation.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import judges, metrics  # noqa: E402

DATASET = ROOT / "evaluation" / "dataset.jsonl"
RESULTS_DIR = ROOT / "evaluation" / "results"


def load_dataset() -> list[dict]:
    records = []
    with DATASET.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def eval_routing(records: list[dict]) -> dict:
    from rag.router import route
    rows = []
    for rec in records:
        decision = route(rec["question"])
        rows.append({"id": rec["id"], "category": rec["category"],
                     "expected": rec["expected_route"], "got": decision.corpus,
                     "reason": decision.reason,
                     "route_correct": 1.0 if decision.corpus == rec["expected_route"] else 0.0})
    agg = metrics.aggregate(rows, ["route_correct"])
    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        by_cat[cat] = metrics.aggregate(cat_rows, ["route_correct"])["route_correct"]
    mistakes = [r for r in rows if not r["route_correct"]]
    return {"overall": agg, "by_category": by_cat, "mistakes": mistakes,
            "rows": rows}


def _score_gamma(rec: dict, evidence, latency_ms: float) -> dict:
    ev_dicts = [e.to_dict() for e in evidence]
    items = judges.gamma_item_ids(ev_dicts)
    return {
        "id": rec["id"], "category": rec["category"], "latency_ms": latency_ms,
        "item_recall": metrics.recall_any(rec["expected_item_ids"], items),
        "item_mrr": metrics.reciprocal_rank(rec["expected_item_ids"], items),
        "fact_in_evidence": metrics.fact_coverage(
            rec["required_facts"], "\n".join(e.get("text") or "" for e in ev_dicts)),
        "non_empty": 1.0 if ev_dicts else 0.0,
    }


def eval_gamma_ablation(records: list[dict], k: int) -> dict:
    from rag import gamma_retriever, reranker
    gamma_records = [r for r in records if r["category"] in "ABCD"]
    def _forced_rerank(query, evidence):
        return reranker.rerank(query, evidence, force=True)

    variants = {
        "dense_only": dict(dense_k=20, bm25_k=0, rerank_fn=None),
        "bm25_only": dict(dense_k=0, bm25_k=20, rerank_fn=None),
        "hybrid": dict(dense_k=20, bm25_k=20, rerank_fn=None),
        "hybrid_rerank": dict(dense_k=20, bm25_k=20, rerank_fn=_forced_rerank),
    }
    out: dict = {}
    for name, kw in variants.items():
        rows = []
        for rec in gamma_records:
            t0 = time.perf_counter()
            res = gamma_retriever.retrieve(rec["question"], k, **kw)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({**_score_gamma(rec, res["evidence"], ms),
                         "exact_hit": 1.0 if res["exact_items"] else 0.0,
                         "rerank_ms": reranker.last_latency_ms})
        agg = metrics.aggregate(rows, ["item_recall", "item_mrr",
                                       "fact_in_evidence", "non_empty",
                                       "exact_hit"])
        agg.update(metrics.latency_summary([r["latency_ms"] for r in rows]))
        rerank_ms = [r["rerank_ms"] for r in rows if r.get("rerank_ms")]
        if rerank_ms:
            agg["rerank"] = metrics.latency_summary(rerank_ms)
        out[name] = {"aggregate": agg, "rows": rows}
    return out


def eval_transcript_ablation(records: list[dict], k: int) -> dict:
    from rag.transcript_retriever import retrieve_chunks, retrieve_chunks_hybrid
    e_records = [r for r in records if r["category"] == "E"]
    systems = {"legacy_dense": lambda q: retrieve_chunks(q, k),
               "hybrid_bm25_rrf": lambda q: retrieve_chunks_hybrid(q, k)}
    out: dict = {}
    for name, fn in systems.items():
        rows = []
        for rec in e_records:
            t0 = time.perf_counter()
            chunks = fn(rec["question"])
            ms = (time.perf_counter() - t0) * 1000
            vids = judges.transcript_video_ids({"chunks": chunks})
            rows.append({
                "id": rec["id"], "latency_ms": ms,
                "video_recall": metrics.recall_any(rec["expected_video_ids"], vids),
                "video_mrr": metrics.reciprocal_rank(rec["expected_video_ids"], vids),
                "non_empty": 1.0 if chunks else 0.0,
            })
        agg = metrics.aggregate(rows, ["video_recall", "video_mrr", "non_empty"])
        agg.update(metrics.latency_summary([r["latency_ms"] for r in rows]))
        out[name] = {"aggregate": agg, "rows": rows}
    return out


def eval_full(records: list[dict], k: int, with_generation: bool) -> dict:
    from rag import generator, pipeline
    if not with_generation:
        # keep evaluation deterministic/fast: skip Ollama generation
        original = generator.OLLAMA_AVAILABLE
        generator.OLLAMA_AVAILABLE = False
    rows = []
    try:
        for rec in records:
            t0 = time.perf_counter()
            try:
                result = pipeline.search(rec["question"], k, corpus="auto")
            except Exception as exc:
                result = {"chunks": [], "evidence": [], "summary": "",
                          "route": f"error:{type(exc).__name__}"}
            ms = (time.perf_counter() - t0) * 1000
            scores = judges.judge_record(rec, result)
            rows.append({"id": rec["id"], "category": rec["category"],
                         "route": result.get("route"), "latency_ms": ms,
                         **scores})
    finally:
        if not with_generation:
            generator.OLLAMA_AVAILABLE = original

    keys = ["route_correct", "item_recall", "item_mrr", "video_recall",
            "video_mrr", "fact_in_evidence", "fact_in_answer",
            "combined_coverage", "empty_result", "rewritten"]
    agg = metrics.aggregate(rows, keys)
    agg.update(metrics.latency_summary([r["latency_ms"] for r in rows]))
    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        by_cat[cat] = metrics.aggregate(cat_rows, keys)
    return {"aggregate": agg, "by_category": by_cat, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 evaluation harness")
    parser.add_argument("--mode", default="all",
                        choices=["routing", "gamma-ablation",
                                 "transcript-ablation", "full", "all"])
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    records = load_dataset()
    report: dict = {
        "dataset": str(DATASET),
        "record_count": len(records),
        "by_category": {c: sum(1 for r in records if r["category"] == c)
                        for c in sorted({r["category"] for r in records})},
        "run_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "k": args.k,
    }

    if args.mode in ("routing", "all"):
        print("== routing ==", flush=True)
        report["routing"] = eval_routing(records)
        print(f"   accuracy: {report['routing']['overall']['route_correct']} "
              f"({len(report['routing']['mistakes'])} mistakes)")
        for m in report["routing"]["mistakes"]:
            print(f"   MISS {m['id']}: expected {m['expected']} got {m['got']} — {m['reason']}")
    if args.mode in ("gamma-ablation", "all"):
        print("== gamma ablation ==", flush=True)
        report["gamma_ablation"] = eval_gamma_ablation(records, args.k)
        for name, res in report["gamma_ablation"].items():
            a = res["aggregate"]
            print(f"   {name:14s} recall={a['item_recall']} mrr={a['item_mrr']} "
                  f"facts={a['fact_in_evidence']} p50={a.get('p50_ms')}ms")
    if args.mode in ("transcript-ablation", "all"):
        print("== transcript ablation ==", flush=True)
        report["transcript_ablation"] = eval_transcript_ablation(records, args.k)
        for name, res in report["transcript_ablation"].items():
            a = res["aggregate"]
            print(f"   {name:16s} video_recall={a['video_recall']} "
                  f"mrr={a['video_mrr']} p50={a.get('p50_ms')}ms")
    if args.mode in ("full", "all"):
        print("== full adaptive pipeline ==", flush=True)
        report["full"] = eval_full(records, args.k, args.with_generation)
        a = report["full"]["aggregate"]
        print(f"   route_acc={a['route_correct']} item_recall={a['item_recall']} "
              f"video_recall={a['video_recall']} facts={a['fact_in_evidence']} "
              f"combined={a['combined_coverage']} rewrite_rate={a['rewritten']} "
              f"p50={a.get('p50_ms')}ms")

    out_path = args.out or (RESULTS_DIR /
                            f"eval_{_dt.datetime.now():%Y%m%d_%H%M%S}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nresults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
