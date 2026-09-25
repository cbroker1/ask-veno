"""Corrective-RAG tests: at most ONE rewrite + ONE re-retrieval, rewrite
failures keep the original query, and the retry is kept only when it is
strictly stronger. All retrieval and Ollama boundaries are mocked."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rag import pipeline
from rag.router import RouteDecision


def chunk(id_, sim):
    return {"id": id_, "distance": 0.3, "sim_pct": sim, "vt": "T", "vid": "v",
            "s": "0:01", "e": "0:02", "url": "u", "txt": "x"}


def _transcript_result(chunks):
    from rag import evidence as ev_mod
    return {"chunks": chunks, "evidence": [ev_mod.from_chunk(c) for c in chunks],
            "exact_items": [], "debug": {}}


class TestCorrective(unittest.TestCase):
    def setUp(self):
        p1 = patch.object(pipeline.router, "route",
                          return_value=RouteDecision("transcripts", "test"))
        p1.start(); self.addCleanup(p1.stop)
        p2 = patch.object(pipeline.generator, "OLLAMA_AVAILABLE", False)
        p2.start(); self.addCleanup(p2.stop)
        p3 = patch.object(pipeline, "CORRECTIVE_RAG_ENABLED", True)
        p3.start(); self.addCleanup(p3.stop)

    def test_strong_evidence_never_rewrites(self):
        strong = _transcript_result([chunk("c1", 80.0)])
        with patch.object(pipeline, "_retrieve_for_route",
                          return_value=strong) as ret, \
             patch("rag.query_rewriter.rewrite_query") as rw:
            out = pipeline.search("q")
        self.assertEqual(ret.call_count, 1)
        rw.assert_not_called()
        self.assertFalse(out["rewritten"])

    def test_weak_evidence_retries_exactly_once_and_keeps_better(self):
        weak = _transcript_result([chunk("c1", 30.0)])
        better = _transcript_result([chunk("c2", 90.0)])
        with patch.object(pipeline, "_retrieve_for_route",
                          side_effect=[weak, better]) as ret, \
             patch("rag.query_rewriter.rewrite_query",
                   return_value="rewritten q") as rw:
            out = pipeline.search("q")
        self.assertEqual(ret.call_count, 2)          # initial + single retry
        rw.assert_called_once_with("q")
        self.assertTrue(out["rewritten"])
        self.assertEqual(out["query_used"], "rewritten q")
        self.assertEqual(out["chunks"][0]["id"], "c2")

    def test_retry_discarded_when_not_stronger(self):
        weak = _transcript_result([chunk("c1", 30.0)])
        worse = _transcript_result([])
        with patch.object(pipeline, "_retrieve_for_route",
                          side_effect=[weak, worse]) as ret, \
             patch("rag.query_rewriter.rewrite_query", return_value="rw q"):
            out = pipeline.search("q")
        self.assertEqual(ret.call_count, 2)
        self.assertFalse(out["rewritten"])
        self.assertEqual(out["query_used"], "q")
        self.assertEqual(out["chunks"][0]["id"], "c1")

    def test_rewrite_failure_keeps_original_single_pass(self):
        weak = _transcript_result([])
        with patch.object(pipeline, "_retrieve_for_route",
                          return_value=weak) as ret, \
             patch("rag.query_rewriter.rewrite_query", return_value=None) as rw:
            out = pipeline.search("q")
        self.assertEqual(ret.call_count, 1)          # no second retrieval
        rw.assert_called_once()
        self.assertFalse(out["rewritten"])

    def test_retry_retrieval_error_keeps_original(self):
        weak = _transcript_result([chunk("c1", 20.0)])
        with patch.object(pipeline, "_retrieve_for_route",
                          side_effect=[weak, RuntimeError("boom")]) as ret, \
             patch("rag.query_rewriter.rewrite_query", return_value="rw"):
            out = pipeline.search("q")
        self.assertEqual(ret.call_count, 2)
        self.assertEqual(out["chunks"][0]["id"], "c1")
        self.assertFalse(out["rewritten"])

    def test_corrective_disabled_never_rewrites(self):
        weak = _transcript_result([])
        with patch.object(pipeline, "CORRECTIVE_RAG_ENABLED", False), \
             patch.object(pipeline, "_retrieve_for_route",
                          return_value=weak) as ret, \
             patch("rag.query_rewriter.rewrite_query") as rw:
            pipeline.search("q")
        self.assertEqual(ret.call_count, 1)
        rw.assert_not_called()

    def test_manual_transcripts_mode_never_touches_corrective(self):
        with patch("rag.pipeline.retrieve_chunks", return_value=[]), \
             patch("rag.query_rewriter.rewrite_query") as rw:
            out = pipeline.search("q", 5, corpus="transcripts")
        rw.assert_not_called()
        self.assertEqual(out["chunks"], [])


class TestWeakness(unittest.TestCase):
    def test_transcript_thresholds(self):
        weak = _transcript_result([chunk("c", 30.0)])
        strong = _transcript_result([chunk("c", 70.0)])
        self.assertIsNotNone(pipeline._is_weak("transcripts", weak, "q"))
        self.assertIsNone(pipeline._is_weak("transcripts", strong, "q"))
        self.assertEqual(pipeline._is_weak("transcripts",
                                           _transcript_result([]), "q"),
                         "no transcript evidence")

    def test_gamma_model_term_without_exact_match_is_weak(self):
        from rag.evidence import Evidence
        ev = Evidence(id="f", corpus="gamma", source_type="gamma_facet",
                      title="t", text="x")
        result = {"chunks": [], "evidence": [ev], "exact_items": [], "debug": {}}
        self.assertIsNotNone(pipeline._is_weak("gamma", result, "ak-74 stats"))
        result_exact = dict(result, exact_items=[{"item_id": "wpn_ak74"}])
        self.assertIsNone(pipeline._is_weak("gamma", result_exact, "ak-74 stats"))

    def test_combined_missing_side_is_weak(self):
        from rag.evidence import Evidence
        g = Evidence(id="f", corpus="gamma", source_type="gamma_facet",
                     title="t", text="x")
        t = Evidence(id="c", corpus="transcripts",
                     source_type="youtube_transcript", title="t", text="x")
        both = {"chunks": [], "evidence": [g, t], "exact_items": [], "debug": {}}
        gamma_only = {"chunks": [], "evidence": [g], "exact_items": [], "debug": {}}
        self.assertIsNone(pipeline._is_weak("combined", both, "q"))
        self.assertEqual(pipeline._is_weak("combined", gamma_only, "q"),
                         "combined query but no transcript evidence")


if __name__ == "__main__":
    unittest.main()
