"""Reranker tests: lazy loading, fail-open behavior, bounded candidates.
No real cross-encoder weights are ever loaded."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rag import reranker
from rag.evidence import Evidence


def ev(id_, text):
    return Evidence(id=id_, corpus="gamma", source_type="gamma_facet",
                    title=id_, text=text)


class _FakeCE:
    def __init__(self, scores):
        self._scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return self._scores[:len(pairs)]


class TestReranker(unittest.TestCase):
    def setUp(self):
        reranker._model = None
        reranker._load_failed = False
        self.addCleanup(lambda: setattr(reranker, "_model", None))
        self.addCleanup(lambda: setattr(reranker, "_load_failed", False))

    def test_disabled_returns_input_unchanged(self):
        evs = [ev("a", "x"), ev("b", "y")]
        with patch.object(reranker, "RERANKER_ENABLED", False):
            out = reranker.rerank("q", evs)
        self.assertEqual(out, evs)

    def test_reorders_by_score(self):
        evs = [ev("a", "x"), ev("b", "y"), ev("c", "z")]
        fake = _FakeCE([0.1, 0.9, 0.5])
        with patch.object(reranker, "_get_model", return_value=fake):
            out = reranker.rerank("q", evs)
        self.assertEqual([e.id for e in out], ["b", "c", "a"])
        self.assertEqual(out[0].metadata["rerank_pos"], 0)
        self.assertIsNotNone(reranker.last_latency_ms)

    def test_model_load_failure_falls_back(self):
        with patch.object(reranker, "_get_model", return_value=None):
            evs = [ev("a", "x")]
            out = reranker.rerank("q", evs)
        self.assertEqual(out, evs)

    def test_predict_failure_falls_back(self):
        class Boom:
            def predict(self, pairs):
                raise RuntimeError("boom")
        evs = [ev("a", "x"), ev("b", "y")]
        with patch.object(reranker, "_get_model", return_value=Boom()):
            out = reranker.rerank("q", evs)
        self.assertEqual(out, evs)

    def test_candidate_set_is_bounded(self):
        evs = [ev(f"e{i:03d}", "t") for i in range(50)]
        fake = _FakeCE([1.0] * 50)
        with patch.object(reranker, "_get_model", return_value=fake), \
             patch.object(reranker, "RERANK_CANDIDATES", 10):
            out = reranker.rerank("q", evs)
        self.assertEqual(len(fake.calls[0]), 10)     # only the bounded head
        self.assertEqual(len(out), 50)               # tail preserved

    def test_lazy_no_model_instantiated_on_import(self):
        self.assertIsNone(reranker._model)

    def test_deterministic_tiebreak(self):
        evs = [ev("b", "x"), ev("a", "y")]
        fake = _FakeCE([0.5, 0.5])
        with patch.object(reranker, "_get_model", return_value=fake):
            out = reranker.rerank("q", evs)
        self.assertEqual([e.id for e in out], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
