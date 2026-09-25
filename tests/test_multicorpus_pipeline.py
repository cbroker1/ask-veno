"""Multi-corpus pipeline tests: corpus normalization, combined assembly,
GAMMA degradation paths, and generation dispatch. All boundaries mocked."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from rag import pipeline
from rag.evidence import Evidence
from rag.router import RouteDecision
from storage.gamma import GammaUnavailable


def chunk(id_, sim=70.0):
    return {"id": id_, "distance": 0.3, "sim_pct": sim, "vt": "T", "vid": "v",
            "s": "0:01", "e": "0:02", "url": "u", "txt": "x"}


def gamma_ev(id_="gamma095:wpn_x:overview", exact=False):
    return Evidence(id=id_, corpus="gamma",
                    source_type="gamma_exact" if exact else "gamma_facet",
                    title="X", text="X (wpn_x) — Rifles — GAMMA 0.9.5",
                    metadata={"item_id": "wpn_x", "facet_type": "overview"})


class TestCorpusNormalization(unittest.TestCase):
    def test_aliases(self):
        for raw, want in [("auto", "auto"), ("", "auto"), (None, "auto"),
                          ("VENO", "transcripts"), ("venoxium", "transcripts"),
                          ("transcripts", "transcripts"), ("Gamma", "gamma"),
                          ("combined", "combined"), ("both", "combined")]:
            self.assertEqual(pipeline.normalize_corpus(raw), want, raw)

    def test_unknown_corpus_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            pipeline.normalize_corpus("nonsense")
        self.assertEqual(ctx.exception.status_code, 422)


class TestCombined(unittest.TestCase):
    def setUp(self):
        p = patch.object(pipeline, "CORRECTIVE_RAG_ENABLED", False)
        p.start(); self.addCleanup(p.stop)
        p2 = patch.object(pipeline.generator, "OLLAMA_AVAILABLE", False)
        p2.start(); self.addCleanup(p2.stop)
        p3 = patch.object(pipeline.reranker, "RERANKER_ENABLED", False)
        p3.start(); self.addCleanup(p3.stop)

    def test_combined_keeps_both_corpora_and_labels(self):
        g = {"evidence": [gamma_ev(exact=True)],
             "exact_items": [{"item_id": "wpn_x", "name": "X"}],
             "debug": {}}
        chunks = [chunk("c1"), chunk("c2")]
        with patch.object(pipeline, "_gamma_retrieve", return_value=g), \
             patch.object(pipeline, "retrieve_chunks_hybrid",
                          return_value=chunks):
            out = pipeline.search("q", 5, corpus="combined")
        corpora = {e["corpus"] for e in out["evidence"]}
        self.assertEqual(corpora, {"gamma", "transcripts"})
        self.assertEqual(out["route"], "combined")
        self.assertEqual([c["id"] for c in out["chunks"]], ["c1", "c2"])
        self.assertEqual(out["retrieval_debug"]["corpus_counts"],
                         {"gamma": 1, "transcripts": 2})

    def test_combined_gamma_failure_keeps_transcript_side(self):
        with patch.object(pipeline, "_gamma_retrieve",
                          side_effect=GammaUnavailable("no db")), \
             patch.object(pipeline, "retrieve_chunks_hybrid",
                          return_value=[chunk("c1")]):
            out = pipeline.search("q", 5, corpus="combined")
        self.assertEqual(out["route"], "combined")
        self.assertEqual(len(out["chunks"]), 1)
        self.assertEqual(out["retrieval_debug"]["corpus_counts"]["gamma"], 0)

    def test_manual_gamma_unavailable_raises_503(self):
        with patch.object(pipeline, "_retrieve_for_route",
                          side_effect=GammaUnavailable("no db")):
            with self.assertRaises(HTTPException) as ctx:
                pipeline.search("q", 5, corpus="gamma")
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("GAMMA 0.9.5 corpus unavailable", ctx.exception.detail)

    def test_auto_route_gamma_falls_back_to_transcripts(self):
        with patch.object(pipeline.router, "route",
                          return_value=RouteDecision("gamma", "kw")), \
             patch.object(pipeline, "_retrieve_for_route",
                          side_effect=GammaUnavailable("no db")), \
             patch.object(pipeline, "retrieve_chunks",
                          return_value=[chunk("c1")]):
            out = pipeline.search("ammo stats", 5, corpus="auto")
        self.assertEqual(out["route"], "transcripts")
        self.assertIn("fell back", out["route_reason"])
        self.assertEqual(len(out["chunks"]), 1)

    def test_gamma_route_uses_multicorpus_generator(self):
        g = {"chunks": [], "evidence": [gamma_ev()], "exact_items": [],
             "debug": {}}
        with patch.object(pipeline.router, "route",
                          return_value=RouteDecision("gamma", "kw")), \
             patch.object(pipeline, "_retrieve_for_route", return_value=g), \
             patch.object(pipeline.generator, "OLLAMA_AVAILABLE", True), \
             patch.object(pipeline.generator, "llm_answer_multicorpus",
                          Mock(return_value="DB ANSWER")) as multi, \
             patch.object(pipeline.generator, "llm_answer", Mock()) as legacy:
            out = pipeline.search("ammo stats", 5, corpus="auto")
        multi.assert_called_once()
        legacy.assert_not_called()
        self.assertEqual(out["summary"], "DB ANSWER")
        self.assertEqual(out["chunks"], [])

    def test_evidence_serialized_as_dicts(self):
        g = {"chunks": [], "evidence": [gamma_ev()], "exact_items": [],
             "debug": {}}
        with patch.object(pipeline.router, "route",
                          return_value=RouteDecision("gamma", "kw")), \
             patch.object(pipeline, "_retrieve_for_route", return_value=g):
            out = pipeline.search("q", 5, corpus="auto")
        self.assertIsInstance(out["evidence"][0], dict)
        self.assertEqual(out["evidence"][0]["corpus"], "gamma")


class TestGenerationLabeling(unittest.TestCase):
    def test_multicorpus_prompt_labels_sources(self):
        from rag import generator
        captured = {}

        def fake_generate(prompt, system):
            captured["prompt"] = prompt
            captured["system"] = system
            return "ok"

        t = Evidence(id="c1", corpus="transcripts",
                     source_type="youtube_transcript", title="Stream 5",
                     text="veno said things",
                     metadata={"video_id": "v", "start_hms": "0:01",
                               "end_hms": "0:02"})
        with patch.object(generator, "_ollama_generate",
                          side_effect=fake_generate):
            out = generator.llm_answer_multicorpus("q", [gamma_ev(), t])
        self.assertEqual(out, "ok")
        self.assertIn("[GAMMA 0.9.5 DATABASE]", captured["prompt"])
        self.assertIn("[VENOXIUM TRANSCRIPT]", captured["prompt"])
        self.assertIn("Never attribute database facts to Venoxium",
                      captured["system"])


if __name__ == "__main__":
    unittest.main()
