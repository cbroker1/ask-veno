"""Routing decision-table tests. Exact-item resolution is mocked so the
tests never touch a real GAMMA database."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rag import router

AK = {"item_id": "wpn_ak74", "name": "AK-74N", "category": "Rifles"}


def route_with(query, exact):
    with patch.object(router, "_exact_items", return_value=exact):
        return router.route(query)


class TestSpecExamples(unittest.TestCase):
    """The exact examples from the Phase 2 corpus-mode specification."""

    def test_magazine_capacity_goes_gamma(self):
        d = route_with("What is the magazine capacity of the AK-74?", [AK])
        self.assertEqual(d.corpus, "gamma")

    def test_veno_opinion_goes_transcripts(self):
        d = route_with("What does Veno think of the AK-74?", [AK])
        self.assertEqual(d.corpus, "transcripts")

    def test_stats_plus_veno_goes_combined(self):
        d = route_with("What are the AK-74 stats and how does Veno use it?", [AK])
        self.assertEqual(d.corpus, "combined")

    def test_radar_story_goes_transcripts(self):
        d = route_with("What happened when Veno entered Radar?", [])
        self.assertEqual(d.corpus, "transcripts")

    def test_disassemble_goes_gamma(self):
        d = route_with("What does this rifle disassemble into?", [])
        self.assertEqual(d.corpus, "gamma")

    def test_early_rifle_plus_veno_goes_combined(self):
        d = route_with("What rifle should I use early game and what does Veno prefer?", [AK])
        self.assertEqual(d.corpus, "combined")


class TestRules(unittest.TestCase):
    def test_exact_item_alone_goes_gamma(self):
        d = route_with("AK-74", [AK])
        self.assertEqual(d.corpus, "gamma")
        self.assertIn("AK-74N", d.reason)

    def test_mechanic_comparison_goes_gamma(self):
        # mechanic/stat comparisons are directly answerable from the database
        d = route_with("compare silencer attachments by weight", [])
        self.assertEqual(d.corpus, "gamma")

    def test_category_recommendation_goes_combined(self):
        d = route_with("best weapon for killing mutants", [])
        self.assertEqual(d.corpus, "combined")

    def test_possessive_veno_recognized(self):
        d = route_with("What was Veno's plan to buy ammo?", [])
        self.assertEqual(d.corpus, "combined")   # veno + mechanic keywords

    def test_pure_mechanic_keywords_go_gamma(self):
        d = route_with("silencer attachment weight", [])
        self.assertEqual(d.corpus, "gamma")

    def test_no_signals_default_transcripts(self):
        d = route_with("how to make money early game", [])
        self.assertEqual(d.corpus, "transcripts")
        self.assertIn("default", d.reason)

    def test_recommendation_plus_item_goes_combined(self):
        d = route_with("is the AK-74 worth using", [AK])
        self.assertEqual(d.corpus, "combined")

    def test_veno_signal_never_consults_gamma_db(self):
        with patch.object(router, "_exact_items",
                          side_effect=AssertionError("must not resolve")) as m:
            d = router.route("what did Veno say yesterday")
        self.assertEqual(d.corpus, "transcripts")
        m.assert_not_called()

    def test_decision_fields_populated(self):
        d = route_with("What are the AK-74 stats and how does Veno use it?", [AK])
        self.assertIn("stats", d.gamma_terms)
        self.assertTrue(d.veno_terms)
        self.assertEqual(d.exact_items[0]["item_id"], "wpn_ak74")

    def test_gamma_db_unavailable_degrades_to_keywords(self):
        # unmocked _exact_items with a nonexistent db path -> []
        d_gamma = router.route("magazine capacity of the akm",
                               db_path="/nonexistent/gamma.sqlite")
        self.assertEqual(d_gamma.corpus, "gamma")   # keywords still route


if __name__ == "__main__":
    unittest.main()
