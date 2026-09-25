"""Unit tests for rag.fusion — RRF math and FTS query sanitization."""

from __future__ import annotations

import unittest

from rag.fusion import build_match_query, rrf_fuse


class TestRRF(unittest.TestCase):
    def test_standard_form(self):
        fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], rrf_k=60))
        self.assertAlmostEqual(fused["a"], 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(fused["c"], 1 / 62)

    def test_agreement_beats_single_list_rank(self):
        fused = rrf_fuse([["a", "b"], ["b", "c"]])
        self.assertEqual(fused[0][0], "b")   # appears in both lists

    def test_duplicates_within_a_list_ignored(self):
        fused = dict(rrf_fuse([["a", "a", "b"]], rrf_k=60))
        self.assertAlmostEqual(fused["a"], 1 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 62)

    def test_weights(self):
        fused = dict(rrf_fuse([["a"], ["b"]], weights=[2.0, 1.0], rrf_k=60))
        self.assertAlmostEqual(fused["a"], 2 / 61)
        self.assertAlmostEqual(fused["b"], 1 / 61)

    def test_weight_length_mismatch(self):
        with self.assertRaises(ValueError):
            rrf_fuse([["a"]], weights=[1.0, 2.0])

    def test_deterministic_tiebreak_on_id(self):
        fused = rrf_fuse([["b"], ["a"]])
        self.assertEqual([f[0] for f in fused], ["a", "b"])  # equal scores

    def test_empty(self):
        self.assertEqual(rrf_fuse([]), [])
        self.assertEqual(rrf_fuse([[], []]), [])


class TestMatchQuery(unittest.TestCase):
    def test_tokens_quoted_and_ored(self):
        self.assertEqual(build_match_query("best weapon"),
                         '"best" OR "weapon"')

    def test_hyphens_and_specials_sanitized(self):
        q = build_match_query('AK-74 "capacity" (magazine)')
        self.assertEqual(q, '"ak" OR "74" OR "capacity" OR "magazine"')

    def test_calibers_keep_dots(self):
        self.assertIn('"5.45"', build_match_query("5.45x39 ammo"))

    def test_dedupe_and_cap(self):
        q = build_match_query("a a a bb bb cc", max_tokens=2)
        self.assertEqual(q, '"bb" OR "cc"')

    def test_single_letters_dropped_digits_kept(self):
        self.assertEqual(build_match_query("a 9 mm"), '"9" OR "mm"')

    def test_empty_input(self):
        self.assertEqual(build_match_query(""), "")
        self.assertEqual(build_match_query("!!!"), "")


if __name__ == "__main__":
    unittest.main()
