"""Unit tests for indexing.gamma.normalize — pure functions."""

from __future__ import annotations

import unittest

from indexing.gamma.normalize import (
    Translator,
    ammo_token_to_id,
    compact_name,
    content_hash,
    norm_name,
    parse_amount,
    parse_stat,
    resolve_display_variant,
)


class TestNames(unittest.TestCase):
    def test_norm_folds_case_and_whitespace(self):
        self.assertEqual(norm_name("  AK-74N   Rifle "), "ak-74n rifle")

    def test_norm_strips_wrapping_quotes(self):
        self.assertEqual(norm_name('"tch_upgr"'), "tch_upgr")

    def test_compact_removes_separators_but_keeps_dots(self):
        self.assertEqual(compact_name("AK-74N"), "ak74n")
        self.assertEqual(compact_name("5.45x39mm Test FMJ"), "5.45x39mmtestfmj")
        self.assertEqual(compact_name("wpn_ak74"), "wpnak74")

    def test_caliber_dots_never_destroyed(self):
        self.assertIn("5.45", compact_name("5.45x39"))


class TestParsing(unittest.TestCase):
    def test_plain_number(self):
        self.assertEqual(parse_stat("3.10"), ("3.10", 3.10, False))

    def test_percent(self):
        self.assertEqual(parse_stat("82%"), ("82%", 82.0, True))

    def test_non_numeric_keeps_text_only(self):
        self.assertEqual(parse_stat("D"), ("D", None, False))

    def test_native_int(self):
        self.assertEqual(parse_stat(20), ("20", 20.0, False))

    def test_bool(self):
        text, num, pct = parse_stat(True)
        self.assertEqual((text, num, pct), ("True", 1.0, False))

    def test_amounts(self):
        self.assertEqual(parse_amount("x3"), ("x3", 3.0))
        self.assertEqual(parse_amount("weird"), ("weird", None))

    def test_content_hash_deterministic(self):
        a = content_hash({"b": 1, "a": [1, 2]})
        b = content_hash({"a": [1, 2], "b": 1})
        self.assertEqual(a, b)
        self.assertNotEqual(a, content_hash({"a": [2, 1], "b": 1}))


class TestTranslator(unittest.TestCase):
    def setUp(self):
        self.tr = Translator({"st_x": "The X"}, {"st_x": "Икс"})

    def test_resolve_hits(self):
        self.assertEqual(self.tr.resolve("st_x"), "The X")

    def test_resolve_falls_back_to_key_and_records_miss(self):
        self.assertEqual(self.tr.resolve("st_missing"), "st_missing")
        self.assertIn("st_missing", self.tr.missing_keys)

    def test_resolve_or_none_returns_none_on_miss(self):
        self.assertIsNone(self.tr.resolve_or_none("st_missing"))

    def test_ru(self):
        self.assertEqual(self.tr.resolve_ru("st_x"), "Икс")

    def test_display_variant_suffix_preserved(self):
        self.assertEqual(resolve_display_variant("st_x [74]", self.tr), "The X [74]")
        self.assertEqual(resolve_display_variant("st_x", self.tr), "The X")
        self.assertIsNone(resolve_display_variant(None, self.tr))


class TestAmmoTokens(unittest.TestCase):
    AMMO_IDS = {"ammo_5.45x39_ap", "ammo_9x39_pab9"}
    PDA = {"ammo-5.45x39-ap": "ammo_5.45x39_ap"}

    def test_pda_name_resolution(self):
        self.assertEqual(
            ammo_token_to_id("ammo-5.45x39-ap", self.AMMO_IDS, self.PDA),
            "ammo_5.45x39_ap")

    def test_dash_to_underscore_fallback(self):
        self.assertEqual(
            ammo_token_to_id("ammo-9x39-pab9", self.AMMO_IDS, {}),
            "ammo_9x39_pab9")

    def test_unresolvable_returns_none(self):
        self.assertIsNone(ammo_token_to_id("ammo-nope", self.AMMO_IDS, self.PDA))
        self.assertIsNone(ammo_token_to_id("", self.AMMO_IDS, self.PDA))


if __name__ == "__main__":
    unittest.main()
