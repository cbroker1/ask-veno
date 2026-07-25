"""Canonicalization tests: loader validation, builder joins, aliases,
relations, facets, audit gates, and SQLite round-trip via GammaDB."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from indexing.gamma import builder as builder_mod
from indexing.gamma.loader import GammaLoadError, load_dataset
from storage.gamma import GammaDB
from tests.gamma_fixtures import write_fixture


def load_fixture(tmp: Path, **kwargs):
    return load_dataset(write_fixture(tmp, **kwargs))


class TestLoader(unittest.TestCase):
    def test_loads_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = load_fixture(Path(tmp))
        self.assertEqual(len(ds.index), 6)
        self.assertIn("wpn_tak", ds.index_by_id)
        self.assertEqual(ds.translations_en["st_wpn_tak"], "TAK-74")

    def test_missing_index_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GammaLoadError):
                load_fixture(Path(tmp), drop={"index.json"})

    def test_duplicate_ids_fail(self):
        def mutate(files):
            files["index.json"].append(dict(files["index.json"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GammaLoadError) as ctx:
                load_fixture(Path(tmp), mutate=mutate)
            self.assertIn("Duplicate", str(ctx.exception))

    def test_malformed_manifest_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_fixture(Path(tmp))
            data_dir = root / "site/public/data/gamma-0.9.5"
            (data_dir / "manifest.json").write_text('["not", "a", "map"]')
            with self.assertRaises(GammaLoadError):
                load_dataset(root)

    def test_version_pin_rejects_other_dirs(self):
        # load_dataset derives the directory from the pinned pack id, so a
        # tree containing only e.g. gamma-0.9.6 must fail (dir missing).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "site/public/data/gamma-0.9.6").mkdir(parents=True)
            with self.assertRaises(GammaLoadError):
                load_dataset(root)


class TestBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.ds = load_fixture(Path(cls._tmp.name))
        cls.res = builder_mod.build(cls.ds)
        cls.items = {i.item_id: i for i in cls.res.items}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_index_item_canonicalized(self):
        self.assertEqual(len(self.res.items), 6)
        self.assertEqual(set(self.items), set(self.ds.index_by_id))

    def test_names_descriptions_prices_resolved(self):
        tak = self.items["wpn_tak"]
        self.assertEqual(tak.name, "TAK-74")
        # upstream literal backslash-n markup becomes a real newline
        self.assertEqual(tak.description,
                         "A dependable test rifle.\nTrusted in the Zone.")
        self.assertNotIn("\\n", tak.description)
        self.assertEqual(tak.price, 26950)
        self.assertEqual(tak.display_variant, "TAK-74 [74]")
        self.assertEqual(tak.category, "Rifles")

    def test_raw_json_losslessly_preserved(self):
        tak = self.items["wpn_tak"]
        self.assertEqual(tak.raw["index"]["id"], "wpn_tak")
        self.assertEqual(tak.raw["tables"]["rifles.json"]["ui_inv_accuracy"], "82%")
        self.assertIn("weapon-addon-status.json", tak.raw)

    def test_aliases_cover_expected_sources(self):
        tak_aliases = {(a[4], a[1]) for a in self.res.aliases if a[0] == "wpn_tak"}
        self.assertIn(("display_name", "TAK-74"), tak_aliases)
        self.assertIn(("item_id", "wpn_tak"), tak_aliases)
        self.assertIn(("name_key", "st_wpn_tak"), tak_aliases)
        self.assertIn(("display_variant", "TAK-74 [74]"), tak_aliases)
        self.assertIn(("name_ru", "ТАК-74"), tak_aliases)

    def test_numeric_price_columns_never_become_aliases(self):
        sources = {a[4] for a in self.res.aliases}
        self.assertNotIn("alt_name", sources)   # fixture has no legit alt names

    def test_properties_parse_percent_and_numbers(self):
        props = {(p[1]): p for p in self.res.properties if p[0] == "wpn_tak"}
        label, text, num, is_pct = (props["ui_inv_accuracy"][2],
                                    props["ui_inv_accuracy"][3],
                                    props["ui_inv_accuracy"][4],
                                    props["ui_inv_accuracy"][5])
        self.assertEqual((label, text, num, is_pct), ("Accuracy", "82%", 82.0, 1))
        self.assertEqual(props["st_prop_weight"][4], 3.10)
        self.assertEqual(props["silencer_status"][3], "attachable")

    def test_caliber_property_from_calibers_file(self):
        cal = [p for p in self.res.properties
               if p[0] == "ammo_5.45x39_test" and p[1] == "caliber"]
        self.assertEqual(len(cal), 1)
        self.assertEqual(cal[0][3], "5.45x39")

    def test_uses_ammo_relation_resolves_dashed_tokens(self):
        rels = [r for r in self.res.relations
                if r[0] == "wpn_tak" and r[1] == "uses_ammo"]
        by_dst = {r[2]: json.loads(r[4]) for r in rels}
        self.assertEqual(by_dst["ammo_5.45x39_test"], {"alt": False})
        self.assertEqual(by_dst["ammo_5.45x39_alt"], {"alt": True})

    def test_disassemble_resolves_name_keys_and_keeps_unresolved_text(self):
        rels = [r for r in self.res.relations
                if r[0] == "wpn_tak" and r[1] == "disassembles_to"]
        resolved = [r for r in rels if r[2] == "prt_w_barrel_t"]
        unresolved = [r for r in rels if r[2] is None]
        self.assertEqual(len(resolved), 1)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0][3], "st_unknown_part")  # fallback text

    def test_unresolved_addon_kept_as_text(self):
        rels = [r for r in self.res.relations
                if r[0] == "wpn_tak" and r[1] == "compatible_addon"]
        self.assertEqual(len(rels), 1)
        self.assertIsNone(rels[0][2])
        self.assertEqual(rels[0][3], "sil_missing")

    def test_sold_by_uses_trader_label(self):
        rels = [r for r in self.res.relations
                if r[0] == "wpn_tak" and r[1] == "sold_by"]
        detail = json.loads(rels[0][4])
        self.assertEqual(detail["trader_label"], "Sidorovich")

    def test_facet_ids_deterministic_and_texts_labeled(self):
        by_id = {f.facet_id: f for f in self.res.facets}
        ov = by_id["gamma095:wpn_tak:overview"]
        self.assertIn("TAK-74 (wpn_tak) — Rifles — GAMMA 0.9.5", ov.text)
        self.assertIn("A dependable test rifle.", ov.text)
        stats = by_id["gamma095:wpn_tak:stats"]
        self.assertIn("Accuracy: 82%", stats.text)
        comp = by_id["gamma095:wpn_tak:compatibility"]
        self.assertIn("5.45x39mm Test FMJ (ammo_5.45x39_test)", comp.text)
        self.assertIn("TAK Standard Magazine", comp.text)
        obt = by_id["gamma095:wpn_tak:obtaining"]
        self.assertIn("Sidorovich", obt.text)
        self.assertIn("jupiter (3% Common)", obt.text)
        ups = by_id["gamma095:wpn_tak:upgrades"]
        self.assertIn("Lightened bolt", ups.text)
        crafting = by_id["gamma095:mag_tak_default:crafting"]
        self.assertIn("Test Barrel (prt_w_barrel_t) x2", crafting.text)
        # reverse crafting reference on the ingredient
        part_crafting = by_id["gamma095:prt_w_barrel_t:crafting"]
        self.assertIn("Used as crafting ingredient for", part_crafting.text)

    def test_build_is_deterministic(self):
        res2 = builder_mod.build(self.ds)
        self.assertEqual([(f.facet_id, f.text) for f in self.res.facets],
                         [(f.facet_id, f.text) for f in res2.facets])
        self.assertEqual(self.res.relations, res2.relations)
        self.assertEqual(self.res.aliases, res2.aliases)

    def test_audit_counts_and_gates(self):
        audit = self.res.audit
        self.assertEqual(audit["index_item_count"], 6)
        self.assertEqual(audit["canonical_item_count"], 6)
        self.assertEqual(audit["missing_index_items"], [])
        self.assertEqual(builder_mod.check_audit(audit), [])

    def test_audit_gate_fails_on_coverage_loss(self):
        bad = dict(self.res.audit)
        bad["canonical_item_count"] = 3
        bad["missing_index_items"] = ["a", "b", "c"]
        problems = builder_mod.check_audit(bad)
        self.assertTrue(any("coverage" in p for p in problems))
        self.assertTrue(any("canonical record" in p for p in problems))


class TestWriteDatabase(unittest.TestCase):
    def test_round_trip_via_gammadb(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = load_fixture(Path(tmp))
            res = builder_mod.build(ds)
            db_path = Path(tmp) / "gamma_test.sqlite"
            builder_mod.write_database(db_path, ds, res, "deadbeef", "col_x")

            db = GammaDB(db_path)
            try:
                meta = db.meta()
                self.assertEqual(meta["pack_id"], "gamma-0.9.5")
                self.assertEqual(meta["upstream_commit"], "deadbeef")
                self.assertEqual(db.active_collection(), "col_x")
                self.assertEqual(db.item_count(), 6)
                self.assertEqual(db.facet_count(), len(res.facets))

                item = db.item("wpn_tak")
                self.assertEqual(item["name"], "TAK-74")

                hits = db.lookup_exact([("tak-74", "tak74")])
                self.assertEqual(hits[0]["item_id"], "wpn_tak")
                prefix = db.lookup_prefix("tak7")
                self.assertTrue(any(h["item_id"] == "wpn_tak" for h in prefix))

                rels = db.relations("wpn_tak", "uses_ammo")
                self.assertEqual(len(rels), 2)
                rev = db.relations_to("ammo_5.45x39_test", "uses_ammo")
                self.assertEqual(rev[0]["src_item_id"], "wpn_tak")

                fts = db.fts_search("rifle OR accuracy")
                self.assertTrue(fts)
                self.assertTrue(all("bm25_score" in r for r in fts))
                self.assertEqual(db.fts_search('malformed "unclosed'), [])
            finally:
                db.close()

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = load_fixture(Path(tmp))
            res = builder_mod.build(ds)
            db_path = Path(tmp) / "gamma_test.sqlite"
            db_path.write_text("existing")
            with self.assertRaises(builder_mod.BuildError):
                builder_mod.write_database(db_path, ds, res, "c", None)


if __name__ == "__main__":
    unittest.main()
