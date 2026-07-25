"""GAMMA retrieval-layer tests against a fixture-built temp database.

No Chroma collection exists in the fixture DB, so dense search degrades and
the exact + BM25 layers carry the retrieval — which is itself part of the
graceful-degradation contract under test.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from indexing.gamma import builder as builder_mod
from indexing.gamma.loader import load_dataset
from rag import gamma_retriever
from storage.gamma import GammaDB, GammaUnavailable
from tests.gamma_fixtures import write_fixture


class TestGammaRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        ds = load_dataset(write_fixture(root))
        res = builder_mod.build(ds)
        cls.db_path = root / "gamma_test.sqlite"
        # active_collection None -> dense layer must degrade gracefully
        builder_mod.write_database(cls.db_path, ds, res, "fixture", None)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_exact_item_resolution_by_display_name(self):
        db = GammaDB(self.db_path)
        try:
            items = gamma_retriever.resolve_exact_items(db, "TAK-74 magazine capacity")
        finally:
            db.close()
        self.assertTrue(items)
        self.assertEqual(items[0]["item_id"], "wpn_tak")

    def test_exact_item_resolution_by_internal_id(self):
        db = GammaDB(self.db_path)
        try:
            items = gamma_retriever.resolve_exact_items(db, "stats for wpn_tak please")
        finally:
            db.close()
        self.assertEqual(items[0]["item_id"], "wpn_tak")

    def test_exact_item_resolution_by_russian_alias(self):
        db = GammaDB(self.db_path)
        try:
            items = gamma_retriever.resolve_exact_items(db, "ТАК-74")
        finally:
            db.close()
        self.assertEqual(items[0]["item_id"], "wpn_tak")

    def test_prefix_fallback_for_truncated_name(self):
        db = GammaDB(self.db_path)
        try:
            items = gamma_retriever.resolve_exact_items(db, "tak-7 rifle")
        finally:
            db.close()
        self.assertTrue(any(i["item_id"] == "wpn_tak" for i in items))

    def test_retrieve_returns_protected_exact_facets_first(self):
        out = gamma_retriever.retrieve("TAK-74 magazine capacity",
                                       top_k=4, db_path=self.db_path)
        self.assertEqual(out["exact_items"][0]["item_id"], "wpn_tak")
        self.assertTrue(out["evidence"])
        first = out["evidence"][0]
        self.assertEqual(first.source_type, "gamma_exact")
        self.assertEqual(first.metadata["item_id"], "wpn_tak")
        self.assertEqual(first.corpus, "gamma")
        # dense degraded, bm25 active
        self.assertIn("dense_skipped", out["debug"])
        self.assertGreaterEqual(out["debug"]["bm25_candidates"], 1)

    def test_retrieve_without_exact_match_uses_bm25(self):
        out = gamma_retriever.retrieve("standard magazine crafting barrel",
                                       top_k=4, db_path=self.db_path)
        self.assertTrue(out["evidence"])
        self.assertTrue(all(e.corpus == "gamma" for e in out["evidence"]))

    def test_missing_database_raises_unavailable(self):
        with self.assertRaises(GammaUnavailable):
            gamma_retriever.retrieve("anything", db_path="/nonexistent/db.sqlite")

    def test_rerank_hook_applies_to_non_exact_only(self):
        def reverse(query, candidates):
            return list(reversed(candidates))
        # top_k large enough that truncation never hides the reordering
        base = gamma_retriever.retrieve("standard magazine crafting barrel",
                                        top_k=50, db_path=self.db_path)
        rev = gamma_retriever.retrieve("standard magazine crafting barrel",
                                       top_k=50, db_path=self.db_path,
                                       rerank_fn=reverse)
        base_ids = [e.id for e in base["evidence"] if e.source_type != "gamma_exact"]
        rev_ids = [e.id for e in rev["evidence"] if e.source_type != "gamma_exact"]
        if len(base_ids) > 1:
            self.assertEqual(rev_ids, base_ids[::-1])
        base_exact = [e.id for e in base["evidence"] if e.source_type == "gamma_exact"]
        rev_exact = [e.id for e in rev["evidence"] if e.source_type == "gamma_exact"]
        self.assertEqual(base_exact, rev_exact)   # protected set untouched

    def test_evidence_serializes(self):
        out = gamma_retriever.retrieve("TAK-74", top_k=3, db_path=self.db_path)
        d = out["evidence"][0].to_dict()
        self.assertEqual(d["corpus"], "gamma")
        self.assertIn("item_id", d["metadata"])


if __name__ == "__main__":
    unittest.main()
