"""Transcript lexical index tests: atomic build, BM25 search, degradation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag.fusion import build_match_query
from storage.lexical import LexicalDB, LexicalUnavailable, build_from_rows

ROWS = [
    {"chunk_id": "vidA_chunk_00001", "video_id": "vidA", "video_title": "Lab Dive",
     "start_hms": "0:10", "end_hms": "0:20", "url": "https://youtu.be/vidA?t=10",
     "text": "we are heading into the lab with the shotgun"},
    {"chunk_id": "vidB_chunk_00002", "video_id": "vidB", "video_title": "Swamp Run",
     "start_hms": "1:00", "end_hms": "1:30", "url": "https://youtu.be/vidB?t=60",
     "text": "the swamp is full of bloodsuckers tonight"},
]


class TestLexical(unittest.TestCase):
    def test_build_and_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "lex.sqlite"
            build_from_rows(db_path, ROWS, source="test")
            db = LexicalDB(db_path)
            try:
                self.assertEqual(db.count(), 2)
                self.assertEqual(db.meta()["source"], "test")
                hits = db.fts_search(build_match_query("shotgun lab"))
                self.assertEqual(hits[0]["chunk_id"], "vidA_chunk_00001")
                self.assertEqual(hits[0]["video_title"], "Lab Dive")
                self.assertLess(hits[0]["bm25_score"], 0)   # fts5 bm25 is negative
            finally:
                db.close()

    def test_rebuild_is_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "lex.sqlite"
            build_from_rows(db_path, ROWS, source="one")
            build_from_rows(db_path, ROWS[:1], source="two")
            db = LexicalDB(db_path)
            try:
                self.assertEqual(db.count(), 1)
                self.assertEqual(db.meta()["source"], "two")
            finally:
                db.close()
            self.assertFalse(db_path.with_suffix(".sqlite.staging").exists())

    def test_missing_db_raises_unavailable(self):
        with self.assertRaises(LexicalUnavailable):
            LexicalDB("/nonexistent/lex.sqlite")

    def test_malformed_match_query_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "lex.sqlite"
            build_from_rows(db_path, ROWS, source="test")
            db = LexicalDB(db_path)
            try:
                self.assertEqual(db.fts_search('"unclosed'), [])
                self.assertEqual(db.fts_search(""), [])
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
