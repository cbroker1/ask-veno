"""Characterization tests for storage.archive using a temporary SQLite DB."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage.archive import get_video_stats, get_videos

EXPECTED_EMPTY_STATS = {"total": 0, "completed": 0, "in_progress": 0,
                        "failed": 0, "total_chunks": 0}

ROW_KEYS = {"video_id", "title", "url", "upload_date", "duration", "ingest_status",
            "audio_status", "whisper_status", "embedding_status", "chunk_count",
            "ingest_st", "ingest_tt", "dur", "dt"}


def _make_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                url TEXT,
                upload_date TEXT,
                duration INTEGER,
                ingest_status TEXT,
                audio_status TEXT,
                whisper_status TEXT,
                embedding_status TEXT,
                chunk_count INTEGER,
                discovered_at TEXT
            )
        """)
        rows = [
            # (video_id, title, url, upload_date, duration, ingest, audio, whisper, embed, chunks, discovered)
            ("vidA", "Alpha Stream", "https://youtube.com/watch?v=vidA", "20240115",
             3725, "complete", "complete", "complete", "complete", 10, "2024-01-15T10:00:00"),
            ("vidB", "Bravo Stream", None, None,
             59, "queued", "not_started", "not_started", "not_started", 0, "2024-01-16T10:00:00"),
            ("vidC", "Charlie Stream", "https://youtube.com/watch?v=vidC", "20231201",
             125, "failed", "complete", "failed", "not_started", 5, "2024-01-14T10:00:00"),
            ("vidD", "Delta Stream", "https://youtube.com/watch?v=vidD", "20240117",
             None, "processing", "complete", "processing", "not_started", None, "2024-01-17T10:00:00"),
        ]
        conn.executemany("INSERT INTO videos VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)


class TestMissingDatabase(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SQLITE_DB_PATH": "/nonexistent/nowhere.sqlite"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_stats_empty_shape(self):
        self.assertEqual(get_video_stats(), EXPECTED_EMPTY_STATS)

    def test_videos_empty_list(self):
        self.assertEqual(get_videos(), [])


class TestPopulatedDatabase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        db_path = Path(self.tmp.name) / "test.sqlite"
        _make_db(db_path)
        env = patch.dict(os.environ, {"SQLITE_DB_PATH": str(db_path)})
        env.start()
        self.addCleanup(env.stop)

    def test_stats_mapping(self):
        self.assertEqual(get_video_stats(),
                         {"total": 4, "completed": 1, "in_progress": 2,
                          "failed": 1, "total_chunks": 15})

    def test_videos_ordered_by_discovered_at_desc(self):
        ids = [v["video_id"] for v in get_videos(20)]
        self.assertEqual(ids, ["vidD", "vidB", "vidA", "vidC"])

    def test_limit_is_parameterized(self):
        ids = [v["video_id"] for v in get_videos(2)]
        self.assertEqual(ids, ["vidD", "vidB"])

    def test_row_mapping_keys(self):
        for v in get_videos(20):
            self.assertEqual(set(v.keys()), ROW_KEYS)

    def test_formatting_fields(self):
        by_id = {v["video_id"]: v for v in get_videos(20)}
        alpha = by_id["vidA"]
        self.assertEqual(alpha["ingest_st"], "COMPLETE")
        self.assertEqual(alpha["ingest_tt"], "ok")
        self.assertEqual(alpha["dur"], "1h 2m")
        self.assertEqual(alpha["dt"], "2024-01-15")
        bravo = by_id["vidB"]
        self.assertEqual(bravo["ingest_st"], "QUEUED")
        self.assertEqual(bravo["ingest_tt"], "pending")
        self.assertEqual(bravo["dur"], "59s")
        self.assertEqual(bravo["dt"], "N/A")
        delta = by_id["vidD"]
        self.assertEqual(delta["ingest_st"], "PROCESSING")
        self.assertEqual(delta["ingest_tt"], "progress")
        self.assertEqual(delta["dur"], "0s")
        self.assertIsNone(delta["chunk_count"])

    def test_default_limit(self):
        self.assertEqual(len(get_videos()), 4)


if __name__ == "__main__":
    unittest.main()
