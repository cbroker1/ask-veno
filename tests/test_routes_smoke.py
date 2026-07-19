"""Route smoke tests via FastAPI TestClient (httpx is already in the env).

The app is imported through web_app to keep exercising the compatibility
entrypoint (`uvicorn web_app:app`). SQLite is pointed at a nonexistent path
so pages render the empty-archive state; the RAG pipeline is mocked — no
Chroma, E5, or Ollama calls happen here.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from web_app import app

STATS_KEYS = {"total", "completed", "in_progress", "failed", "total_chunks"}

FAKE_CHUNKS = [
    {"id": "vidA_chunk_1", "distance": 0.295, "sim_pct": 70.5, "vt": "Test Stream",
     "vid": "youtube_id:vidA", "s": "1:48:38", "e": "1:49:50",
     "url": "https://youtu.be/vidA?t=6518", "txt": "some transcript text"},
]


class RouteSmokeTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"SQLITE_DB_PATH": "/nonexistent/nowhere.sqlite"})
        env.start()
        self.addCleanup(env.stop)
        self.client = TestClient(app)

    def test_dashboard_returns_html(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/html"))
        self.assertIn("ASK VENO", r.text)
        self.assertIn("ARCHIVE EMPTY", r.text)
        self.assertNotIn("{{", r.text)  # every placeholder resolved

    def test_api_stats_shape(self):
        r = self.client.get("/api/stats")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json().keys()), STATS_KEYS)

    def test_api_videos_returns_list(self):
        r = self.client.get("/api/videos?limit=3")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.json(), list)

    def test_search_empty_string_is_422(self):
        # FastAPI treats an empty required form string as missing — current
        # behavior captured from the original app.
        r = self.client.post("/search", data={"query": ""})
        self.assertEqual(r.status_code, 422)

    def test_search_missing_field_is_422(self):
        r = self.client.post("/search")
        self.assertEqual(r.status_code, 422)

    def test_search_whitespace_renders_dashboard(self):
        r = self.client.post("/search", data={"query": "   "})
        self.assertEqual(r.status_code, 200)
        # the CSS block always mentions .echo-bar — only the rendered element
        # must be absent on the plain dashboard
        self.assertNotIn('<div class="echo-bar">', r.text)
        self.assertNotIn("STREAM EVIDENCE", r.text)

    def test_search_renders_results_and_report(self):
        fake = {"chunks": FAKE_CHUNKS, "summary": "A direct answer."}
        with patch("rag.pipeline.search", return_value=fake) as mock_search:
            r = self.client.post("/search", data={"query": "best weapon"})
        mock_search.assert_called_once_with("best weapon", 5)
        self.assertEqual(r.status_code, 200)
        self.assertIn("STREAM EVIDENCE", r.text)
        self.assertIn("FIELD REPORT", r.text)
        self.assertIn("best weapon", r.text)
        self.assertIn("70.5% SIGNAL", r.text)

    def test_search_error_renders_fault_card(self):
        with patch("rag.pipeline.search", side_effect=RuntimeError("boom")):
            r = self.client.post("/search", data={"query": "anything"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("SCAN FAULT", r.text)
        self.assertIn("boom", r.text)

    def test_api_search_delegates_to_pipeline(self):
        fake = {"chunks": FAKE_CHUNKS, "summary": "s"}
        with patch("rag.pipeline.search", return_value=fake) as mock_search:
            r = self.client.post("/api/search", data={"query": "q", "top_k": "3"})
        mock_search.assert_called_once_with("q", top_k=3)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), fake)

    def test_api_search_default_top_k(self):
        fake = {"chunks": [], "summary": ""}
        with patch("rag.pipeline.search", return_value=fake) as mock_search:
            r = self.client.post("/api/search", data={"query": "q"})
        mock_search.assert_called_once_with("q", top_k=5)
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
