"""Query-rewriter tests: strict JSON contract, every failure keeps the
original query. Ollama HTTP is fully mocked."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from rag.query_rewriter import rewrite_query


def _resp(status=200, response_text='{"query": "better query"}'):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = {"response": response_text}
    return m


class TestRewriteQuery(unittest.TestCase):
    def test_happy_path(self):
        with patch("requests.post", return_value=_resp()) as post:
            out = rewrite_query("orig query")
        self.assertEqual(out, "better query")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["temperature"], 0.0)
        self.assertEqual(payload["format"], "json")

    def test_think_tags_stripped(self):
        raw = '<think>musing</think>{"query": "clean"}'
        with patch("requests.post", return_value=_resp(response_text=raw)):
            self.assertEqual(rewrite_query("orig"), "clean")

    def test_http_error_returns_none(self):
        with patch("requests.post", return_value=_resp(status=500)):
            self.assertIsNone(rewrite_query("orig"))

    def test_connection_error_returns_none(self):
        with patch("requests.post", side_effect=OSError("refused")):
            self.assertIsNone(rewrite_query("orig"))

    def test_bad_json_returns_none(self):
        with patch("requests.post", return_value=_resp(response_text="not json")):
            self.assertIsNone(rewrite_query("orig"))

    def test_missing_query_key_returns_none(self):
        with patch("requests.post", return_value=_resp(response_text='{"x": 1}')):
            self.assertIsNone(rewrite_query("orig"))

    def test_non_string_query_returns_none(self):
        with patch("requests.post", return_value=_resp(response_text='{"query": 5}')):
            self.assertIsNone(rewrite_query("orig"))

    def test_empty_rewrite_returns_none(self):
        with patch("requests.post", return_value=_resp(response_text='{"query": "  "}')):
            self.assertIsNone(rewrite_query("orig"))

    def test_identical_rewrite_returns_none(self):
        with patch("requests.post",
                   return_value=_resp(response_text='{"query": "Orig"}')):
            self.assertIsNone(rewrite_query("orig"))

    def test_overlong_rewrite_truncated(self):
        long = '{"query": "' + "x" * 500 + '"}'
        with patch("requests.post", return_value=_resp(response_text=long)):
            out = rewrite_query("orig")
        self.assertEqual(len(out), 300)


if __name__ == "__main__":
    unittest.main()
