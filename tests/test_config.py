"""Characterization tests for app.config — defaults and env overrides."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from app import config

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPathSettings(unittest.TestCase):
    def test_default_db_path(self):
        with patch.dict(os.environ):
            os.environ.pop("SQLITE_DB_PATH", None)
            self.assertEqual(config.get_db_path(), Path(".state/youtube_ingest.sqlite"))

    def test_db_path_env_override(self):
        with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/other.sqlite"}):
            self.assertEqual(config.get_db_path(), Path("/tmp/other.sqlite"))

    def test_default_chroma_path(self):
        with patch.dict(os.environ):
            os.environ.pop("CHROMA_PATH", None)
            self.assertEqual(config.get_chroma_path(), Path("data/chroma"))

    def test_chroma_path_env_override(self):
        with patch.dict(os.environ, {"CHROMA_PATH": "/tmp/other_chroma"}):
            self.assertEqual(config.get_chroma_path(), Path("/tmp/other_chroma"))

    def test_paths_are_pathlib(self):
        self.assertIsInstance(config.get_db_path(), Path)
        self.assertIsInstance(config.get_chroma_path(), Path)

    def test_default_constants(self):
        self.assertEqual(config.DEFAULT_DB_PATH, ".state/youtube_ingest.sqlite")
        self.assertEqual(config.DEFAULT_CHROMA_PATH, "data/chroma")


class TestImportTimeSettings(unittest.TestCase):
    """CHROMA_COLLECTION and EMBED_MODEL_NAME are frozen at import time, so
    they are exercised in a subprocess with a controlled environment."""

    def _read_constants(self, extra_env: dict) -> list[str]:
        env = {k: v for k, v in os.environ.items()
               if k not in ("CHROMA_COLLECTION", "EMBED_MODEL_NAME")}
        env.update(extra_env)
        env["PYTHONPATH"] = str(REPO_ROOT)
        out = subprocess.run(
            [sys.executable, "-c",
             "import app.config as c; print(c.CHROMA_COLLECTION); print(c.EMBED_MODEL_NAME)"],
            capture_output=True, text=True, cwd=REPO_ROOT, env=env, check=True)
        return out.stdout.splitlines()

    def test_defaults(self):
        collection, model = self._read_constants({})
        self.assertEqual(collection, "youtube_chunks")
        self.assertEqual(model, "intfloat/multilingual-e5-large")

    def test_env_overrides(self):
        collection, model = self._read_constants(
            {"CHROMA_COLLECTION": "test_collection_xyz", "EMBED_MODEL_NAME": "test-model-xyz"})
        self.assertEqual(collection, "test_collection_xyz")
        self.assertEqual(model, "test-model-xyz")


class TestOllamaSettings(unittest.TestCase):
    def test_fixed_generation_settings(self):
        self.assertEqual(config.OLLAMA_BASE_URL, "http://127.0.0.1:11434")
        self.assertEqual(config.OLLAMA_MODEL, "qwen3:0.6b")
        self.assertEqual(config.OLLAMA_PROBE_TIMEOUT_S, 2)
        self.assertEqual(config.OLLAMA_GENERATE_TIMEOUT_S, 120)
        self.assertEqual(config.OLLAMA_NUM_PREDICT, 1024)
        self.assertEqual(config.OLLAMA_TEMPERATURE, 0.2)


if __name__ == "__main__":
    unittest.main()
