"""Characterization tests for Chroma result mapping and the RAG pipeline.

Chroma and the embedding model are faked — no real E5 weights are ever
loaded here.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from fastapi import HTTPException

from app import config
from rag import pipeline
from rag.transcript_retriever import retrieve_chunks

CHUNK_KEYS = {"id", "distance", "sim_pct", "vt", "vid", "s", "e", "url", "txt"}


class _FakeVector:
    def tolist(self):
        return [0.1, 0.2, 0.3]


class _FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, convert_to_numpy=False):
        self.calls.append((texts, convert_to_numpy))
        return [_FakeVector()]


def _chroma_result():
    long_title = "T" * 150
    long_doc = "d" * 1300
    return {
        "ids": [["c1", "c2", "c3"]],
        "metadatas": [[
            {"video_title": long_title, "video_id": "youtube_id:vidA",
             "start_hms": "1:00:00", "end_hms": "1:02:00",
             "youtube_time_url": "https://youtu.be/vidA?t=3600"},
            {"video_title": "Short", "video_id": "vidB",
             "start_hms": "0:05", "end_hms": "0:59",
             "youtube_time_url": "https://youtu.be/vidB?t=5"},
            {},  # missing metadata falls back to Unknown/N/A
        ]],
        "documents": [[long_doc, "text two", "text three"]],
        "distances": [[0.25, 0.0, None]],
    }


class TestRetrieveChunks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        env = patch.dict(os.environ, {"CHROMA_PATH": self.tmp.name})
        env.start()
        self.addCleanup(env.stop)

        self.model = _FakeModel()
        model_patch = patch("rag.transcript_retriever.get_embedding_model",
                            return_value=self.model)
        model_patch.start()
        self.addCleanup(model_patch.stop)

        self.collection = MagicMock()
        self.collection.query.return_value = _chroma_result()
        self.client = MagicMock()
        self.client.get_collection.return_value = self.collection
        client_patch = patch("rag.transcript_retriever.chromadb.PersistentClient",
                             return_value=self.client)
        self.client_cls = client_patch.start()
        self.addCleanup(client_patch.stop)

    def test_query_uses_e5_prefix_and_include_fields(self):
        retrieve_chunks("my question", 7)
        self.assertEqual(self.model.calls, [(["query: my question"], True)])
        self.client_cls.assert_called_once_with(path=self.tmp.name)
        self.client.get_collection.assert_called_once_with(config.CHROMA_COLLECTION)
        _, kwargs = self.collection.query.call_args
        self.assertEqual(kwargs["n_results"], 7)
        self.assertEqual(kwargs["include"], ["metadatas", "documents", "distances"])
        self.assertEqual(kwargs["query_embeddings"], [[0.1, 0.2, 0.3]])

    def test_chunk_mapping(self):
        chunks = retrieve_chunks("q")
        self.assertEqual(len(chunks), 3)
        for c in chunks:
            self.assertEqual(set(c.keys()), CHUNK_KEYS)
        first = chunks[0]
        self.assertEqual(first["id"], "c1")
        self.assertEqual(first["distance"], 0.25)
        self.assertEqual(first["sim_pct"], 75.0)
        self.assertEqual(first["vt"], "T" * 100)      # title truncated to 100
        self.assertEqual(first["vid"], "youtube_id:vidA")
        self.assertEqual(first["s"], "1:00:00")
        self.assertEqual(first["e"], "1:02:00")
        self.assertEqual(first["url"], "https://youtu.be/vidA?t=3600")
        self.assertEqual(first["txt"], "d" * 1200)     # document truncated to 1200

    def test_zero_and_missing_distance_map_to_zero_similarity(self):
        chunks = retrieve_chunks("q")
        self.assertEqual(chunks[1]["distance"], 0.0)
        self.assertEqual(chunks[1]["sim_pct"], 0)      # falsy distance -> 0 (existing quirk)
        self.assertIsNone(chunks[2]["distance"])
        self.assertEqual(chunks[2]["sim_pct"], 0)

    def test_missing_metadata_defaults(self):
        c = retrieve_chunks("q")[2]
        self.assertEqual(c["vt"], "Unknown")
        self.assertEqual(c["vid"], "Unknown")
        self.assertEqual(c["s"], "N/A")
        self.assertEqual(c["e"], "N/A")
        self.assertEqual(c["url"], "N/A")

    def test_empty_ids_return_no_chunks(self):
        self.collection.query.return_value = {"ids": [[]], "metadatas": [[]],
                                              "documents": [[]], "distances": [[]]}
        self.assertEqual(retrieve_chunks("q"), [])

    def test_missing_documents_return_no_chunks(self):
        self.collection.query.return_value = {"ids": [["c1"]], "metadatas": [[{}]],
                                              "documents": [[]], "distances": [[0.5]]}
        self.assertEqual(retrieve_chunks("q"), [])

    def test_collection_error_maps_to_http_500(self):
        self.client.get_collection.side_effect = ValueError("nope")
        with self.assertRaises(HTTPException) as ctx:
            retrieve_chunks("q")
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertTrue(ctx.exception.detail.startswith("Collection not found:"))


class TestMissingChromaPath(unittest.TestCase):
    def test_missing_path_raises_http_500(self):
        with patch.dict(os.environ, {"CHROMA_PATH": "/nonexistent/chroma_dir"}):
            with self.assertRaises(HTTPException) as ctx:
                retrieve_chunks("q")
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(ctx.exception.detail, "ChromaDB not found")


class TestPipeline(unittest.TestCase):
    CHUNKS = [{"id": "c1", "s": "0:01", "e": "0:02", "vt": "T", "vid": "v", "txt": "x",
               "distance": 0.3, "sim_pct": 70.0, "url": "u"}]

    def test_summary_generated_when_chunks_and_ollama(self):
        with patch("rag.pipeline.retrieve_chunks", return_value=self.CHUNKS) as ret, \
             patch.object(pipeline.generator, "OLLAMA_AVAILABLE", True), \
             patch.object(pipeline.generator, "llm_answer",
                          Mock(return_value="ANSWER")) as llm:
            result = pipeline.search("q", top_k=3)
        ret.assert_called_once_with("q", 3)
        llm.assert_called_once_with("q", self.CHUNKS)
        self.assertEqual(result, {"chunks": self.CHUNKS, "summary": "ANSWER"})

    def test_generation_failure_falls_back_to_empty_summary(self):
        with patch("rag.pipeline.retrieve_chunks", return_value=self.CHUNKS), \
             patch.object(pipeline.generator, "OLLAMA_AVAILABLE", True), \
             patch.object(pipeline.generator, "llm_answer",
                          Mock(side_effect=RuntimeError("boom"))):
            result = pipeline.search("q")
        self.assertEqual(result, {"chunks": self.CHUNKS, "summary": ""})

    def test_no_chunks_skips_generation(self):
        with patch("rag.pipeline.retrieve_chunks", return_value=[]), \
             patch.object(pipeline.generator, "OLLAMA_AVAILABLE", True), \
             patch.object(pipeline.generator, "llm_answer", Mock()) as llm:
            result = pipeline.search("q")
        llm.assert_not_called()
        self.assertEqual(result, {"chunks": [], "summary": ""})

    def test_ollama_offline_skips_generation(self):
        with patch("rag.pipeline.retrieve_chunks", return_value=self.CHUNKS), \
             patch.object(pipeline.generator, "OLLAMA_AVAILABLE", False), \
             patch.object(pipeline.generator, "llm_answer", Mock()) as llm:
            result = pipeline.search("q")
        llm.assert_not_called()
        self.assertEqual(result, {"chunks": self.CHUNKS, "summary": ""})


if __name__ == "__main__":
    unittest.main()
