"""REQUIRED regression: corpus="transcripts" must reproduce the pre-Phase-2
baseline exactly — IDs, ordering, video ids, timestamps, URLs, excerpts,
similarity values.

The baseline JSON files were captured from the unmodified Phase 1 app
(master 238dabe) in .state/phase2_baseline/ before any Phase 2 change.
Because live comparison needs the real Chroma store and the E5 model, these
tests only run when ASKVENO_INTEGRATION=1 is set; the plain unit suite
skips them (no multi-GB model loads by default).

    ASKVENO_INTEGRATION=1 python -m unittest tests.test_transcript_regression -v
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parents[1] / ".state" / "phase2_baseline"

BASELINE_QUERIES = {
    "api_search_q1.json": "best weapon for killing mutants",
    "api_search_q2.json": "what does veno think about the AK-74",
    "api_search_q3.json": "surviving emissions and psi storms",
}

RUN = os.environ.get("ASKVENO_INTEGRATION") == "1"


@unittest.skipUnless(RUN, "integration regression (set ASKVENO_INTEGRATION=1)")
@unittest.skipUnless(BASELINE_DIR.exists(), "no recorded phase2 baseline")
class TestTranscriptRegression(unittest.TestCase):
    maxDiff = None

    def _compare(self, baseline_file: str, query: str):
        from rag import pipeline
        baseline = json.loads((BASELINE_DIR / baseline_file).read_text())
        live = pipeline.search(query, 5, corpus="transcripts")

        base_chunks = baseline["chunks"]
        live_chunks = live["chunks"]
        self.assertEqual([c["id"] for c in live_chunks],
                         [c["id"] for c in base_chunks], "id/order changed")
        for lc, bc in zip(live_chunks, base_chunks):
            for key in ("id", "distance", "sim_pct", "vt", "vid", "s", "e",
                        "url", "txt"):
                self.assertEqual(lc[key], bc[key],
                                 f"{key} regressed for chunk {bc['id']}")
        # legacy response fields still present
        self.assertIn("summary", live)
        self.assertIn("chunks", live)

    def test_q1(self):
        self._compare("api_search_q1.json", BASELINE_QUERIES["api_search_q1.json"])

    def test_q2(self):
        self._compare("api_search_q2.json", BASELINE_QUERIES["api_search_q2.json"])

    def test_q3(self):
        self._compare("api_search_q3.json", BASELINE_QUERIES["api_search_q3.json"])


if __name__ == "__main__":
    unittest.main()
