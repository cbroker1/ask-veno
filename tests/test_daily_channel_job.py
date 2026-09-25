"""Unit tests for scripts/daily_channel_job.py step orchestration.

Subprocess steps and the registry are faked; what matters is the step order
and that the transcript BM25 sync runs at the end of every job.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import daily_channel_job as job  # noqa: E402

PIPELINE = [
    "Download queued audio",
    "Whisper transcription",
    "Transcript cleanup",
    "ChromaDB embedding",
    "Final pipeline status",
]
LEXICAL = "Transcript lexical index sync"
NEW_VIDEO = {"video_id": "abc123def45", "upload_date": "20260924", "title": "t"}


class TestDailyChannelJob(unittest.TestCase):
    def run_job(self, new_videos, codes=None):
        codes = codes or {}
        calls = []

        def fake_step(name, command):
            calls.append((name, command))
            return codes.get(name, 0)

        with patch.object(job, "run_step", side_effect=fake_step), \
                patch.object(job, "connect_db"), \
                patch.object(job, "count_videos", return_value=0), \
                patch.object(job, "get_new_videos", return_value=new_videos), \
                patch.object(job, "log"):
            code = job.run_daily_job()
        return code, [name for name, _ in calls], calls

    def test_no_new_videos_still_syncs_lexical_index(self):
        code, names, calls = self.run_job([])
        self.assertEqual(code, 0)
        self.assertEqual(names, ["Exhaustive discovery", LEXICAL])
        self.assertEqual(calls[-1][1][1:], ["scripts/build_lexical_index.py"])

    def test_new_videos_run_pipeline_then_sync(self):
        code, names, _ = self.run_job([NEW_VIDEO])
        self.assertEqual(code, 0)
        self.assertEqual(names, ["Exhaustive discovery", *PIPELINE, LEXICAL])

    def test_discovery_failure_still_syncs_and_fails(self):
        code, names, _ = self.run_job([NEW_VIDEO], {"Exhaustive discovery": 1})
        self.assertEqual(code, 1)
        self.assertEqual(names, ["Exhaustive discovery", LEXICAL])

    def test_pipeline_failure_keeps_its_exit_code(self):
        code, names, _ = self.run_job([NEW_VIDEO], {"Whisper transcription": 2})
        self.assertEqual(code, 2)
        self.assertEqual(names, ["Exhaustive discovery", "Download queued audio",
                                 "Whisper transcription", LEXICAL])

    def test_lexical_sync_failure_fails_job(self):
        code, _, _ = self.run_job([], {LEXICAL: 1})
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
