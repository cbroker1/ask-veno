"""Unit tests for scripts/sync_gamma_095.py — no network, no real clone.

Git interactions are faked; dataset validation runs against small temp trees
shaped like the real gamma-0.9.5 export.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sync_gamma_095 as sync  # noqa: E402


def make_dataset(root: Path, manifest: dict | None = None) -> Path:
    data_dir = root / sync.GAMMA_DATA_SUBDIR
    data_dir.mkdir(parents=True)
    manifest = manifest if manifest is not None else {"index.json": "aabbccdd"}
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    (data_dir / "index.json").write_text("[]")
    return data_dir


class TestManifestParsing(unittest.TestCase):
    def test_flat_map_counts_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            p.write_text(json.dumps({"a.json": "11111111", "b.json": "22222222"}))
            self.assertEqual(sync.manifest_file_count(p), 2)
            self.assertEqual(sync.load_manifest(p),
                             {"a.json": "11111111", "b.json": "22222222"})

    def test_unexpected_structures_rejected(self):
        cases = [[], ["a.json"], {}, {"files": ["a.json"]}, {"a.json": 3}, "x"]
        for case in cases:
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "manifest.json"
                p.write_text(json.dumps(case))
                with self.assertRaises(ValueError, msg=repr(case)):
                    sync.load_manifest(p)

    def test_sha256_is_of_raw_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            p.write_bytes(b'{"a.json": "11111111"}')
            import hashlib
            self.assertEqual(sync.manifest_sha256(p),
                             hashlib.sha256(b'{"a.json": "11111111"}').hexdigest())


class TestValidateDataset(unittest.TestCase):
    def test_missing_data_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                sync.validate_dataset(Path(tmp))

    def test_missing_index_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = make_dataset(Path(tmp))
            (data_dir / "index.json").unlink()
            with self.assertRaises(ValueError):
                sync.validate_dataset(Path(tmp))

    def test_valid_tree_reports_pinned_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_dataset(Path(tmp), {"index.json": "aabbccdd", "rifles.json": "eeff0011"})
            fake_head = type("R", (), {"returncode": 0, "stdout": "abc123\n", "stderr": ""})()
            with patch.object(sync, "_git", return_value=fake_head):
                info = sync.validate_dataset(Path(tmp))
        self.assertEqual(info["pack_id"], "gamma-0.9.5")
        self.assertEqual(info["upstream_commit"], "abc123")
        self.assertEqual(info["manifest_file_count"], 2)
        self.assertEqual(info["upstream_url"], sync.GAMMA_UPSTREAM_URL)


class TestUpdateSafety(unittest.TestCase):
    def test_wrong_remote_refused(self):
        with patch.object(sync, "_remote_url",
                          return_value="https://github.com/evil/other-repo"):
            with self.assertRaises(ValueError) as ctx:
                sync.update(Path("/nonexistent"))
        self.assertIn("unexpected remote", str(ctx.exception))

    def test_dirty_worktree_refused(self):
        with patch.object(sync, "_remote_url", return_value=sync.GAMMA_UPSTREAM_URL), \
             patch.object(sync, "_worktree_dirty", return_value=True):
            with self.assertRaises(ValueError) as ctx:
                sync.update(Path("/nonexistent"))
        self.assertIn("local modifications", str(ctx.exception))

    def test_offline_fetch_reports_not_fetched(self):
        def fake_git(repo, *args, **kwargs):
            joined = " ".join(args)
            if joined.startswith("rev-parse"):
                return type("R", (), {"returncode": 0, "stdout": "abc\n", "stderr": ""})()
            if joined.startswith("fetch"):
                return type("R", (), {"returncode": 1, "stdout": "", "stderr": "no route"})()
            raise AssertionError(f"unexpected git call: {joined}")

        with patch.object(sync, "_remote_url", return_value=sync.GAMMA_UPSTREAM_URL), \
             patch.object(sync, "_worktree_dirty", return_value=False), \
             patch.object(sync, "_git", side_effect=fake_git):
            res = sync.update(Path("/nonexistent"))
        self.assertFalse(res["fetched"])
        self.assertEqual(res["error"], "no route")


class TestSyncOffline(unittest.TestCase):
    def test_offline_with_valid_local_data_is_stale_not_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_dataset(Path(tmp))
            (Path(tmp) / ".git").mkdir()
            fake_head = type("R", (), {"returncode": 0, "stdout": "abc\n", "stderr": ""})()
            with patch.object(sync, "update",
                              return_value={"fetched": False, "error": "offline"}), \
                 patch.object(sync, "_git", return_value=fake_head):
                res = sync.sync(Path(tmp))
        self.assertEqual(res["status"], "offline_stale")
        self.assertFalse(res["synced_new_commit"])

    def test_check_only_never_calls_git_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_dataset(Path(tmp))
            fake_head = type("R", (), {"returncode": 0, "stdout": "abc\n", "stderr": ""})()
            with patch.object(sync, "clone", side_effect=AssertionError("clone called")), \
                 patch.object(sync, "update", side_effect=AssertionError("update called")), \
                 patch.object(sync, "_git", return_value=fake_head):
                res = sync.sync(Path(tmp), check_only=True)
        self.assertEqual(res["status"], "ok")
        self.assertTrue(res["checked_only"])


if __name__ == "__main__":
    unittest.main()
