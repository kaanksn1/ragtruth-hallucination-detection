"""Checksum failures must never replace a teammate's existing raw file."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_data.py"
SPEC = importlib.util.spec_from_file_location("ragtruth_download", SCRIPT)
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


class DownloadTests(unittest.TestCase):
    def test_check_only_verifies_bytes_without_network_and_preserves_bad_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "repository": "https://github.com/ParticleMedia/RAGTruth", "revision": "a" * 40,
                "files": ["dataset/response.jsonl", "dataset/source_info.jsonl", "LICENSE"],
                "sha256": {},
            }
            for remote in config["files"]:
                name = Path(remote).name
                value = (name + " fixture").encode()
                (root / name).write_bytes(value)
                config["sha256"][name] = hashlib.sha256(value).hexdigest()
            manifest = root / "source.json"
            manifest.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(DOWNLOAD, "urlopen", side_effect=AssertionError("Unexpected network")):
                self.assertEqual(len(DOWNLOAD.download(manifest, root, check_only=True)), 3)
                (root / "response.jsonl").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "not overwritten"):
                    DOWNLOAD.download(manifest, root, check_only=True)
                self.assertEqual((root / "response.jsonl").read_bytes(), b"changed")


if __name__ == "__main__":
    unittest.main()
