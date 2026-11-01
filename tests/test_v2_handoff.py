"""Regression coverage for missing V2 runtime files and flattened report links."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts import package_v2_handoff as pack


class V2HandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in pack.REQUIRED:
            self.put(name, "fixture\n")
        self.put("docs/V2_FINAL_PROTOCOL.json", json.dumps({"status": "frozen",
            "study_id": "ragtruth-v2-fresh-original-train-holdout",
            "original_official_test_read": False, "inputs_sha256": {"structured_transformer.py":
                hashlib.sha256(b"fixture\n").hexdigest()}}))
        self.put("docs/DATA_LICENSE.txt", "MIT License\n")
        self.put("results/FINAL_REPORT.md", "Historical report\n")
        self.put("results/v2/FINAL_REPORT.md", "[Plan](../../docs/V2_PLAN.md)\n")
        self.put("docs/V2_PLAN.md", "Fixed plan\n")
        self.put("README.md", "[Results](results/v2/FINAL_REPORT.md)\n")
        self.demo = self.put("artifacts/demo.html", "<html>Saved dev predictions</html>")
        self.output = self.root / "artifacts/audited.zip"

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
        return path

    def test_runtime_files_and_relative_report_paths_are_preserved(self):
        pack.build_package(self.root, self.demo, self.output)
        pack.base.verify_archive(self.output)
        with zipfile.ZipFile(self.output) as archive:
            self.assertTrue(set(pack.REQUIRED) <= set(archive.namelist()))
            self.assertIn("results/FINAL_REPORT.md", archive.namelist())
            self.assertIn("docs/V2_PLAN.md", archive.namelist())
            self.assertEqual(archive.read("structured_transformer.py"), b"fixture\n")
            self.assertIn(b"results/v2/FINAL_REPORT.md", archive.read("README.md"))

    def test_missing_v2_model_or_split_contract_is_a_hard_failure(self):
        for name in ("structured_transformer.py", "data/splits/split_manifest_v2.csv"):
            path = self.root / name
            before = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing"):
                pack.build_package(self.root, self.demo, self.output)
            self.assertFalse(self.output.exists())
            path.write_bytes(before)

    def test_frozen_model_source_mismatch_is_rejected(self):
        self.put("structured_transformer.py", "changed\n")
        with self.assertRaisesRegex(ValueError, "frozen V2 identity"):
            pack.build_package(self.root, self.demo, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_delivery_is_never_overwritten(self):
        pack.build_package(self.root, self.demo, self.output)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            pack.build_package(self.root, self.demo, self.output)
        self.assertEqual(self.output.read_bytes(), before)

    def test_wrong_study_identity_is_rejected(self):
        path = self.root / "docs/V2_FINAL_PROTOCOL.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        protocol["study_id"] = "other-study"
        self.put("docs/V2_FINAL_PROTOCOL.json", json.dumps(protocol))
        with self.assertRaisesRegex(ValueError, "frozen V2 protocol"):
            pack.build_package(self.root, self.demo, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
