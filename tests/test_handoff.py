"""The technical handoff contains only allowed, checked local artifacts."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import package_handoff as handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.demo = self.put("artifacts/comparison-demo.html", "<!doctype html><p>Saved validation examples</p>")
        self.results = self.root / "results"
        self.put("results/validation.json", '{"split":"validation","models":{}}')
        self.put("results/model_comparison.svg", "<svg xmlns='http://www.w3.org/2000/svg'></svg>")
        self.put("docs/DATA_LICENSE.txt", "MIT License\nCopyright fixture\n")
        self.put("docs/RUNNING.md", "Reproducible local commands.")
        self.put("README.md", "Original project README.")
        self.put("transformer.py", "# Model source.\n")
        self.put("scripts/train.py", "# Reproducible training.\n")
        self.put("ragtruth_data/schema.py", "LABEL_TYPES = ()\n")
        self.put("configs/model.json", '{"seed":42}')
        self.put("requirements-transformer.txt", "torch>=2.6,<3\n")
        self.put("demo/viewer.html", "<main>Reusable saved-output viewer.</main>")
        self.put("demo/viewer.css", "main { color: black; }")
        self.put("demo/viewer.js", "// Offline UI only.\n")
        self.put("data/splits/split_manifest_v1.csv", "source_id,split\nfixture,train\n")
        self.put("data/splits/split_summary_v1.json", '{"seed":42}')
        self.put("data/samples/example.jsonl", '{"id":"example"}\n')
        self.output = self.root / "artifacts/handoff.zip"

    def put(self, name, contents):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def test_manifest_crc_payload_bytes_and_generated_readme(self):
        original = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        result = handoff.build_package(self.root, self.demo, self.results, self.output)
        manifest = handoff.verify_archive(self.output)
        self.assertEqual(result["sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        with zipfile.ZipFile(self.output) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(set(archive.namelist()), set(manifest["files"]) | {"MANIFEST.json"})
            self.assertEqual(archive.read("demo.html"), self.demo.read_bytes())
            self.assertIn("docs/DATA_LICENSE.txt", archive.namelist())
            self.assertIn("transformer.py", archive.namelist())
            self.assertIn("data/samples/example.jsonl", archive.namelist())
            for name in ("demo/viewer.html", "demo/viewer.css", "demo/viewer.js",
                         "data/splits/split_manifest_v1.csv", "data/splits/split_summary_v1.json"):
                self.assertIn(name, archive.namelist())
            self.assertIn("saved VALIDATION", archive.read("README.md").decode())
            self.assertIn("does not perform live", archive.read("README.md").decode())
            self.assertEqual(archive.read("PROJECT_README.md"), original[self.root / "README.md"])
        self.assertTrue(all(path.read_bytes() == payload for path, payload in original.items()))
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            handoff.build_package(self.root, self.demo, self.results, self.output)
        self.assertEqual(self.output.read_bytes(), before)

    def test_secrets_dotfiles_model_weights_raw_data_and_slides_are_absent(self):
        excluded = [".env", ".git/config", ".cache/token.txt", ".venv/config.json", "docs/.hidden.md",
                    "docs/api_key.txt", "configs/credentials.json", "scripts/password.py",
                    "results/weights.json", "results/checkpoints/metadata.json", "results/model.safetensors",
                    "results/model.crfsuite", "data/raw/data.jsonl", "data/processed/validation.jsonl",
                    "contributions/person.md", "docs/personal_report.md", "slides/talk.pptx"]
        for name in excluded:
            self.put(name, "private fixture data")
        handoff.build_package(self.root, self.demo, self.results, self.output)
        with zipfile.ZipFile(self.output) as archive:
            names = archive.namelist()
            self.assertFalse(any(name in names for name in excluded))
            self.assertFalse(any(name.startswith(".") or "password" in name or "credentials" in name for name in names))
            self.assertNotIn(b"private fixture data", b"".join(archive.read(name) for name in names))

    def test_explicit_outside_paths_and_output_outside_ignored_directories_fail(self):
        with tempfile.TemporaryDirectory() as external:
            outside = Path(external) / "outside.html"
            outside.write_text("external", encoding="utf-8")
            for demo, results, output in ((outside, self.results, self.output),
                                          (self.demo, Path(external), self.output),
                                          (self.demo, self.results, Path(external) / "outside.zip"),
                                          (self.demo, self.results, self.root / "unsafe.zip")):
                with self.subTest(demo=demo, results=results, output=output), self.assertRaises(ValueError):
                    handoff.build_package(self.root, demo, results, output)
        self.assertFalse(self.output.exists())

    def test_symlink_escape_or_hidden_target_is_rejected(self):
        # Creating symlinks may require Windows Developer Mode; canonical helper
        # behavior is still exercised through actual symlinks when available.
        with tempfile.TemporaryDirectory() as external:
            target = Path(external) / "outside.md"
            target.write_text("outside content", encoding="utf-8")
            alias = self.root / "docs/linked.md"
            try:
                alias.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("this Windows account cannot create symlinks")
            with self.assertRaisesRegex(ValueError, "leaves"):
                handoff.build_package(self.root, self.demo, self.results, self.output)
            alias.unlink()
            target = self.put(".private.txt", "hidden target")
            alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "excluded"):
                handoff.build_package(self.root, self.demo, self.results, self.output)

    def test_embedded_credential_blocks_packaging_without_printing_the_value(self):
        credential = "github_" + "pat_" + "A" * 40
        self.put("docs/innocent.md", "token=" + credential)
        with self.assertRaises(ValueError) as error:
            handoff.build_package(self.root, self.demo, self.results, self.output)
        self.assertNotIn(credential, str(error.exception))
        self.assertFalse(self.output.exists())

    def test_tampered_manifest_member_is_rejected(self):
        handoff.build_package(self.root, self.demo, self.results, self.output)
        damaged = self.root / "artifacts/tampered.zip"
        with zipfile.ZipFile(self.output) as source, zipfile.ZipFile(damaged, "w") as destination:
            for name in source.namelist():
                destination.writestr(name, b"changed" if name == "demo.html" else source.read(name))
        with self.assertRaisesRegex(ValueError, "checksum manifest"):
            handoff.verify_archive(damaged)

    def test_large_samples_are_omitted_and_cli_uses_selected_root(self):
        with patch.object(handoff, "MAX_SAMPLE_BYTES", 1), patch.object(handoff, "ROOT", self.root), \
             patch("sys.stdout", new_callable=io.StringIO):
            self.assertEqual(handoff.main(["--demo", str(self.demo), "--results-dir", str(self.results),
                                           "--output", str(self.output)]), 0)
        manifest = handoff.verify_archive(self.output)
        self.assertNotIn("data/samples/example.jsonl", manifest["files"])
        self.assertEqual(manifest["omitted_counts"]["sample_size_limit"], 1)


if __name__ == "__main__":
    unittest.main()
