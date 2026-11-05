"""Byte-only dataset delivery checks must reject incomplete/tampered payloads."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_submission_data as delivery


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def fixture(root):
    # Intentionally invalid JSON lines prove dataset rows are never decoded.
    for relative in delivery.FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = 3 if relative == "data/raw/response.jsonl" else 3 if relative == "data/raw/source_info.jsonl" else 1
        path.write_bytes(b"THIS IS NOT JSON\n" * lines)
    identities = {relative: delivery.byte_identity(root / relative) for relative in delivery.FILES}
    source = {"revision": "a" * 40, "sha256": {Path(relative).name: identities[relative]["sha256"]
              for relative in delivery.RAW}}
    write_json(root / delivery.SOURCE, source)
    summary = {"revision": "a" * 40, "raw_sha256": {Path(relative).name: identities[relative]["sha256"]
                for relative in delivery.RAW if not relative.endswith("LICENSE")},
               "processed_sha256": {Path(relative).name: identities[relative]["sha256"]
                for relative in delivery.PREPARED if "test.jsonl" not in relative},
               "responses_per_split": {split: 1 for split in ("train", "validation", "test")},
               "sources_per_split": {split: {"total": 1} for split in ("train", "validation", "test")}}
    write_json(root / delivery.SUMMARY, summary)
    (root / delivery.MEMBERSHIP).write_bytes(b"fixture membership bytes\n")
    final = {"status": "frozen", "code_sha256": {delivery.SOURCE:
             delivery.byte_identity(root / delivery.SOURCE)["sha256"]},
             "data_sha256": {relative: value["sha256"] for relative, value in identities.items()
                             if "test.jsonl" not in relative}}
    final["data_sha256"].update({relative: delivery.byte_identity(root / relative)["sha256"]
                                for relative in (delivery.SUMMARY, delivery.MEMBERSHIP)})
    write_json(root / delivery.FINAL, final)
    for relative in (delivery.EVALUATION, delivery.CRF_TEST):
        protocol = {"status": "frozen", "scientific_decision": delivery.FINAL,
                    "scientific_decision_sha256": delivery.byte_identity(root / delivery.FINAL)["sha256"],
                    "inputs_sha256": {name: value["sha256"] for name, value in identities.items()}}
        protocol["inputs_sha256"].update({name: delivery.byte_identity(root / name)["sha256"]
                                        for name in (delivery.SOURCE, delivery.SUMMARY,
                                                     delivery.MEMBERSHIP, delivery.FINAL)})
        write_json(root / relative, protocol)
    manifest = {"format": "ragtruth-data-delivery-v1", "upstream_revision": "a" * 40,
                "references_sha256": {name: delivery.byte_identity(root / name)["sha256"]
                                      for name in delivery.REFERENCES},
                "files": identities, "total_bytes": sum(x["size_bytes"] for x in identities.values())}
    write_json(root / delivery.MANIFEST, manifest)
    return manifest


class SubmissionDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = fixture(self.root)

    def test_exact_bytes_pass_without_decoding_any_dataset_labels(self):
        original_json = delivery._json
        decoded = []
        def metadata_only(path):
            relative = Path(path).relative_to(self.root).as_posix()
            self.assertNotIn(relative, delivery.FILES)
            decoded.append(relative)
            return original_json(path)
        with patch.object(delivery, "_json", side_effect=metadata_only):
            report = delivery.verify(self.root)
        self.assertEqual(report["dataset_files"], 9)
        self.assertFalse(report["test_labels_parsed"])
        self.assertFalse(report["tracked_check"])
        self.assertEqual(set(decoded), {delivery.MANIFEST, delivery.SOURCE, delivery.SUMMARY,
                                       delivery.FINAL, delivery.EVALUATION, delivery.CRF_TEST})

    def test_changed_file_rejected_and_never_overwritten(self):
        path = self.root / "data/processed/test.jsonl"
        path.write_bytes(b"changed, still not JSON\n")
        with self.assertRaisesRegex(ValueError, "bytes/size/line count mismatch"):
            delivery.verify(self.root)
        self.assertEqual(path.read_bytes(), b"changed, still not JSON\n")

    def test_missing_file_is_rejected(self):
        (self.root / "data/raw/LICENSE").unlink()
        with self.assertRaisesRegex(ValueError, "missing or nonlocal"):
            delivery.verify(self.root)

    def test_forged_delivery_hash_cannot_replace_old_pinned_identity(self):
        relative = "data/processed/crf_test.jsonl"
        self.manifest["files"][relative]["sha256"] = "f" * 64
        write_json(self.root / delivery.MANIFEST, self.manifest)
        with self.assertRaisesRegex(ValueError, "pinned historical hash"):
            delivery.verify(self.root)

    def test_changed_reference_rejected_before_data(self):
        path = self.root / delivery.SOURCE
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "reference SHA-256 mismatch"):
            delivery.verify(self.root)

    def test_consistently_forged_config_and_manifest_still_conflicts_with_frozen_protocol(self):
        path = self.root / delivery.SOURCE
        source = json.loads(path.read_text())
        source["sha256"]["response.jsonl"] = "f" * 64
        write_json(path, source)
        self.manifest["references_sha256"][delivery.SOURCE] = delivery.byte_identity(path)["sha256"]
        write_json(self.root / delivery.MANIFEST, self.manifest)
        with self.assertRaisesRegex(ValueError, "source config identity differs"):
            delivery.verify(self.root)

    def test_missing_or_extra_delivery_entries_and_git_size_limit_are_rejected(self):
        entry = self.manifest["files"].pop("data/raw/LICENSE")
        write_json(self.root / delivery.MANIFEST, self.manifest)
        with self.assertRaisesRegex(ValueError, "exactly the nine"):
            delivery.verify(self.root)
        self.manifest["files"]["data/raw/LICENSE"] = entry
        entry["size_bytes"] = delivery.MAX_FILE_BYTES
        write_json(self.root / delivery.MANIFEST, self.manifest)
        with self.assertRaisesRegex(ValueError, "100 MiB"):
            delivery.verify(self.root)

    def test_tracked_option_checks_every_file_and_fails_closed(self):
        with patch.object(delivery.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run:
            report = delivery.verify(self.root, require_tracked=True)
        self.assertTrue(report["tracked_check"])
        arguments = run.call_args.args[0]
        self.assertTrue(set(delivery.FILES) <= set(arguments))
        self.assertIn(delivery.MANIFEST, arguments)
        with patch.object(delivery.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
            with self.assertRaisesRegex(ValueError, "not tracked"):
                delivery.verify(self.root, require_tracked=True)

    def test_row_count_must_match_existing_split_summary(self):
        relative = "data/raw/response.jsonl"
        # Identity remains genuine; claiming a different observed line count fails.
        self.manifest["files"][relative]["line_count"] = 4
        write_json(self.root / delivery.MANIFEST, self.manifest)
        with self.assertRaisesRegex(ValueError, "line count mismatch"):
            delivery.verify(self.root)


if __name__ == "__main__":
    unittest.main()
