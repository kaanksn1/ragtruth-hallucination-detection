"""Development-only source leakage and shared boilerplate audit."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragtruth_data.data import sha256_file, write_jsonl
from scripts import audit_development_data as audit


def row(rid, source, source_text, response="Unable to answer based on given passages.", labels=None):
    return {"id": rid, "source_id": source, "source_text": source_text, "response": response, "labels": labels or []}


class DevelopmentAuditTests(unittest.TestCase):
    def test_shared_empty_gold_boilerplate_is_reported_without_source_leakage(self):
        train = [row("t1", "s1", "first source"), row("t2", "s2", "second source")]
        validation = [row("v1", "s3", "third source")]
        before = copy.deepcopy((train, validation))
        result = audit.audit_development_records(train, validation, {"s1": "g1", "s2": "g2", "s3": "g3"})
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["response_text_overlap"]["exact"]["distinct_texts"], 1)
        shared = result["response_text_overlap"]["exact"]["items"][0]
        self.assertEqual((shared["train_count"], shared["validation_count"]), (2, 1))
        self.assertTrue(shared["all_gold_empty"])
        self.assertEqual(result["rows_removed"], 0)
        self.assertEqual((train, validation), before)

    def test_duplicate_source_ids_and_groups_are_flagged(self):
        result = audit.audit_development_records([row("t", "s1", "a")], [row("v", "s1", "a")], {"s1": "g1"})
        self.assertEqual(result["status"], "failed_source_or_id_leakage")
        self.assertEqual(result["source_id_overlap"]["count"], 1)
        self.assertEqual(result["group_id_overlap"]["count"], 1)
        grouped = audit.audit_development_records([row("t", "s1", "a")], [row("v", "s2", "b")],
                                                   {"s1": "same", "s2": "same"})
        self.assertEqual(grouped["source_id_overlap"]["count"], 0)
        self.assertEqual(grouped["group_id_overlap"]["count"], 1)
        self.assertEqual(grouped["status"], "failed_source_or_id_leakage")

    def test_casefold_and_whitespace_source_duplicates_are_flagged(self):
        result = audit.audit_development_records([row("t", "s1", "  Straße\nContent ")],
            [row("v", "s2", "STRASSE   CONTENT")], {"s1": "g1", "s2": "g2"})
        self.assertEqual(result["source_text_overlap"]["exact"]["distinct_texts"], 0)
        self.assertEqual(result["source_text_overlap"]["normalized"]["distinct_texts"], 1)
        self.assertEqual(result["status"], "failed_source_or_id_leakage")

    def test_shared_annotated_response_is_warning_not_proof_of_source_leakage(self):
        result = audit.audit_development_records([row("t", "s1", "a", "SAME", [{"start": 0, "end": 4}])],
            [row("v", "s2", "b", " same ")], {"s1": "g1", "s2": "g2"})
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["response_text_overlap"]["exact"]["distinct_texts"], 0)
        self.assertFalse(result["response_text_overlap"]["normalized"]["all_shared_gold_empty"])
        self.assertTrue(result["warnings"])

    def test_missing_groups_or_duplicate_within_split_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "manifest group"):
            audit.audit_development_records([row("t", "s1", "a")], [row("v", "s2", "b")], {"s1": "g1"})
        with self.assertRaisesRegex(ValueError, "unique"):
            audit.audit_development_records([row("t", "s1", "a")] * 2, [row("v", "s2", "b")], {"s1": "g1", "s2": "g2"})

    def test_cli_checks_hashes_manifest_no_test_read_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, source in (("train", "s1"), ("validation", "s2")):
                write_jsonl(root / f"{split}.jsonl", [dict(row(split, source, split), split=split, quality="good")])
            # Invalid sentinel stays unopened: the script has no test split path.
            (root / "test.jsonl").write_bytes(b"\xff\x00not JSON")
            manifest, summary = root / "manifest.csv", root / "summary.json"
            manifest.write_text("source_id,group_id,split\ns1,g1,train\ns2,g2,validation\n", encoding="utf-8")
            summary.write_text(json.dumps({"processed_sha256": {f"{split}.jsonl": sha256_file(root / f"{split}.jsonl")
                                                                  for split in ("train", "validation")}}), encoding="utf-8")
            output = root / "audit.json"
            args = ["--processed-dir", str(root), "--manifest", str(manifest), "--split-summary", str(summary), "--output", str(output)]
            original = {name: sha256_file(root / name) for name in ("train.jsonl", "validation.jsonl", "test.jsonl")}
            with redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(args), 0)
            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(result["official_test_read"])
            self.assertEqual(result["input_sha256"]["train.jsonl"], original["train.jsonl"])
            self.assertEqual({name: sha256_file(root / name) for name in original}, original)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                audit.main(args)


if __name__ == "__main__":
    unittest.main()
