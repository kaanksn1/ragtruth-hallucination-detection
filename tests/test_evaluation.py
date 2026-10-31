"""Common metric definitions, integrity failures and gated CLI integration."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate as cli
from ragtruth_data.data import sha256_file, write_jsonl
from ragtruth_data.schema import LABEL_TYPES
from ragtruth_eval import character_scores, evaluate_records, exact_span_scores, load_gold, read_predictions
from scripts.baseline_crf import char_scores as legacy_character_scores


def span(start, end, kind=0):
    return {"start": start, "end": end, "label_type": LABEL_TYPES[kind]}


def record(rid="one", text="abcdefgh", labels=None, **kwargs):
    return {"id": rid, "response": text, "labels": labels or [], "task_type": "QA", "generator": "model", **kwargs}


class MetricTests(unittest.TestCase):
    def test_perfect_four_type_overlap_retained_without_double_counting_any(self):
        labels = [span(0, 4), span(2, 6, 1), span(6, 8, 2), span(1, 2, 3)]
        result = evaluate_records([record(labels=labels)], [{"id": "one", "spans": labels}])
        self.assertEqual(result["character"]["any"]["gold_chars"], 8)
        self.assertEqual(result["character"]["micro_typed"]["gold_chars"], 11)
        self.assertEqual(result["character"]["macro_f1_types"], 1)
        self.assertEqual(result["span_exact"]["typed"]["tp"], 4)

    def test_partial_overlap_has_expected_micro_counts_and_not_exact_match(self):
        result = evaluate_records([record(labels=[span(0, 4)])], {"one": [span(2, 6)]})
        any_ = result["character"]["any"]
        self.assertEqual((any_["tp"], any_["fp"], any_["fn"]), (2, 2, 2))
        self.assertEqual(any_["f1"], .5)
        self.assertEqual(result["span_exact"]["typed"]["f1"], 0)
        self.assertEqual(result["character"]["macro_f1_types"], .125)

    def test_wrong_type_is_any_correct_but_typed_wrong(self):
        result = evaluate_records([record(labels=[span(0, 4)])], {"one": [span(0, 4, 1)]})
        self.assertEqual(result["character"]["any"]["f1"], 1)
        self.assertEqual(result["character"]["macro_f1_types"], 0)
        self.assertEqual(result["span_exact"]["any"]["f1"], 1)
        self.assertEqual(result["span_exact"]["typed"]["f1"], 0)

    def test_exact_duplicates_count_once_but_raw_support_is_reported(self):
        labels = [span(0, 4), span(0, 4), span(0, 4, 1)]
        result = evaluate_records([record(labels=labels)], {"one": labels * 2})
        self.assertEqual(result["raw_gold_spans"], 3)
        self.assertEqual(result["raw_predicted_spans"], 6)
        self.assertEqual(result["unique_gold_typed_spans"], 2)
        self.assertEqual(result["span_exact"]["any"]["gold_spans"], 1)
        self.assertEqual(result["span_exact"]["typed"]["f1"], 1)

    def test_union_prediction_does_not_exactly_match_separate_raw_annotations(self):
        result = evaluate_records([record(labels=[span(0, 3), span(2, 5)])], {"one": [span(0, 5)]})
        self.assertEqual(result["character"]["any"]["f1"], 1)
        self.assertEqual(result["span_exact"]["typed"]["gold_spans"], 2)
        self.assertEqual(result["span_exact"]["typed"]["f1"], 0)

    def test_empty_gold_and_prediction_have_zero_prf_and_absent_types_stay_in_macro(self):
        result = evaluate_records([record()], {"one": []})
        for key in ("any", *LABEL_TYPES):
            self.assertEqual(result["character"][key]["precision"], 0)
            self.assertEqual(result["character"][key]["recall"], 0)
            self.assertEqual(result["character"][key]["f1"], 0)
        self.assertEqual(result["span_exact"]["typed"]["f1"], 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluate_records([], [])

    def test_whitespace_is_included_without_normalizing_original_text(self):
        result = evaluate_records([record(text="A B", labels=[span(0, 3)])], {"one": [span(0, 1), span(2, 3)]})
        self.assertEqual(result["character"]["any"]["gold_chars"], 3)
        self.assertAlmostEqual(result["character"]["any"]["recall"], 2 / 3)
        self.assertAlmostEqual(result["character"]["any"]["f1"], .8)

    def test_offsets_count_unicode_codepoints_not_bytes(self):
        result = evaluate_records([record(text="İ😊 x", labels=[span(0, 2)])], {"one": [span(1, 2)]})
        self.assertEqual(result["character"]["any"]["gold_chars"], 2)
        self.assertAlmostEqual(result["character"]["any"]["f1"], 2 / 3)

    def test_invalid_spans_and_gold_text_are_rejected(self):
        bad = [span(-1, 2), span(2, 2), span(4, 2), span(0, 9), span(False, 2), span(0, 1.5),
               dict(span(0, 1), label_type="unknown")]
        for value in bad:
            with self.subTest(span=value), self.assertRaises(ValueError):
                evaluate_records([record()], {"one": [value]})
        with self.assertRaisesRegex(ValueError, "text differs"):
            evaluate_records([record(labels=[dict(span(0, 2), text="WRONG")])], {"one": []})
        with self.assertRaisesRegex(ValueError, "prediction spans must contain exactly"):
            evaluate_records([record()], {"one": [dict(span(0, 2), meta="annotation leakage")]})

    def test_missing_extra_duplicate_and_nonstring_ids_are_rejected(self):
        for predictions in ({}, {"extra": []}, [{"id": "one", "spans": []}] * 2, {1: []}):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate_records([record()], predictions)
        with self.assertRaisesRegex(ValueError, "duplicate gold"):
            evaluate_records([record(), record()], {"one": []})
        with self.assertRaisesRegex(ValueError, "exactly id and spans"):
            evaluate_records([record()], [{"id": "one", "spans": [], "gold": []}])

    def test_breakdowns_share_the_exact_same_fixed_predictions(self):
        rows = [record("one", labels=[span(0, 2)]), record("two", labels=[span(0, 6)], task_type="Summary", generator="other")]
        result = evaluate_records(rows, {"one": [span(0, 2)], "two": []})
        self.assertEqual(result["breakdowns"]["task_type"]["QA"]["character"]["any"]["f1"], 1)
        self.assertEqual(result["breakdowns"]["generator"]["other"]["responses"], 1)
        self.assertAlmostEqual(result["character"]["any"]["f1"], .4)

    def test_random_interval_parity_with_existing_crf_character_helper(self):
        rng = random.Random(42)
        gold, predicted = {}, {}
        for index in range(40):
            gold[str(index)], predicted[str(index)] = [], []
            for destination in (gold[str(index)], predicted[str(index)]):
                for _ in range(rng.randrange(10)):
                    start = rng.randrange(20)
                    destination.append(span(start, rng.randrange(start + 1, 22), rng.randrange(4)))
        actual, expected = character_scores(gold, predicted), legacy_character_scores(gold, predicted)
        for key in ("any", *LABEL_TYPES):
            for field in ("precision", "recall", "f1", "gold_chars"):
                self.assertAlmostEqual(actual[key][field], expected[key][field])
        self.assertAlmostEqual(actual["macro_f1_types"], expected["macro_f1_types"])


class EvaluationFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gold = self.root / "validation.jsonl"
        self.predictions = self.root / "predictions.jsonl"
        self.manifest = self.root / "manifest.csv"
        self.summary = self.root / "summary.json"
        self.rows = [record(labels=[span(0, 2)], split="validation", source_id="val", quality="good"),
                     record("excluded", split="validation", source_id="val", quality="truncated")]
        write_jsonl(self.gold, self.rows)
        write_jsonl(self.predictions, [{"id": "one", "spans": [span(0, 2)]}])
        self.manifest.write_text("source_id,group_id,split\nval,val-group,validation\ntrain,train-group,train\n", encoding="utf-8")
        self.summary.write_text(json.dumps({"processed_sha256": {"validation.jsonl": sha256_file(self.gold)}}), encoding="utf-8")

    def test_quality_exclusion_manifest_and_summary_checks(self):
        rows, metadata = load_gold(self.gold, "validation", manifest=self.manifest, summary=self.summary)
        self.assertEqual(len(rows), 1)
        self.assertEqual(metadata["excluded_by_quality"], {"truncated": 1})
        self.gold.write_text(self.gold.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_gold(self.gold, "validation", manifest=self.manifest, summary=self.summary)

    def test_manifest_group_leakage_is_rejected_before_quality_filter(self):
        self.manifest.write_text("source_id,group_id,split\nval,same,validation\ntrain,same,train\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            load_gold(self.gold, "validation", manifest=self.manifest)

    def test_frozen_test_can_use_unchanged_train_validation_summary(self):
        # Synthetic tempfile only: no official test annotations are opened.
        test_file = self.root / "test.jsonl"
        write_jsonl(test_file, [record(split="test", source_id="test-src", quality="good")])
        self.manifest.write_text("source_id,group_id,split\ntest-src,test-group,test\n", encoding="utf-8")
        before = self.summary.read_bytes()
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_gold(test_file, "test", manifest=self.manifest, summary=self.summary)
        rows, metadata = load_gold(test_file, "test", manifest=self.manifest, summary=self.summary,
                                   frozen_test_hash_verified=True)
        self.assertEqual(len(rows), 1)
        self.assertFalse(metadata["summary_covers_split"])
        self.assertEqual(metadata["gold_identity_source"], "frozen_test_job")
        self.assertEqual(self.summary.read_bytes(), before)
        self.assertEqual(metadata["sha256"]["gold"], sha256_file(test_file))

    def test_frozen_test_flag_never_bypasses_wrong_test_or_missing_validation_hash(self):
        test_file = self.root / "test.jsonl"
        write_jsonl(test_file, [record(split="test", source_id="test-src", quality="good")])
        self.summary.write_text(json.dumps({"processed_sha256": {"test.jsonl": "wrong"}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_gold(test_file, "test", summary=self.summary, frozen_test_hash_verified=True)
        self.summary.write_text(json.dumps({"processed_sha256": {"test.jsonl": None}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_gold(test_file, "test", summary=self.summary, frozen_test_hash_verified=True)
        self.summary.write_text(json.dumps({"processed_sha256": {}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_gold(self.gold, "validation", summary=self.summary, frozen_test_hash_verified=True)

    def test_duplicate_file_ids_including_excluded_records_are_rejected(self):
        write_jsonl(self.gold, self.rows + [dict(self.rows[1])])
        with self.assertRaisesRegex(ValueError, "duplicate gold"):
            load_gold(self.gold, "validation")
        write_jsonl(self.predictions, [{"id": "one", "spans": []}] * 2)
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            read_predictions(self.predictions)

    def test_cli_writes_common_json_markdown_csv_and_refuses_overwrite(self):
        output, report, csv = [self.root / name for name in ("comparison.json", "comparison.md", "comparison.csv")]
        args = ["--split", "validation", "--gold", str(self.gold), "--manifest", str(self.manifest),
                "--split-summary", str(self.summary), "--predictions", "tiny=" + str(self.predictions),
                "--output", str(output), "--report", str(report), "--csv", str(csv)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args), 0)
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["models"]["tiny"]["metrics"]["character"]["any"]["f1"], 1)
        self.assertIn("Project metrics", report.read_text(encoding="utf-8"))
        self.assertIn("char_any_f1", csv.read_text(encoding="utf-8"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            cli.main(args)
        self.assertEqual(caught.exception.code, 1)

    def test_test_gate_runs_before_opening_official_gold_or_predictions(self):
        args = ["--split", "test", "--predictions", "tiny=missing.jsonl", "--output", str(self.root / "output.json")]
        with patch.object(cli, "require_frozen_test_job", side_effect=ValueError("frozen protocol required")) as gate, \
             patch.object(cli, "load_gold", side_effect=AssertionError("opened gold before gate")), \
             redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.main(args)
        self.assertEqual(gate.call_args.kwargs["stage"], "evaluate")

    def test_frozen_evaluation_needs_exact_declared_code_and_prediction_bytes(self):
        names = list(cli.METRIC_FILES) + ["gold.jsonl", "manifest.csv", "summary.json", "prediction.jsonl"]
        for name in names:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("synthetic fixture", encoding="utf-8")
        inputs = {name: sha256_file(self.root / name) for name in names}
        args = SimpleNamespace(invocation=["evaluate.py"], gold=self.root / "gold.jsonl", manifest=self.root / "manifest.csv",
                               split_summary=self.root / "summary.json", predictions=[("tiny", self.root / "prediction.jsonl")])
        with patch.object(cli, "ROOT", self.root), patch.object(cli, "require_frozen_test_job", return_value=inputs):
            cli._test_gate(args)
            del inputs["ragtruth_eval/metrics.py"]
            with self.assertRaisesRegex(ValueError, "frozen evaluation job"):
                cli._test_gate(args)
            inputs["ragtruth_eval/metrics.py"] = sha256_file(self.root / "ragtruth_eval/metrics.py")
            (self.root / "prediction.jsonl").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input changed"):
                cli._test_gate(args)


if __name__ == "__main__":
    unittest.main()
