"""Checks protecting the initial annotation audit and official test boundary."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inspect_training_data.py"
SPEC = importlib.util.spec_from_file_location("ragtruth_inspection", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class InspectionTests(unittest.TestCase):
    def run_fixture(self, responses):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            sources = [
                {"source_id": source_id, "task_type": "QA", "prompt": "Source context"}
                for source_id in ("a", "b")
            ]
            for name, rows in (("source_info.jsonl", sources), ("response.jsonl", responses)):
                (raw / name).write_text(
                    "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
                )
            (raw / "LICENSE").write_text("fixture", encoding="utf-8")
            return AUDIT.inspect(raw)

    @staticmethod
    def training(labels=None):
        return {
            "id": "1", "source_id": "a", "split": "train",
            "response": "Wrong date", "model": "fixture", "quality": "good",
            "labels": labels if labels is not None else [
                {"start": 0, "end": 5, "text": "Wrong", "label_type": "Evident Conflict"}
            ],
        }

    def test_official_test_text_and_labels_are_not_required_or_inspected(self):
        # A malformed labels value would fail if test labels were processed.
        test_row = {"id": "2", "source_id": "b", "split": "test", "labels": None}
        report = self.run_fixture([self.training(), test_row])
        self.assertEqual(report["counts"]["train_spans"], 1)
        self.assertEqual(report["counts"]["test_responses"], 1)
        self.assertEqual(report["train_invalid_span_count"], 0)

    def test_group_leakage_is_reported(self):
        test_row = {"id": "2", "source_id": "a", "split": "test"}
        report = self.run_fixture([self.training(), test_row])
        self.assertEqual(report["source_groups_shared_between_official_train_test"], ["a"])

    def test_out_of_range_and_mismatched_spans_are_reported(self):
        labels = [
            {"start": 0, "end": 999, "text": "Wrong", "label_type": "Evident Conflict"},
            {"start": 0, "end": 5, "text": "Other", "label_type": "Evident Conflict"},
        ]
        report = self.run_fixture([self.training(labels)])
        self.assertEqual(report["train_invalid_span_count"], 2)

    def test_overlapping_types_are_preserved_in_counts(self):
        labels = [
            {"start": 0, "end": 5, "text": "Wrong", "label_type": "Evident Conflict"},
            {"start": 0, "end": 10, "text": "Wrong date", "label_type": "Subtle Conflict"},
        ]
        report = self.run_fixture([self.training(labels)])
        self.assertEqual(report["counts"]["train_spans"], 2)
        self.assertEqual(report["counts"]["train_responses_with_overlapping_spans"], 1)
        self.assertEqual(report["train_span_types"]["Subtle Conflict"], 1)


if __name__ == "__main__":
    unittest.main()
