"""Bounded threshold selection and separate-artifact inference regressions."""
import copy
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ragtruth_data.data import sha256_file, write_jsonl
from ragtruth_data.schema import LABEL_TYPES
from scripts import calibrate_transformer as calibration
from scripts import transformer as cli
import test_transformer as fixtures

if fixtures.HAVE_HF:
    import torch
    import transformer as tr


class CalibrationTests(unittest.TestCase):
    def test_rare_class_can_be_recovered_without_changing_weights_and_ties_are_conservative(self):
        rows = [{"id": "positive", "response": "a b", "labels": [{"start": 0, "end": 3, "label_type": LABEL_TYPES[1]}]},
                {"id": "negative", "response": "a", "labels": []}]
        cache = {"positive": [[0., 0., 0.], [.02, 0., .02], [0., 0., 0.], [0., 0., 0.]],
                 "negative": [[0.], [.006], [0.], [0.]]}
        thresholds, table = calibration.tune_thresholds(rows, cache, [.005, .01, .02, .5, .6])
        self.assertEqual(thresholds[LABEL_TYPES[1]], .02)
        self.assertEqual(thresholds[LABEL_TYPES[0]], .6)
        selected = next(row for row in table[LABEL_TYPES[1]] if row["threshold"] == .02)
        self.assertEqual(selected["f1"], 1)
        self.assertEqual(selected["gold_chars"], 3)  # Include the whitespace bridge.
        self.assertEqual(calibration.decode_cached(rows[0], cache["positive"], thresholds)["spans"], rows[0]["labels"])
        self.assertEqual(calibration.decode_cached(rows[0], cache["positive"], {t: .5 for t in LABEL_TYPES})["spans"], [])

    def test_invalid_grids_or_mismatched_cache_ids_are_rejected(self):
        for grid in ([], [.1, .1], [.5, .1], [0], [float("nan")], [True]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                calibration.validate_grid(grid)
        with self.assertRaisesRegex(ValueError, "IDs"):
            calibration.tune_thresholds([{"id": "x", "response": "", "labels": []}], {}, [.1])
        with self.assertRaisesRegex(ValueError, "scores"):
            calibration.tune_thresholds([{"id": "x", "response": "a", "labels": []}],
                                        {"x": [[float("nan")], [.1], [.1], [.1]]}, [.1])

    def test_artifact_is_bound_to_exact_checkpoint_and_training_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            saved = {"metadata": {"data_sha256": {"validation.jsonl": "hash"}}}
            (checkpoint / "checkpoint.json").write_text(json.dumps(saved), encoding="utf-8")
            artifact = self.artifact(checkpoint, saved)
            path = root / "calibration.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertEqual(calibration.load_calibration(path, checkpoint, saved), artifact)
            other = root / "other"
            other.mkdir()
            (other / "checkpoint.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different checkpoint"):
                calibration.load_calibration(path, other, saved)
            with self.assertRaisesRegex(ValueError, "identity"):
                calibration.load_calibration(path, checkpoint, {"metadata": {"data_sha256": {"validation.jsonl": "changed"}}})
            for change in ({"label_types": list(reversed(LABEL_TYPES))}, {"validation_split": "test"},
                           {"thresholds_by_type": {LABEL_TYPES[0]: .1}}):
                path.write_text(json.dumps(dict(artifact, **change)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    calibration.load_calibration(path, checkpoint, saved)

    @staticmethod
    def artifact(checkpoint, saved):
        return {"schema_version": 1, "status": "selected_on_validation", "validation_split": "validation",
                "label_types": list(LABEL_TYPES), "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
                "data_sha256": saved["metadata"]["data_sha256"], "threshold_grid": [.01, .5],
                "thresholds_by_type": {kind: .01 if kind == LABEL_TYPES[1] else .5 for kind in LABEL_TYPES}}

    def test_cache_checksum_response_identity_and_probabilities_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl.gz"
            record = {"id": "r", "response": "a"}
            row = {"id": "r", "response_sha256": calibration.text_sha256("a"), "scores_by_type": [[.1]] * 4}
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            identity = {"checkpoint": "fixture"}
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta_path.write_text(json.dumps({"identity": identity, "cache_sha256": sha256_file(path)}), encoding="utf-8")
            self.assertEqual(calibration.read_cache(path, identity, [record]), {"r": [[.1]] * 4})
            with self.assertRaisesRegex(ValueError, "text"):
                calibration.read_cache(path, identity, [dict(record, response="b")])
            with self.assertRaisesRegex(ValueError, "identity/checksum"):
                calibration.read_cache(path, {"checkpoint": "wrong"}, [record])

    def test_frozen_test_requires_the_calibration_artifact_and_decoder_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(processed_dir=root / "data", manifest=root / "manifest.csv",
                                   checkpoint=root / "checkpoint", calibration=root / "calibration.json")
            names = ["transformer.py", "scripts/transformer.py", "scripts/run_job.py", "ragtruth_data/data.py",
                     "ragtruth_data/features.py", "ragtruth_data/alignment.py", "ragtruth_data/schema.py",
                     "requirements-transformer.txt", "data/test.jsonl", "manifest.csv", "checkpoint/checkpoint.json",
                     "scripts/calibrate_transformer.py", "scripts/baseline_crf.py", "calibration.json"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            inputs = {name: sha256_file(root / name) for name in names}
            with patch.object(cli, "ROOT", root), patch.object(cli, "__file__", str(root / "scripts/transformer.py")):
                cli.verify_frozen_inputs(args, inputs)
                for omitted in ("calibration.json", "scripts/calibrate_transformer.py"):
                    bad = dict(inputs)
                    del bad[omitted]
                    with self.subTest(omitted=omitted), self.assertRaisesRegex(ValueError, "frozen test job"):
                        cli.verify_frozen_inputs(args, bad)


@unittest.skipUnless(fixtures.HAVE_HF, "install requirements-transformer.txt")
class CalibratedInferenceTests(unittest.TestCase):
    setUp = fixtures.DataContractTests.setUp

    def test_cpu_cli_applies_artifact_without_gold_or_checkpoint_mutation(self):
        torch.set_num_threads(1)
        with self.assertRaisesRegex(ValueError, "finite"):
            tr.decode_character_scores({"id": "invalid", "response": "a"}, [[float("nan")], [0.], [0.], [0.]], self.cfg)
        rows = [{key: value for key, value in row.items() if key != "labels"} for row in self.rows["validation"]]
        write_jsonl(self.path / "validation.jsonl", rows)
        summary_path = self.path / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["processed_sha256"]["validation.jsonl"] = sha256_file(self.path / "validation.jsonl")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        identity = cli.verify_data(self.cfg, ("train", "validation"))
        model = fixtures.tiny_model()
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.fill_(-10)
            model.classifier.bias[1] = math_logit(.02)
        checkpoint = self.path / "checkpoint"
        tr.save_checkpoint(checkpoint, model, fixtures.offline_tokenizer(), self.cfg, {"data_sha256": identity})
        saved = json.loads((checkpoint / "checkpoint.json").read_text())
        before = (checkpoint / "checkpoint.json").read_bytes()
        artifact_path = self.path / "calibration.json"
        artifact_path.write_text(json.dumps(CalibrationTests.artifact(checkpoint, saved)), encoding="utf-8")
        for name, extra in (("default", []), ("calibrated", ["--calibration", str(artifact_path)])):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["predict", "--checkpoint", str(checkpoint), "--split", "validation",
                    "--processed-dir", str(self.path), "--manifest", str(self.manifest), "--split-summary", str(summary_path),
                    "--device", "cpu", "--output", str(self.path / (name + ".jsonl")), *extra]), 0)
        defaults = [json.loads(line) for line in (self.path / "default.jsonl").read_text().splitlines()]
        calibrated = [json.loads(line) for line in (self.path / "calibrated.jsonl").read_text().splitlines()]
        self.assertTrue(all(not row["spans"] for row in defaults))
        self.assertTrue(all(row["spans"] and all(span["label_type"] == LABEL_TYPES[1] for span in row["spans"]) for row in calibrated))
        self.assertEqual((checkpoint / "checkpoint.json").read_bytes(), before)


def math_logit(value):
    import math
    return math.log(value / (1 - value))


if __name__ == "__main__":
    unittest.main()
