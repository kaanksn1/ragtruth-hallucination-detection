"""Regression checks for data/model integrity in the assigned CRF review."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stderr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("crf_integrity", ROOT / "scripts/baseline_crf.py")
CRF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CRF)
from ragtruth_data.alignment import word_level_example  # noqa: E402


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.processed = self.root / "processed"
        self.summary = self.root / "split_summary.json"
        self.records = {}
        for side in ("train", "validation"):
            text = "The shop opened in 1999."
            start = text.index("1999")
            row = {"id": side + "-id", "response": text, "source_text": "The shop opened in 2004.",
                   "quality": "good", "labels": [{"start": start, "end": start + 4,
                                                  "label_type": CRF.LABEL_TYPES[0]}]}
            self.records[side] = row
            CRF.write_jsonl(self.processed / f"{side}.jsonl", [row])
            CRF.write_jsonl(self.processed / f"crf_{side}.jsonl", [word_level_example(row)])
        self.update_summary()
        self.config = json.loads((ROOT / "configs/crf.json").read_text(encoding="utf-8"))
        self.config.update(processed_dir=str(self.processed), split_summary=str(self.summary))
        self.config_path = self.root / "crf.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.args = SimpleNamespace(config=self.config_path, output=self.root / "run",
                                    seed=42, limit_train=None, limit_validation=None)

    def update_summary(self):
        self.summary.write_text(json.dumps({"processed_sha256": {
            p.name: CRF.sha256_file(p) for p in self.processed.glob("*.jsonl")
        }}), encoding="utf-8")

    def test_changed_data_stops_before_model_fitting_or_output_creation(self):
        with (self.processed / "validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        trainer = Mock(side_effect=AssertionError("must not train changed data"))
        with self.assertRaisesRegex(ValueError, "differs from split summary"):
            CRF.cmd_train(self.args, SimpleNamespace(Trainer=trainer))
        trainer.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_duplicate_or_missing_crf_rows_are_rejected(self):
        path = self.processed / "crf_train.jsonl"
        row = word_level_example(self.records["train"])
        CRF.write_jsonl(path, [row, row])
        with self.assertRaisesRegex(ValueError, "duplicate CRF"):
            CRF.load_examples(self.processed, "train")
        CRF.write_jsonl(path, [])
        with self.assertRaisesRegex(ValueError, "missing CRF rows"):
            CRF.load_examples(self.processed, "train")

    def test_token_offsets_must_match_original_text(self):
        row = word_level_example(self.records["train"])
        row["offsets"][0] = (1, 3)
        CRF.write_jsonl(self.processed / "crf_train.jsonl", [row])
        with self.assertRaisesRegex(ValueError, "tokens/offsets"):
            CRF.load_examples(self.processed, "train")

    def test_inference_loader_needs_neither_gold_spans_nor_bio_targets(self):
        record = dict(self.records["validation"])
        record.pop("labels")
        crf_row = word_level_example(self.records["validation"])
        crf_row.pop("bio_by_type")
        CRF.write_jsonl(self.processed / "validation.jsonl", [record])
        CRF.write_jsonl(self.processed / "crf_validation.jsonl", [crf_row])
        example = CRF.load_examples(self.processed, "validation", with_labels=False)[0]
        self.assertNotIn("labels", example)
        self.assertNotIn("bio_by_type", example)
        self.assertEqual(example["tokens"], crf_row["tokens"])

    def test_existing_model_run_cannot_be_overwritten(self):
        (self.args.output / "models").mkdir(parents=True)
        with self.assertRaisesRegex(FileExistsError, "overwrite"):
            CRF.cmd_train(self.args, None)

    def test_overlapping_train_validation_ids_are_rejected(self):
        record = dict(self.records["validation"], id=self.records["train"]["id"])
        CRF.write_jsonl(self.processed / "validation.jsonl", [record])
        CRF.write_jsonl(self.processed / "crf_validation.jsonl", [word_level_example(record)])
        self.update_summary()
        with self.assertRaisesRegex(ValueError, "IDs overlap"):
            CRF.cmd_train(self.args, None)

    @unittest.skipUnless(importlib.util.find_spec("pycrfsuite"), "python-crfsuite not installed")
    def test_disabled_source_features_are_saved_and_control_is_not_duplicated(self):
        import pycrfsuite

        self.config["features"]["source_features"] = False
        self.config["training"]["max_iterations"] = 5
        self.config["grid"] = [{"c1": 0.0, "c2": 0.1}]
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        summary = CRF.cmd_train(self.args, pycrfsuite)
        metadata = json.loads((self.args.output / "models/source/model_config.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["use_source_features"])
        self.assertFalse(metadata["use_source_features"])
        self.assertNotIn("response_only_control", summary)
        self.assertFalse((self.args.output / "models/response_only").exists())


class PredictionIntegrityTests(unittest.TestCase):
    def test_nonascii_windows_paths_use_ascii_training_file_and_memory_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "Masaüstü"
            target.mkdir()

            def native_train(filename):
                self.assertTrue(filename.isascii())
                Path(filename).write_bytes(b"checkpoint fixture")

            fake = SimpleNamespace(Trainer=None, Tagger=lambda: SimpleNamespace(
                open=Mock(side_effect=AssertionError("must not pass non-ASCII filename to native loader")),
                open_inmemory=Mock()))
            with patch.object(CRF.sys, "platform", "win32"):
                for t in CRF.LABEL_TYPES:
                    CRF.train_to_file(SimpleNamespace(train=native_train), target / f"{CRF.type_slug(t)}.crfsuite")
                taggers = CRF.open_taggers(fake, target)
            for tagger in taggers.values():
                tagger.open.assert_not_called()
                tagger.open_inmemory.assert_called_once_with(b"checkpoint fixture")

    def test_silent_native_training_failure_is_reported_immediately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.crfsuite"
            with self.assertRaisesRegex(RuntimeError, "did not write a model"):
                CRF.train_to_file(SimpleNamespace(train=lambda filename: None), path)

    def test_missing_o_label_has_full_positive_marginals(self):
        tagger = SimpleNamespace(tag=Mock(return_value=["B"]), labels=lambda: ["B"],
                                 marginal=Mock(side_effect=AssertionError("O does not exist")))
        example = {"id": "x", "tokens": ["Wrong"], "source_text": "source"}
        result = CRF.predict(SimpleNamespace(ItemSequence=lambda x: x), [example],
                             {"m": {t: tagger for t in CRF.LABEL_TYPES}}, False, 3)
        for t in CRF.LABEL_TYPES:
            self.assertEqual(result["m"]["probs"]["x"][t], [1.0])
        tagger.marginal.assert_not_called()

    def test_selection_metric_rejects_missing_and_extra_predictions(self):
        for predicted in ({}, {"x": [], "extra": []}):
            with self.subTest(predicted=predicted):
                with self.assertRaisesRegex(ValueError, "IDs must exactly match"):
                    CRF.char_scores({"x": []}, predicted)

    def test_modified_model_is_rejected_before_opening_tagger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src, dst = root / "source", root / "model"
            src.mkdir()
            for t in CRF.LABEL_TYPES:
                (src / f"{CRF.type_slug(t)}.crfsuite").write_bytes(b"fixture checkpoint")
            chosen = {"setting": {"c1": .1, "c2": .1}, "decoding": {"name": "viterbi"}}
            CRF.save_models(src, dst, True, 3, chosen, {"files_sha256": {}})
            (dst / f"{CRF.type_slug(CRF.LABEL_TYPES[0])}.crfsuite").write_bytes(b"changed checkpoint")
            args = SimpleNamespace(model_dir=dst, processed_dir=root, split="validation",
                                    limit=None, output=root / "predictions.jsonl")
            with self.assertRaisesRegex(ValueError, "model checksum mismatch"):
                CRF.cmd_predict(args, None)
            self.assertFalse(args.output.exists())

    def test_direct_test_command_is_blocked_before_reading_models_or_data(self):
        args = SimpleNamespace(model_dir=Path("missing-model"), processed_dir="missing-data", split="test",
                                limit=None, output=Path("unused.jsonl"), invocation=[str(CRF.__file__), "predict"])
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "run_job and a frozen"):
                CRF.cmd_predict(args, None)

    def test_test_predictions_cannot_use_a_smoke_limit(self):
        args = SimpleNamespace(model_dir=Path("missing-model"), processed_dir="missing-data", split="test", limit=3)
        with self.assertRaisesRegex(ValueError, "cannot be limited"):
            CRF.cmd_predict(args, None)

    def test_invalid_config_values_are_rejected(self):
        base = json.loads((ROOT / "configs/crf.json").read_text(encoding="utf-8"))
        for field, value in (("quality_filter", "all"), ("schema_version", 2)):
            changed = dict(base, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                CRF.validate_config(changed)
        for key, value in (("max_iterations", 0), ("algorithm", "ap")):
            changed = copy.deepcopy(base)
            changed["training"][key] = value
            with self.subTest(training=key), self.assertRaises(ValueError):
                CRF.validate_config(changed)
        changed = copy.deepcopy(base)
        changed["decoding"]["posterior_thresholds"] = [1.5]
        with self.assertRaisesRegex(ValueError, "thresholds"):
            CRF.validate_config(changed)

    def test_zero_smoke_limit_cannot_accidentally_start_full_training(self):
        for limit in ("0", "-1"):
            with self.subTest(limit=limit), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                CRF.main(["train", "--output", "unused", "--limit-train", limit])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
