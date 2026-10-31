"""Hash-only final protocol generation on fake files in disposable git repos."""
from contextlib import redirect_stderr
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import generate_final_jobs as generator
from scripts import run_job as runner
from ragtruth_data.data import sha256_file


class FinalJobGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.git("init", "--initial-branch=main")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.autocrlf", "false")
        code = set(generator.COMMON_FILES) | set(generator.CRF_CODE_FILES) | set(generator.TRANSFORMER_CODE_FILES)
        code.update(generator.METRIC_FILES)
        for name in code:
            self.file(name, b"synthetic code; never executed")
        # Deliberately invalid JSON/UTF8: generation may hash, never parse test data.
        self.file(generator.TEST_GOLD, b"\xff\x00not JSON: fake gold")
        self.file(generator.TEST_CRF, b"\xfe\x00not JSON: fake CRF view")
        self.file("data/processed/train.jsonl", b"fake training bytes")
        models, artifacts = {}, {}
        for name in generator.MODEL_NAMES[:2]:
            directory = "artifacts/" + name
            models[name] = {"model_dir": directory}
            for filename in generator.CRF_MODEL_FILES:
                relative = directory + "/" + filename
                artifacts[relative] = self.file(relative)
        checkpoint = "artifacts/transformer/epoch-2"
        models["transformer"] = {"checkpoint": checkpoint, "calibration": "artifacts/calibration.json"}
        for filename in (*generator.TRANSFORMER_MINIMUM_FILES, "encoder/model.safetensors", "tokenizer/added_tokens.json"):
            relative = checkpoint + "/" + filename
            artifacts[relative] = self.file(relative)
        artifacts["artifacts/calibration.json"] = self.file("artifacts/calibration.json")
        self.decision = {"status": "frozen", "seed": 42, "models": models,
            "artifacts_sha256": artifacts, "code_sha256": {name: sha256_file(self.root / name) for name in sorted(code)},
            "data_sha256": {"data/processed/train.jsonl": sha256_file(self.root / "data/processed/train.jsonl")}}
        self.decision_path = self.root / "docs/FINAL_PROTOCOL.json"
        self.write_decision()
        self.commit()

    def git(self, *args):
        return runner.git(self.root, *args)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-m", "Fixture")

    def file(self, name, content=b"synthetic artifact; not model weights"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return sha256_file(path)

    def write_decision(self):
        self.decision_path.parent.mkdir(parents=True, exist_ok=True)
        self.decision_path.write_text(json.dumps(self.decision, indent=2) + "\n", encoding="utf-8")

    def test_three_prediction_pairs_bind_exact_bytes_without_parsing_fake_gold(self):
        pairs = generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)
        self.assertEqual(len(pairs), 3)
        self.assertEqual([Path(pair["job"]).name for pair in pairs],
                         ["predict_" + name + ".json" for name in generator.MODEL_NAMES])
        for name, pair in zip(generator.MODEL_NAMES, pairs):
            config_path, protocol_path = self.root / pair["job"], self.root / pair["protocol"]
            config = json.loads(config_path.read_text())
            protocol = json.loads(protocol_path.read_text())
            self.assertEqual(config["stage"], "predict")
            self.assertEqual(config["split"], "test")
            self.assertEqual(config["frozen_protocol"], pair["protocol"])
            self.assertEqual(protocol["test_config_sha256"], sha256_file(config_path))
            self.assertEqual(protocol["inputs_sha256"], {key: sha256_file(self.root / key) for key in config["inputs"]})
            self.assertIn("docs/FINAL_PROTOCOL.json", config["inputs"])
            self.assertNotIn(pair["job"], config["inputs"])
            self.assertNotIn(pair["protocol"], config["inputs"])
            self.assertNotIn("--limit", config["command"])
            if name == "transformer":
                self.assertIn("--calibration", config["command"])
                self.assertIn("scripts/calibrate_transformer.py", config["inputs"])
                self.assertIn("artifacts/transformer/epoch-2/tokenizer/added_tokens.json", config["inputs"])
            else:
                self.assertIn(generator.TEST_CRF, config["inputs"])
        # The runner's actual machine gate accepts the generated pairs after commit.
        self.commit()
        for pair in pairs:
            config, inputs, state, protocol = runner.validate(self.root, self.root / pair["job"])
            self.assertFalse(state["working_tree_status"])
            self.assertIsNotNone(protocol)

    def test_uncommitted_or_draft_scientific_decision_is_rejected(self):
        self.decision["seed"] = 43
        self.write_decision()
        with self.assertRaisesRegex(ValueError, "committed HEAD bytes"):
            generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)
        self.decision["status"] = "draft"
        self.write_decision()
        self.commit()
        with self.assertRaisesRegex(ValueError, "must be frozen"):
            generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)

    def test_untracked_decision_has_no_frozen_authority(self):
        other = self.root / "docs/OTHER.json"
        other.write_bytes(self.decision_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "already be committed"):
            generator.generate_predictions("docs/OTHER.json", root=self.root)

    def test_changed_model_data_or_code_hash_stops_before_writes(self):
        for field in ("artifacts_sha256", "code_sha256", "data_sha256"):
            name = next(iter(self.decision[field]))
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "scientific input changed"):
                generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)
            self.assertFalse((self.root / "configs/final").exists())
            path.write_bytes(original)

    def test_incomplete_artifact_manifest_and_new_model_file_are_rejected(self):
        omitted = "artifacts/transformer/epoch-2/head.safetensors"
        del self.decision["artifacts_sha256"][omitted]
        self.write_decision()
        self.commit()
        with self.assertRaisesRegex(ValueError, "every selected"):
            generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)
        self.decision["artifacts_sha256"][omitted] = sha256_file(self.root / omitted)
        self.write_decision()
        self.commit()
        self.file("artifacts/crf_source/undeclared-extra.bin")
        with self.assertRaisesRegex(ValueError, "every selected"):
            generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)

    def test_missing_prepared_test_view_stops_without_partial_configs(self):
        (self.root / generator.TEST_CRF).unlink()
        with self.assertRaisesRegex(ValueError, "missing required input"):
            generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)
        self.assertFalse((self.root / "configs/final").exists())

    def test_relative_containment_rejects_output_or_model_traversal(self):
        for output in ("../outside", "configs/../final", str(self.root / "absolute"), "C:/outside"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "paths|repository"):
                generator.generate_predictions("docs/FINAL_PROTOCOL.json", output, root=self.root)
        self.decision["models"]["crf_source"]["model_dir"] = "../outside"
        self.write_decision()
        self.commit()
        with self.assertRaisesRegex(ValueError, "paths"):
            generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)

    def test_existing_job_is_never_overwritten_or_partially_supplemented(self):
        collision = self.root / "configs/final/predict_crf_response_only.json"
        collision.parent.mkdir(parents=True)
        collision.write_bytes(b"retain this existing job")
        with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
            generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)
        self.assertEqual(collision.read_bytes(), b"retain this existing job")
        self.assertFalse((self.root / "configs/final/predict_crf_source.json").exists())

    def test_generated_job_modification_is_rejected_by_real_runner_gate(self):
        pair = generator.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)[0]
        config_path = self.root / pair["job"]
        config = json.loads(config_path.read_text())
        config["timeout_seconds"] = 100
        config_path.write_text(json.dumps(config), encoding="utf-8")
        self.commit()
        with self.assertRaisesRegex(ValueError, "exact test config"):
            runner.validate(self.root, config_path)

    def test_evaluation_pair_requires_three_distinct_files_and_binds_metric_code(self):
        predictions = {}
        for name in generator.MODEL_NAMES:
            relative = "outputs/" + name + ".jsonl"
            self.file(relative, b"\xff\x00not parsed prediction data")
            predictions[name] = relative
        pairs = generator.generate_evaluation("docs/FINAL_PROTOCOL.json", predictions, root=self.root)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(Path(pairs[0]["job"]).name, "final_evaluate.json")
        config = json.loads((self.root / pairs[0]["job"]).read_text())
        self.assertEqual(config["stage"], "evaluate")
        self.assertEqual(config["command"].count("--predictions"), 3)
        self.assertEqual(set(generator.METRIC_FILES) - set(config["inputs"]), set())
        self.assertEqual(set(predictions.values()) - set(config["inputs"]), set())
        self.assertEqual(len(config["expected_outputs"]), 3)
        self.commit()
        runner.validate(self.root, self.root / pairs[0]["job"])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            generator.generate_evaluation("docs/FINAL_PROTOCOL.json", {"crf_source": predictions["crf_source"]}, root=self.root)
        duplicate = {name: predictions["crf_source"] for name in generator.MODEL_NAMES}
        with self.assertRaisesRegex(ValueError, "distinct"):
            generator.generate_evaluation("docs/FINAL_PROTOCOL.json", duplicate, "configs/other", root=self.root)

    def test_cli_duplicate_prediction_names_are_rejected_before_file_access(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            generator.main(["evaluate", "--decision", "missing.json", "--predictions", "crf_source=one",
                            "--predictions", "crf_source=two"])
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
