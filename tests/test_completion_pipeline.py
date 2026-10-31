"""Supervisor guards with temporary files and mocked processes; no models/test data."""
import copy
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragtruth_data.data import sha256_file
from ragtruth_data.schema import LABEL_TYPES
from scripts import complete_technical_pipeline as pipeline


def artifact(macro=.3, any_=.5):
    return {"status": "selected_on_validation", "validation_split": "validation", "label_types": list(LABEL_TYPES),
            "data_sha256": {"train.jsonl": "frozen"}, "validation_metrics": {"macro_f1_types": macro, "any": {"f1": any_}}}


def weighted_summary():
    return {"status": "complete", "label_types": list(LABEL_TYPES),
            "config": {"epochs": 3, "positive_weighting": "sqrt_capped", "positive_weighting_cap": 50,
                       "positive_weights": [2., 3., 4., 5.]},
            "positive_weighting": {"method": "sqrt_capped", "split": "train", "label_types": list(LABEL_TYPES),
                                   "weights_vector": [2., 3., 4., 5.], "records": 12702, "response_tokens": 100},
            "counts": {"train": 12702, "validation": 2240}, "limits": {"train": None, "validation": None},
            "epochs": [{"epoch": i, "validation": {"loss": loss}, "train": {"answer_tokens": 100}}
                       for i, loss in enumerate((.4, .3, .35), 1)]}


class CompletionSelectionTests(unittest.TestCase):
    def test_experiment_selection_macro_then_any_then_exact_tie_base(self):
        self.assertEqual(pipeline.choose_experiment(artifact(), artifact()), "base")
        self.assertEqual(pipeline.choose_experiment(artifact(.3, .9), artifact(.31, .1)), "weighted")
        self.assertEqual(pipeline.choose_experiment(artifact(.3, .5), artifact(.3, .51)), "weighted")
        self.assertEqual(pipeline.choose_experiment(artifact(.4, .2), artifact(.3, .9)), "base")

    def test_calibration_identity_order_and_finite_metric_guards(self):
        for change in ({"data_sha256": {"other": "data"}}, {"label_types": list(reversed(LABEL_TYPES))},
                       {"validation_split": "test"}, {"validation_metrics": {"macro_f1_types": float("nan"), "any": {"f1": .4}}}):
            candidate = dict(artifact(), **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                pipeline.choose_experiment(artifact(), candidate)

    def test_weighted_summary_requires_full_train_only_consistent_support(self):
        original = weighted_summary()
        pipeline.verify_weighted_summary(original)
        cases = []
        for key, value in (("counts", {"train": 8, "validation": 4}), ("limits", {"train": 8, "validation": None})):
            cases.append(dict(original, **{key: value}))
        changed = copy.deepcopy(original)
        changed["positive_weighting"]["split"] = "validation"
        cases.append(changed)
        changed = copy.deepcopy(original)
        changed["config"]["positive_weights"][1] = 9.
        cases.append(changed)
        changed = copy.deepcopy(original)
        changed["epochs"][2]["train"]["answer_tokens"] = 99
        cases.append(changed)
        changed = copy.deepcopy(original)
        changed["epochs"] = changed["epochs"][:2]
        cases.append(changed)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                pipeline.verify_weighted_summary(candidate)

    def test_saved_candidates_are_only_strict_improvements_in_epoch_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for epoch in (1, 2, 3):
                path = root / "checkpoints" / f"epoch-{epoch}"
                path.mkdir(parents=True)
                (path / "checkpoint.json").write_text("{}")
            result = pipeline.saved_improving_checkpoints(root, weighted_summary())
            self.assertEqual([p.name for p in result], ["epoch-1", "epoch-2"])
            (root / "checkpoints/epoch-2/checkpoint.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing saved improving"):
                pipeline.saved_improving_checkpoints(root, weighted_summary())
            tied = weighted_summary()
            tied["epochs"][1]["validation"]["loss"] = .4
            tied["epochs"][2]["validation"]["loss"] = .5
            self.assertEqual([p.name for p in pipeline.saved_improving_checkpoints(root, tied)], ["epoch-1"])

    def test_json_writes_have_literal_lf_and_refuse_replacing_old_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decision.json"
            pipeline.write_new(path, {"note": "İstanbul", "status": "frozen"})
            self.assertNotIn(b"\r", path.read_bytes())
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            with self.assertRaises(FileExistsError):
                pipeline.write_new(path, {})
            state = path.with_name("state.json")
            pipeline.write_state(state, {"status": "waiting"})
            pipeline.write_state(state, {"status": "done"})
            self.assertNotIn(b"\r", state.read_bytes())
            self.assertEqual(json.loads(state.read_text())["status"], "done")


class CompletionPathTests(unittest.TestCase):
    def test_windows_unicode_space_paths_convert_without_shell_escaping(self):
        root = PureWindowsPath("C:/Users/serda/My Project/Masaüstü")
        linux = "/mnt/c/Users/serda/My Project/Masaüstü"
        self.assertEqual(pipeline.windows_to_wsl(root), linux)
        mapped = pipeline.wsl_record_to_windows(linux + "/outputs/runs/one/run.json", root)
        self.assertEqual(mapped, root / "outputs/runs/one/run.json")

    def test_record_conversion_rejects_other_roots_traversal_and_nonrecord_files(self):
        root = "C:/Users/serda/repo"
        for bad in ("/mnt/c/Users/serda/repo-evil/outputs/runs/one/run.json",
                    "/mnt/c/Users/other/repo/outputs/runs/one/run.json",
                    "/mnt/c/Users/serda/repo/outputs/../runs/one/run.json",
                    "/mnt/c/Users/serda/repo/outputs/runs/one/not-a-record.json"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                pipeline.wsl_record_to_windows(bad, root)
        for bad in ("relative/path", "C:/Users/../outside", "//server/share/repo"):
            with self.subTest(path=bad), self.assertRaises(ValueError):
                pipeline.windows_to_wsl(bad)


class CompletionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.input = self.root / "input.txt"
        self.input.write_bytes(b"fixed input")
        self.record = {"status": "completed", "exit_code": 0, "inputs_sha256": {"input.txt": sha256_file(self.input)}}

    def test_changed_inputs_failed_job_and_escaped_paths_are_rejected(self):
        pipeline.verify_run_inputs(self.record, root=self.root)
        self.input.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            pipeline.verify_run_inputs(self.record, root=self.root)
        with self.assertRaisesRegex(ValueError, "complete successfully"):
            pipeline.verify_run_inputs(dict(self.record, status="failed", exit_code=1), root=self.root)
        with self.assertRaisesRegex(ValueError, "contained"):
            pipeline.verify_run_inputs(dict(self.record, inputs_sha256={"../outside": "bad"}), root=self.root)

    def test_only_unreadable_hf_aliases_use_hash_only_wsl_fallback(self):
        key = ".cache/hf-hub/snapshots/config.json"
        digest = "a" * 64
        record = dict(self.record, inputs_sha256={key: digest})
        with patch.object(pipeline, "sha256_file", side_effect=OSError("WinError1920")), \
             patch.object(pipeline, "unreadable_cache_hashes", return_value={key: digest}) as fallback:
            pipeline.verify_run_inputs(record, root=self.root)
        fallback.assert_called_once_with(self.root, [key])
        with patch.object(pipeline, "sha256_file", side_effect=OSError("WinError1920")), \
             patch.object(pipeline, "unreadable_cache_hashes", return_value={key: "b" * 64}):
            with self.assertRaisesRegex(ValueError, "input changed"):
                pipeline.verify_run_inputs(record, root=self.root)
        with patch.object(pipeline, "sha256_file", side_effect=OSError("ordinary file missing")), \
             patch.object(pipeline, "unreadable_cache_hashes", side_effect=AssertionError("broad fallback")):
            with self.assertRaises(OSError):
                pipeline.verify_run_inputs(self.record, root=self.root)

    def test_wsl_cache_hash_process_uses_argv_and_exact_names_only(self):
        key, digest = ".cache/hf-hub/snapshot/config.json", "a" * 64
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps({key: digest}), stderr="")
        with patch.object(pipeline.subprocess, "run", return_value=result) as process:
            self.assertEqual(pipeline.unreadable_cache_hashes("C:/Users/serda/My Project", [key]), {key: digest})
        argv = process.call_args.args[0]
        self.assertIsInstance(argv, list)
        self.assertEqual(json.loads(argv[-1]), [key])
        self.assertEqual(argv[argv.index("--cd") + 1], "/mnt/c/Users/serda/My Project")
        with self.assertRaisesRegex(ValueError, "restricted"):
            pipeline.unreadable_cache_hashes("C:/repo", ["data/validation.jsonl"])

    def test_historical_outputs_allow_old_code_identity_but_reject_changed_output(self):
        output = self.root / "predictions.jsonl"
        output.write_bytes(b"fixed predictions")
        old = dict(self.record, inputs_sha256={"old-code.py": "historical"}, outputs_sha256={output.name: sha256_file(output)})
        pipeline.verify_run_outputs(self.root, old)
        output.write_bytes(b"changed predictions")
        with self.assertRaisesRegex(ValueError, "output checksum"):
            pipeline.verify_run_outputs(self.root, old)

    def test_native_job_parser_requires_success_and_contained_run_record(self):
        record_path = self.root / "outputs/runs/one/run.json"
        stdout = "unrelated startup text\n" + json.dumps({"status": "completed", "exit_code": 0, "record": str(record_path)})
        self.assertEqual(pipeline.parse_job_record(stdout, root=self.root), record_path)
        with self.assertRaisesRegex(ValueError, "reported failure"):
            pipeline.parse_job_record(json.dumps({"status": "failed", "exit_code": 1, "record": str(record_path)}), root=self.root)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            pipeline.parse_job_record(json.dumps({"status": "completed", "exit_code": 0, "record": str(self.input)}), root=self.root)

    def test_cpu_job_is_mocked_and_verifies_saved_input_and_output_hashes(self):
        (self.root / ".venv/Scripts").mkdir(parents=True)
        (self.root / ".venv/Scripts/python.exe").write_bytes(b"never executed")
        args = SimpleNamespace(output="outputs/completion.json", final_test=False)
        with patch.object(pipeline, "source_state", return_value={"working_tree_status": []}):
            worker = pipeline.Pipeline(args, root=self.root)
        run = self.root / "outputs/runs/synthetic"
        run.mkdir(parents=True)
        output = run / "predictions.jsonl"
        output.write_bytes(b"synthetic output")
        record = dict(self.record, outputs_sha256={"predictions.jsonl": sha256_file(output)})
        (run / "run.json").write_text(json.dumps(record), encoding="utf-8")
        config = worker.work / "job.json"
        config.write_text("{}")
        process_result = subprocess.CompletedProcess([], 0, stdout=json.dumps({"status": "completed", "exit_code": 0,
                                                                            "record": str(run / "run.json")}), stderr="")
        with patch.object(pipeline.subprocess, "run", return_value=process_result) as process:
            self.assertEqual(worker.job(config, "mocked_native_job"), run)
        self.assertIsInstance(process.call_args.args[0], list)
        self.assertNotIn("shell", process.call_args.kwargs)

    def test_scoped_commit_never_stages_unrelated_paths(self):
        path = self.root / "docs/FINAL_PROTOCOL.json"
        path.parent.mkdir()
        path.write_text("{}")
        def fake_git(root, *args):
            return b"docs/FINAL_PROTOCOL.json\n" if args[:2] == ("ls-files", "--others") else b""
        with patch.object(pipeline, "git", side_effect=fake_git) as git_call, \
             patch.object(pipeline, "source_state", return_value={"working_tree_status": []}):
            pipeline.commit_new_paths(self.root, [path], "Synthetic scoped commit")
        self.assertIn(unittest.mock.call(self.root, "add", "--", "docs/FINAL_PROTOCOL.json"), git_call.call_args_list)
        with patch.object(pipeline, "git", side_effect=lambda root, *args:
                b"docs/FINAL_PROTOCOL.json\nunrelated.py\n" if args[:2] == ("ls-files", "--others") else b""):
            with self.assertRaisesRegex(ValueError, "unrelated/unexpected"):
                pipeline.commit_new_paths(self.root, [path], "must not commit")


if __name__ == "__main__":
    unittest.main()
