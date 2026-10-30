"""Integration checks for failure records, exclusive runs and the test gate."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_job.py"
SPEC = importlib.util.spec_from_file_location("ragtruth_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RunJobTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.config_path = self.root / "job.json"
        self.config = {
            "schema_version": 1, "name": "fixture", "stage": "integration_check",
            "split": "synthetic", "seed": 42, "inputs": [],
            "command": ["{python}", "-c", "print('fixture')"],
            "expected_outputs": [], "timeout_seconds": 10,
        }
        self.git("init", "--initial-branch=main")
        self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.autocrlf", "false")
        (self.root / ".gitignore").write_text("outputs/\n", encoding="utf-8")
        self.write_config()
        self.commit()

    def git(self, *args):
        return RUNNER.git(self.root, *args)

    def commit(self):
        self.git("add", ".")
        self.git("commit", "-m", "Fixture")

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def freeze(self, inputs=None):
        self.config.update(stage="predict", split="test", inputs=inputs or ["manifest.json"],
                           frozen_protocol="protocol.json")
        (self.root / "manifest.json").write_text('{"fixture": true}', encoding="utf-8")
        self.write_config()
        frozen = {"status": "frozen", "test_config_sha256": RUNNER.sha256(self.config_path),
                  "inputs_sha256": {name: RUNNER.sha256(self.root / name) for name in self.config["inputs"]}}
        (self.root / "protocol.json").write_text(json.dumps(frozen), encoding="utf-8")
        self.commit()

    def test_failed_process_retains_log_and_exit_code(self):
        self.config["command"] = ["{python}", "-c", "print('failure evidence'); raise SystemExit(7)"]
        self.write_config()
        code, record = RUNNER.run_job(self.config_path, root=self.root)
        result = json.loads(record.read_text())
        self.assertEqual(code, 7)
        self.assertEqual(result["status"], "failed")
        self.assertIn("failure evidence", (record.parent / "console.log").read_text())
        self.assertFalse((self.root / "outputs/.run.lock").exists())

    def test_missing_output_does_not_count_as_success(self):
        self.config["expected_outputs"] = ["predictions.jsonl"]
        self.write_config()
        code, record = RUNNER.run_job(self.config_path, root=self.root)
        self.assertNotEqual(code, 0)
        self.assertIn("Missing expected output", json.loads(record.read_text())["error"])

    def test_existing_lock_is_preserved_and_child_does_not_run(self):
        output = self.root / "outputs"
        output.mkdir()
        lock = output / ".run.lock"
        lock.write_text("other run", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            RUNNER.run_job(self.config_path, root=self.root)
        self.assertEqual(lock.read_text(), "other run")
        self.assertFalse((output / "runs").exists())

    def test_pilot_cannot_be_assigned_official_test(self):
        self.config.update(stage="pilot", split="test")
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Stage/split"):
            RUNNER.validate(self.root, self.config_path)

    def test_frozen_protocol_accepts_only_matching_input_bytes(self):
        self.freeze()
        _, _, _, protocol = RUNNER.validate(self.root, self.config_path)
        self.assertEqual(protocol["commit"], self.git("rev-parse", "HEAD").decode().strip())
        # Even a new clean commit cannot silently change frozen input bytes.
        (self.root / "manifest.json").write_text('{"changed": true}', encoding="utf-8")
        self.commit()
        with self.assertRaisesRegex(ValueError, "input bytes"):
            RUNNER.validate(self.root, self.config_path)

    def test_test_rejects_uncommitted_or_draft_protocol(self):
        self.freeze()
        path = self.root / "protocol.json"
        frozen = json.loads(path.read_text())
        frozen["status"] = "draft"
        path.write_text(json.dumps(frozen), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "clean committed"):
            RUNNER.validate(self.root, self.config_path)
        self.commit()
        with self.assertRaisesRegex(ValueError, "explicitly frozen"):
            RUNNER.validate(self.root, self.config_path)

    def test_inputs_cannot_escape_repository(self):
        self.config["inputs"] = ["../outside.json"]
        self.write_config()
        with self.assertRaisesRegex(ValueError, "leaves the repository"):
            RUNNER.validate(self.root, self.config_path)

    def run_frozen_fixture(self, mismatch=False):
        # Only synthetic temporary files are involved, never the official test.
        invocation = "[sys.argv[0], '--unexpected']" if mismatch else "sys.argv"
        source = ("import sys\nfrom pathlib import Path\n"
                  f"sys.path.insert(0, {str(SCRIPT.parent)!r})\n"
                  "from run_job import require_frozen_test_job\n"
                  f"require_frozen_test_job(Path.cwd(), {invocation})\n"
                  "print('frozen fixture verified')\n")
        (self.root / "predictor.py").write_text(source, encoding="utf-8")
        self.config["command"] = ["{python}", "predictor.py"]
        self.freeze(inputs=["manifest.json", "predictor.py"])
        return RUNNER.run_job(self.config_path, root=self.root)

    def test_frozen_child_receives_record_and_rechecks_exact_command(self):
        code, record = self.run_frozen_fixture()
        self.assertEqual(code, 0)
        self.assertIn("frozen fixture verified", (record.parent / "console.log").read_text())
        self.assertEqual(json.loads(record.read_text())["status"], "completed")

    def test_frozen_child_rejects_a_different_invocation(self):
        code, record = self.run_frozen_fixture(mismatch=True)
        self.assertNotEqual(code, 0)
        self.assertIn("invocation differs", (record.parent / "console.log").read_text())


if __name__ == "__main__":
    unittest.main()
