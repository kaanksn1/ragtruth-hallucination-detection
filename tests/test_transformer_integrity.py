"""Regression checks for person 4's central-run and checkpoint contracts."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ragtruth_data.data import sha256_file, write_jsonl
from scripts import transformer as cli
import test_transformer as fixtures

HAVE_HF = fixtures.HAVE_HF

if HAVE_HF:
    import transformer as tr


class TransformerIntegrityTests(unittest.TestCase):
    setUp = fixtures.DataContractTests.setUp

    def test_validation_rejects_replaced_data_even_with_a_matching_new_summary(self):
        if not HAVE_HF:
            self.skipTest("install requirements-transformer.txt")
        original_identity = cli.verify_data(self.cfg, ("train", "validation"))
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "checkpoint.json").write_text(json.dumps({
            "config": self.cfg, "metadata": {"data_sha256": original_identity},
        }), encoding="utf-8")
        rows = copy.deepcopy(self.rows["validation"])
        rows[0]["response"] = "Different validation answer."
        write_jsonl(self.path / "validation.jsonl", rows)
        summary = json.loads((self.path / "summary.json").read_text(encoding="utf-8"))
        summary["processed_sha256"]["validation.jsonl"] = sha256_file(self.path / "validation.jsonl")
        (self.path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        args = SimpleNamespace(split="validation", limit=None, checkpoint=checkpoint, device="cpu",
                               processed_dir=self.path, manifest=self.manifest,
                               split_summary=self.path / "summary.json", output=self.path / "predictions.jsonl")
        with patch.object(tr, "load_checkpoint", side_effect=AssertionError("weights loaded before data check")):
            with self.assertRaisesRegex(ValueError, "saved checkpoint.*data identity"):
                cli.cmd_predict(args)
        self.assertFalse(args.output.exists())

    def test_data_identity_uses_portable_roles_not_machine_paths(self):
        identity = cli.verify_data(self.cfg, ("validation",))
        self.assertEqual(set(identity), {"split_summary", "split_manifest", "validation.jsonl"})
        self.assertEqual(identity["validation.jsonl"], sha256_file(self.path / "validation.jsonl"))
        relocated = self.path / "Masaüstü"
        relocated.mkdir()
        for name in ("validation.jsonl", "manifest.csv", "summary.json"):
            shutil.copyfile(self.path / name, relocated / name)
        new = cli.verify_data(dict(self.cfg, processed_dir=str(relocated),
                                   split_manifest=str(relocated / "manifest.csv"),
                                   split_summary=str(relocated / "summary.json")), ("validation",))
        cli.verify_validation_identity({"metadata": {"data_sha256": identity}}, new)

    def test_frozen_test_must_declare_inference_code_before_loading_checkpoint(self):
        # Synthetic missing paths only: no official test data/weights are accessed.
        args = SimpleNamespace(split="test", limit=None, invocation=["predict"],
                               processed_dir=cli.ROOT / "data/processed", manifest=cli.ROOT / "manifest.csv",
                               checkpoint=cli.ROOT / "outputs/missing-checkpoint")
        with patch.object(cli, "require_frozen_test_job", return_value={}):
            with self.assertRaisesRegex(ValueError, "frozen test job.*code"):
                cli.cmd_predict(args)

    def test_frozen_input_contract_accepts_exact_files_and_rejects_changed_code(self):
        checkpoint = self.path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "checkpoint.json").write_text("{}", encoding="utf-8")
        args = SimpleNamespace(processed_dir=self.path, manifest=self.manifest, checkpoint=checkpoint)
        names = ["transformer.py", "scripts/transformer.py", "scripts/run_job.py", "ragtruth_data/data.py",
                 "ragtruth_data/features.py", "ragtruth_data/alignment.py", "ragtruth_data/schema.py",
                 "requirements-transformer.txt", "test.jsonl"]
        for name in names:
            path = self.path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic fixture", encoding="utf-8")
        paths = [self.path / name for name in names] + [self.manifest, checkpoint / "checkpoint.json"]
        # Path(__file__) belongs to the real CLI; replace only the test root/script location.
        with patch.object(cli, "ROOT", self.path), patch.object(cli, "__file__", str(self.path / "scripts/transformer.py")):
            inputs = {p.relative_to(self.path).as_posix(): sha256_file(p) for p in paths}
            cli.verify_frozen_inputs(args, inputs)
            omitted = dict(inputs)
            del omitted["transformer.py"]
            with patch.object(cli, "sha256_file", side_effect=AssertionError("read before membership check")):
                with self.assertRaisesRegex(ValueError, "frozen test job.*code"):
                    cli.verify_frozen_inputs(args, omitted)
            (self.path / "transformer.py").write_text("changed fixture", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unchanged input"):
                cli.verify_frozen_inputs(args, inputs)

    def test_official_test_cannot_be_limited(self):
        with self.assertRaisesRegex(ValueError, "cannot be limited"):
            cli.cmd_predict(SimpleNamespace(split="test", limit=1))

    @unittest.skipUnless(HAVE_HF, "install requirements-transformer.txt")
    def test_floating_revision_is_rejected(self):
        for revision in ("main", "", "latest"):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "revision"):
                tr.validate_config(dict(fixtures.config(), revision=revision))


if __name__ == "__main__":
    unittest.main()
