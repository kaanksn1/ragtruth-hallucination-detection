"""The selected per-type decoder must survive export and ordinary prediction."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts import baseline_crf as crf
from ragtruth_data.alignment import word_level_example


class SelectedDecoderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.record = {"id": "r", "response": "a b c", "source_text": "a source", "quality": "good", "labels": []}
        self.processed = self.root / "processed"
        crf.write_jsonl(self.processed / "validation.jsonl", [self.record])
        crf.write_jsonl(self.processed / "crf_validation.jsonl", [word_level_example(self.record)])
        self.identity = {"files_sha256": {p.name: crf.sha256_file(p) for p in self.processed.glob("*.jsonl")}}
        self.examples = crf.load_examples(self.processed, "validation", with_labels=False)
        self.run = {"probs": {"r": dict(zip(crf.LABEL_TYPES, ([0.3, 0, 0], [0, 0.7, 0], [0, 0, 0.9], [0, 0, 0])))}}
        self.thresholds = dict(zip(crf.LABEL_TYPES, (0.2, 0.8, 0.4, 0.5)))
        self.best = {"name": "setting", "setting": {"c1": 0.1, "c2": 1.0},
                     "decoding": {"name": "posterior_0.5", "threshold": 0.5}}
        self.source = self.root / "grid/setting"
        self.source.mkdir(parents=True)
        for kind in crf.LABEL_TYPES:
            (self.source / (crf.type_slug(kind) + ".crfsuite")).write_bytes(b"fixture")

    def test_joint_export_is_loadable_and_replays_selected_predictions(self):
        metrics = {"global": {"any": {"f1": 0.4}, "macro_f1_types": 0.2},
                   "joint": {"any": {"f1": 0.5}, "macro_f1_types": 0.2}}
        exported = crf.export_selected_configuration(self.root, self.examples, self.run, self.best,
            self.identity, True, 3, metrics, {"global": self.best["decoding"],
                "joint": {"name": "posterior_by_type", "thresholds_by_type": self.thresholds}})
        self.assertEqual(exported["winner"], "joint")
        self.assertEqual(self.best["decoding"], {"name": "posterior_0.5", "threshold": 0.5})
        selected = self.root / exported["model_dir"]
        metadata = json.loads((selected / "model_config.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["decoding"]["thresholds_by_type"], self.thresholds)
        args = SimpleNamespace(processed_dir=str(self.processed), split="validation", limit=None,
                               output=self.root / "replay.jsonl", model_dir=selected)
        with patch.object(crf, "open_taggers", return_value={}), patch.object(crf, "predict", return_value={"m": self.run}):
            crf.cmd_predict(args, None)
        saved = (self.root / exported["predictions"]).read_bytes()
        self.assertEqual(saved, args.output.read_bytes())
        spans = list(crf.read_jsonl(args.output))[0]["spans"]
        self.assertEqual([(s["start"], s["end"], s["label_type"]) for s in spans],
                         [(0, 1, crf.LABEL_TYPES[0]), (4, 5, crf.LABEL_TYPES[2])])

    def test_no_eligible_threshold_fails_instead_of_accepting_unsafe_maximum(self):
        run = copy.deepcopy(self.run)
        run["probs"]["r"][crf.LABEL_TYPES[0]] = [0.9, 0.9, 0.9]
        gold = {"r": []}
        for tune in (crf.tune_type_thresholds, crf.tune_joint_thresholds):
            with self.subTest(tune=tune.__name__), self.assertRaisesRegex(ValueError, "precision floor"):
                tune(self.examples, run, gold, [0.1, 0.2])

    def test_bad_grid_probabilities_and_nonfinite_scores_fail(self):
        gold = {"r": []}
        for grid in ([], [True], [0.2, 0.2], [float("nan")], [1.1]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                crf.tune_type_thresholds(self.examples, self.run, gold, grid)
        run = copy.deepcopy(self.run)
        run["probs"]["r"][crf.LABEL_TYPES[0]][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "probabilities"):
            crf.tune_joint_thresholds(self.examples, run, gold, [0.5])
        with self.assertRaisesRegex(ValueError, "finite F1"):
            crf.select_best_configuration({"bad": {"any": {"f1": float("nan")}, "macro_f1_types": 0}})

    def test_invalid_saved_type_thresholds_are_rejected(self):
        for thresholds in ({}, dict(self.thresholds, extra=0.5), dict(self.thresholds, **{crf.LABEL_TYPES[0]: True}),
                           dict(self.thresholds, **{crf.LABEL_TYPES[0]: float("inf")})):
            with self.subTest(thresholds=thresholds), self.assertRaises(ValueError):
                crf.decode(self.examples, self.run, {"name": "posterior_by_type", "thresholds_by_type": thresholds})


if __name__ == "__main__":
    unittest.main()
