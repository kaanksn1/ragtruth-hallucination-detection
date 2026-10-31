"""Guard wrong cohorts/model selection and verify report percentages on synthetic aggregates."""
import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragtruth_data.schema import LABEL_TYPES
from ragtruth_eval import METRIC_VERSION
from scripts import build_technical_report as report


def scores(macro=.32, any_f1=.54):
    return {"any": {"precision": any_f1, "recall": any_f1, "f1": any_f1},
            **{kind: {"f1": macro} for kind in LABEL_TYPES}, "macro_f1_types": macro}


def model_metrics(count, macro=.32, any_f1=.54):
    return {"metric_version": METRIC_VERSION, "responses": count,
            "character": scores(macro, any_f1), "span_exact": {"typed": {"f1": .15}}}


def fixture(root, weighted_macro=.4, weighted_any=.6):
    paths = {name.replace("-", "_"): root / "outputs/reports" / (name + ".json") for name in report.ARGUMENTS}
    paths.update(base_calibration=root / "outputs/base-cal/calibration_report.json",
                 weighted_calibration=root / "outputs/weighted-cal/calibration_report.json",
                 weighted_summary=root / "outputs/weighted-training/transformer_summary.json",
                 decision=root / "docs/FINAL_PROTOCOL.json")
    identity = {"train.jsonl": "a" * 64, "validation.jsonl": "b" * 64,
                "split_summary": "c" * 64, "split_manifest": "d" * 64}
    grids = [[.005, .01, .02, .05, .1, .15, .2, .3, .4, .5, .6],
             [.001, .005, .01, .02, .05, .1, .15, .2, .3, .4, .5, .6, .7, .8, .9]]
    def calibration(family, macro, any_f1, grid):
        epochs = [1, 2] if family == "base" else [1]
        candidates = []
        for index, epoch in enumerate(epochs):
            metrics = scores(macro - .01 if index == 0 and len(epochs) > 1 else macro, any_f1)
            candidates.append({"candidate_index": index, "checkpoint": f"outputs/{family}-training/checkpoints/epoch-{epoch}",
                "checkpoint_sha256": ("e" if family == "base" else "f") * 64,
                "thresholds_by_type": {kind: .1 for kind in LABEL_TYPES}, "calibrated_metrics": metrics,
                "default_metrics": scores(.22444037046957488, .446423375445767)})
        chosen = candidates[-1]
        selected = {"status": "selected_on_validation", "validation_split": "validation", "selection_metric": "macro_f1_types",
            "label_types": list(LABEL_TYPES), "official_test_evaluated": False, "data_sha256": identity,
            "checkpoint": chosen["checkpoint"], "checkpoint_sha256": chosen["checkpoint_sha256"],
            "thresholds_by_type": chosen["thresholds_by_type"], "validation_metrics": chosen["calibrated_metrics"]}
        return {"config": {"split": "validation", "threshold_grid": grid, "checkpoints": [row["checkpoint"] for row in candidates]},
                "candidates": candidates, "selected": selected, "official_test_evaluated": False}
    base = calibration("base", .32, .54, grids[0])
    weighted = calibration("weighted", weighted_macro, weighted_any, grids[1])
    primary = weighted if (weighted_macro, weighted_any) > (.32, .54) else base
    primary_family = "weighted" if primary is weighted else "base"
    frozen = {"crf_source": {"model_dir": "outputs/crf/models/source"},
              "crf_response_only": {"model_dir": "outputs/crf/models/response_only"},
              "transformer": {"checkpoint": primary["selected"]["checkpoint"],
                              "calibration": f"outputs/{primary_family}-cal/calibration.json"}}
    decision = {"status": "frozen", "models": frozen, "artifacts_sha256": {
        frozen["transformer"]["checkpoint"] + "/checkpoint.json": primary["selected"]["checkpoint_sha256"],
        frozen["transformer"]["calibration"]: "0" * 64}}
    def comparison(split, count):
        return {"split": split, "metric_version": METRIC_VERSION, "data": {
            "quality_filter": "good", "included_responses": count, "file_responses": count + 10,
            "excluded_by_quality": {"truncated": 10}, "gold_identity_source": "frozen_test_job" if split == "test" else "split_summary"},
            "models": {name: {"metrics": model_metrics(count, primary["selected"]["validation_metrics"]["macro_f1_types"],
                primary["selected"]["validation_metrics"]["any"]["f1"])} for name in report.MODEL_NAMES}}
    positives = [31671, 1472, 64342, 20311]
    negatives = [2151637 - value for value in positives]
    vector = [math.sqrt(n / p) for p, n in zip(positives, negatives)]
    weighting = {"method": "sqrt_capped", "split": "train", "cap": 50, "label_types": list(LABEL_TYPES),
        "weights_vector": vector, "positive_response_tokens": dict(zip(LABEL_TYPES, positives)),
        "negative_response_tokens": dict(zip(LABEL_TYPES, negatives))}
    weighted_summary = {"status": "complete", "epochs": [{}, {}, {}], "limits": {"train": None, "validation": None},
        "config": {"epochs": 3, "seed": 42, "positive_weighting": "sqrt_capped", "positive_weighting_cap": 50,
                   "positive_weights": vector}, "positive_weighting": weighting, "data_sha256": identity}
    audit = {"status": "passed", "official_test_read": False, "rows_removed": 0,
        "source_id_overlap": {"count": 0}, "group_id_overlap": {"count": 0}, "response_id_overlap": {"count": 0},
        "source_text_overlap": {"exact": {"distinct_texts": 0}, "normalized": {"distinct_texts": 0}},
        "response_text_overlap": {"exact": {"train_responses": 83, "validation_responses": 20, "all_shared_gold_empty": True}},
        "counts": {"train_sources": 2138, "validation_sources": 377, "train_groups": 2136, "validation_groups": 377},
        "quality_selection": {"train": {"file_responses": 12828, "included_responses": 12702, "excluded_by_quality": {"truncated": 126}},
                              "validation": {"file_responses": 2262, "included_responses": 2240, "excluded_by_quality": {"truncated": 22}}}}
    training = {kind: {"stopped_at_max_iterations": True} for kind in LABEL_TYPES}
    crf = {"run_kind": "full_validation", "official_test_evaluated": False, "selected": {"name": "synthetic", "decoding": {"name": "posterior_0.1"}},
        "total_seconds": 1500, "config": {"training": {"max_iterations": 150}},
        "grid": [{"training": training}] * 3, "response_only_control": {"training": training}}
    support = {kind: {"raw_spans": 160 if index == 1 else 1000, "positive_word_fraction": .001} for index, kind in enumerate(LABEL_TYPES)}
    errors = {"split": "validation", "support": {"train": support, "validation": {kind: {"raw_spans": 25} for kind in LABEL_TYPES}}}
    inputs = {"validation": comparison("validation", 2240), "test": comparison("test", 2690), "decision": decision,
        "base_calibration": base, "weighted_calibration": weighted, "weighted_summary": weighted_summary,
        "development_audit": audit, "crf": crf, "error_analysis": errors}
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(inputs[name]), encoding="utf-8")
    return inputs, paths


class TechnicalReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.patch = patch.object(report, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_flat_type_metrics_are_rendered_as_percentages_and_primary_is_from_decision(self):
        inputs, paths = fixture(self.root)
        for index, kind in enumerate(LABEL_TYPES):
            inputs["test"]["models"]["transformer"]["metrics"]["character"][kind]["f1"] = (index + 1) / 10
        text = report.render_report(inputs, paths)
        self.assertIn("| transformer | 10.00 | 20.00 | 30.00 | 40.00 |", text)
        self.assertIn("The frozen primary `transformer` is **weighted**", text)
        self.assertIn("8.1815", text)
        self.assertIn("Original data, CRF and Transformer implementation authorship is preserved", text)

    def test_wrong_splits_metric_version_and_model_sets_are_rejected(self):
        inputs, paths = fixture(self.root)
        variants = []
        wrong = copy.deepcopy(inputs); wrong["test"]["split"] = "validation"; variants.append(wrong)
        wrong = copy.deepcopy(inputs); wrong["validation"]["metric_version"] = "other"; variants.append(wrong)
        wrong = copy.deepcopy(inputs); del wrong["test"]["models"]["crf_source"]; variants.append(wrong)
        wrong = copy.deepcopy(inputs); wrong["validation"]["models"]["undeclared"] = wrong["validation"]["models"]["transformer"]; variants.append(wrong)
        for wrong in variants:
            with self.subTest(), self.assertRaises(ValueError):
                report.validate_inputs(wrong, paths)
        inputs["validation"]["models"]["transformer_default"] = copy.deepcopy(inputs["validation"]["models"]["transformer"])
        report.validate_inputs(inputs, paths)

    def test_family_selection_uses_validation_and_retains_base_on_exact_tie(self):
        inputs, paths = fixture(self.root, weighted_macro=.32, weighted_any=.54)
        self.assertEqual(report.validate_inputs(inputs, paths)["family"], "unweighted")
        inputs["test"]["models"]["transformer"]["metrics"]["character"] = scores(.99, .99)
        self.assertEqual(report.validate_inputs(inputs, paths)["family"], "unweighted")
        inputs["decision"]["models"]["transformer"] = {
            "checkpoint": "outputs/weighted-training/checkpoints/epoch-1", "calibration": "outputs/weighted-cal/calibration.json"}
        with self.assertRaisesRegex(ValueError, "selection rule"):
            report.validate_inputs(inputs, paths)

    def test_cli_writes_new_report_without_following_record_paths_or_overwriting(self):
        inputs, paths = fixture(self.root)
        output = self.root / "results/FINAL_REPORT.md"
        args = [argument for name in report.ARGUMENTS for argument in ("--" + name, str(paths[name.replace("-", "_")]))]
        args += ["--output", str(output)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(report.main(args), 0)
        before = output.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            report.main(args)
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
