"""Write a technical handoff from fixed aggregate JSON artifacts only.

Never opens gold/prediction JSONL, a checkpoint, or official-test annotations.
Model choice comes from the frozen decision and validation selection reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data.schema import LABEL_TYPES
from ragtruth_eval import METRIC_VERSION

MODEL_NAMES = ("crf_source", "crf_response_only", "transformer")
OPTIONAL_REFERENCE = "transformer_reference"
VALIDATION_ABLATIONS = {"transformer_default", "transformer_base", "weighted_trial"}
ARGUMENTS = ("validation", "test", "decision", "base-calibration", "weighted-summary",
             "weighted-calibration", "crf", "error-analysis", "development-audit")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"aggregate must be a JSON object: {path}")
    return value


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ratio(value, context):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            context + " must be a finite ratio in [0,1]")
    return float(value)


def number(value, context):
    require(type(value) is int and value >= 0, context + " must be a nonnegative integer")
    return value


def relative(value):
    value = str(value).replace("\\", "/")
    require(value and not PureWindowsPath(value).drive and not PurePosixPath(value).is_absolute()
            and ".." not in PurePosixPath(value).parts, "expected a repository-relative artifact path")
    return PurePosixPath(value).as_posix().rstrip("/")


def repository_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT.resolve()), "calibration report must be inside the repository")
    return path.relative_to(ROOT.resolve()).as_posix()


def md(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def metric_values(metrics):
    c = metrics["character"]
    return [ratio(c["any"][name], "any " + name) for name in ("precision", "recall", "f1")] + [
        ratio(c["macro_f1_types"], "typed macro F1"), ratio(metrics["span_exact"]["typed"]["f1"], "exact typed F1")]


def validate_comparison(comparison, split, models, *, allow_ablations=False):
    require(comparison.get("split") == split, f"expected {split} comparison")
    require(comparison.get("metric_version") == METRIC_VERSION, "comparison metric version mismatch")
    actual = set(comparison.get("models", {}))
    require(set(models) <= actual and (actual - set(models) <= VALIDATION_ABLATIONS if allow_ablations else actual == set(models)),
            f"{split} model keys do not match frozen decision")
    data = comparison["data"]
    require(data.get("quality_filter") == "good", "comparison quality policy differs")
    included = number(data["included_responses"], "included responses")
    require(included > 0, "comparison cannot be empty")
    total = number(data["file_responses"], "file responses")
    excluded = sum(number(count, "excluded responses") for count in data["excluded_by_quality"].values())
    require(included + excluded == total, "quality cohort counts do not add up")
    for name, entry in comparison["models"].items():
        metrics = entry["metrics"]
        require(metrics["metric_version"] == METRIC_VERSION and metrics["responses"] == included,
                "model metric version/cohort count mismatch: " + name)
        metric_values(metrics)
        for kind in LABEL_TYPES:
            ratio(metrics["character"][kind]["f1"], kind + " F1")
    if split == "test":
        require(data.get("gold_identity_source") == "frozen_test_job", "test aggregate lacks frozen-job provenance")


def calibration(report, path, expected_grid, family):
    config, selected = report["config"], report["selected"]
    require(config.get("split") == "validation" and selected.get("validation_split") == "validation",
            family + " calibration must use validation only")
    require(report.get("official_test_evaluated") is False and selected.get("official_test_evaluated") is False,
            family + " calibration claims test inspection")
    require(selected.get("status") == "selected_on_validation" and selected.get("selection_metric") == "macro_f1_types",
            family + " calibration selection policy differs")
    require(selected.get("label_types") == list(LABEL_TYPES), "calibration label order mismatch")
    grid = config["threshold_grid"]
    require(isinstance(grid, list) and len(grid) == expected_grid and len(set(grid)) == expected_grid,
            family + " threshold grid size differs from declared bounded search")
    require(all(0 < ratio(value, "threshold") < 1 for value in grid), "thresholds must be in (0,1)")
    candidates = report["candidates"]
    require(1 <= len(candidates) <= 3 and [row["checkpoint"] for row in candidates] == config["checkpoints"],
            family + " checkpoint candidates differ from configuration")
    if family == "unweighted":
        require(len(candidates) == 2 and [PurePosixPath(row["checkpoint"]).name for row in candidates] == ["epoch-1", "epoch-2"],
                "unweighted search must retain declared epochs 1 and 2")
    require(set(selected["thresholds_by_type"]) == set(LABEL_TYPES), "missing calibrated type threshold")
    require(all(value in grid for value in selected["thresholds_by_type"].values()), "selected threshold is outside grid")
    winner = max(candidates, key=lambda row: (row["calibrated_metrics"]["macro_f1_types"],
                                            row["calibrated_metrics"]["any"]["f1"], -row["candidate_index"]))
    require(winner["checkpoint"] == selected["checkpoint"] and winner["thresholds_by_type"] == selected["thresholds_by_type"],
            family + " selected checkpoint/thresholds do not follow validation rule")
    require(winner["calibrated_metrics"] == selected["validation_metrics"], "selected calibration metrics mismatch")
    for key in ("macro_f1_types",):
        ratio(selected["validation_metrics"][key], family + " " + key)
    ratio(selected["validation_metrics"]["any"]["f1"], family + " any F1")
    pair = (relative(selected["checkpoint"]), repository_path(Path(path).parent / "calibration.json"))
    return selected, pair


def validate_inputs(inputs, paths):
    decision = inputs["decision"]
    require(decision.get("status") == "frozen", "scientific decision must be frozen")
    model_set = set(decision.get("models", {}))
    require(model_set in (set(MODEL_NAMES), set(MODEL_NAMES) | {OPTIONAL_REFERENCE}), "unexpected frozen model keys")
    models = MODEL_NAMES + ((OPTIONAL_REFERENCE,) if OPTIONAL_REFERENCE in model_set else ())
    validate_comparison(inputs["validation"], "validation", models, allow_ablations=True)
    validate_comparison(inputs["test"], "test", models)
    base, base_pair = calibration(inputs["base_calibration"], paths["base_calibration"], 11, "unweighted")
    weighted, weighted_pair = calibration(inputs["weighted_calibration"], paths["weighted_calibration"], 15, "weighted")
    require(base_pair != weighted_pair, "weighted trial duplicates unweighted artifacts")
    require(base["data_sha256"] == weighted["data_sha256"], "calibrations use different split identities")
    expected_family = "weighted" if (weighted["validation_metrics"]["macro_f1_types"], weighted["validation_metrics"]["any"]["f1"]) > (
        base["validation_metrics"]["macro_f1_types"], base["validation_metrics"]["any"]["f1"]) else "unweighted"
    entry = decision["models"]["transformer"]
    primary_pair = relative(entry["checkpoint"]), relative(entry["calibration"])
    require(primary_pair in (base_pair, weighted_pair), "frozen primary Transformer does not match supplied calibration artifacts")
    family = "unweighted" if primary_pair == base_pair else "weighted"
    require(family == expected_family, "frozen primary violates macro/any/unweighted-tie selection rule")
    selected = base if family == "unweighted" else weighted
    artifacts = decision.get("artifacts_sha256", {})
    require(artifacts.get(primary_pair[0] + "/checkpoint.json") == selected["checkpoint_sha256"],
            "primary checkpoint checksum differs from frozen decision")
    require(primary_pair[1] in artifacts, "primary calibration is not frozen")
    if OPTIONAL_REFERENCE in model_set:
        reference = decision["models"][OPTIONAL_REFERENCE]
        reference_pair = relative(reference["checkpoint"]), relative(reference["calibration"])
        require(reference_pair == (weighted_pair if family == "unweighted" else base_pair),
                "reference model must be the other supplied calibrated family")
    actual = inputs["validation"]["models"]["transformer"]["metrics"]["character"]
    require(math.isclose(actual["macro_f1_types"], selected["validation_metrics"]["macro_f1_types"], abs_tol=1e-12)
            and math.isclose(actual["any"]["f1"], selected["validation_metrics"]["any"]["f1"], abs_tol=1e-12),
            "validation primary metrics do not match frozen calibrated model")

    summary = inputs["weighted_summary"]
    cfg, weighting = summary["config"], summary["positive_weighting"]
    require(summary.get("status") == "complete" and len(summary["epochs"]) == cfg["epochs"] == 3,
            "weighted trial must complete exactly three epochs")
    require(summary["limits"] == {"train": None, "validation": None} and cfg["seed"] == 42,
            "weighted trial limits/seed differ")
    require(cfg.get("positive_weighting") == weighting.get("method") == "sqrt_capped"
            and cfg.get("positive_weighting_cap") == weighting.get("cap") == 50 and weighting.get("split") == "train",
            "weighted loss must use train-only sqrt weights capped at 50")
    require(weighting["label_types"] == list(LABEL_TYPES), "weighting label order differs")
    vector = weighting["weights_vector"]
    require(len(vector) == len(LABEL_TYPES) and cfg["positive_weights"] == vector, "actual loss vector is missing/inconsistent")
    for index, kind in enumerate(LABEL_TYPES):
        positive = number(weighting["positive_response_tokens"][kind], "positive training tokens")
        negative = number(weighting["negative_response_tokens"][kind], "negative training tokens")
        expected = 50 if not positive else min(50, max(1.0, math.sqrt(negative / positive)))
        require(math.isclose(vector[index], expected, rel_tol=1e-10), "loss vector is not the declared train-derived weighting")
    require(summary["data_sha256"] == weighted["data_sha256"], "weighted trial/calibration data identities differ")
    weighted_run = repository_path(Path(paths["weighted_summary"]).parent)
    for candidate in inputs["weighted_calibration"]["candidates"]:
        name = relative(candidate["checkpoint"])
        require(name.startswith(weighted_run + "/checkpoints/epoch-") and PurePosixPath(name).name in {"epoch-1", "epoch-2", "epoch-3"},
                "weighted calibration checkpoint belongs to another training run")
    audit = inputs["development_audit"]
    require(audit.get("status") == "passed" and audit.get("official_test_read") is False and audit["rows_removed"] == 0,
            "development data audit is not a passed, unchanged train/validation audit")
    for name in ("source_id_overlap", "group_id_overlap", "response_id_overlap"):
        require(audit[name]["count"] == 0, "development identity/group overlap is unresolved")
    for mode in ("exact", "normalized"):
        require(audit["source_text_overlap"][mode]["distinct_texts"] == 0, "development source text overlap is unresolved")
    require(inputs["error_analysis"].get("split") == "validation", "error analysis must be validation-only")
    require(inputs["crf"].get("run_kind") == "full_validation" and inputs["crf"].get("official_test_evaluated") is False,
            "CRF summary must be the full validation run")
    return {"models": models, "family": family, "selected": selected, "base": base, "weighted": weighted}


def metric_table(comparison, names):
    lines = ["| Model | Any P (%) | Any R (%) | Any char F1 (%) | Typed macro F1 (%) | Exact typed span F1 (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        lines.append("| " + md(name) + " | " + " | ".join(f"{value * 100:.2f}" for value in metric_values(comparison["models"][name]["metrics"])) + " |")
    return lines


def render_report(inputs, paths):
    state = validate_inputs(inputs, paths)
    validation, test, decision, errors, audit, crf, weighted_summary = [inputs[name] for name in
        ("validation", "test", "decision", "error_analysis", "development_audit", "crf", "weighted_summary")]
    base, weighted, selected = state["base"], state["weighted"], state["selected"]
    counts = audit["counts"]
    quality = audit["quality_selection"]
    test_data = test["data"]
    boilerplate = audit["response_text_overlap"]["exact"]
    base_default = next(row["default_metrics"] for row in inputs["base_calibration"]["candidates"]
                        if PurePosixPath(row["checkpoint"]).name == "epoch-2")
    grid_taggers = [entry for setting in crf["grid"] for entry in setting["training"].values()]
    control_taggers = list(crf["response_only_control"]["training"].values())
    capped = lambda rows: sum(bool(row["stopped_at_max_iterations"]) for row in rows)
    lines = ["# RAGTruth technical completion report", "",
        "Scope: code, experiments, evaluation, technical error analysis and the offline demo. Slides/presentation "
        "are separate. Original data, CRF and Transformer implementation authorship is preserved; subsequent "
        "integration/completion work belongs to the repository owner, without fabricated personal contributions.", "",
        "## Task, data and integrity", "",
        "Given supplied context and an LLM response, locate annotated response spans and identify four raw "
        "hallucination types. Unsupported content is not necessarily false in the world. Only source/response "
        "text enters the model; IDs, generator/task metadata and annotation comments are excluded. "
        "`implicit_true` and `due_to_null` annotations remain in the targets.", "",
        "References: pinned [RAGTruth source](https://github.com/ParticleMedia/RAGTruth/tree/c103204b9ce28d6bbad859304bf30de72b8ed8fe), "
        "[MIT license](https://github.com/ParticleMedia/RAGTruth/blob/c103204b9ce28d6bbad859304bf30de72b8ed8fe/LICENSE) "
        "and [ACL paper](https://aclanthology.org/2024.acl-long.585/). Raw inputs are checksum-verified. "
        "This project does not reproduce the paper's generative detector or claim scorer/SOTA equivalence.", "",
        "| Cohort | Raw responses | Good responses scored/used | Quality exclusions |", "|---|---:|---:|---|",
        *[f"| {split} | {quality[split]['file_responses']:,} | {quality[split]['included_responses']:,} | {md(quality[split]['excluded_by_quality'])} |"
          for split in ("train", "validation")],
        f"| Reserved test | {test_data['file_responses']:,} | {test_data['included_responses']:,} | {md(test_data['excluded_by_quality'])} |", "",
        f"The fixed manifest has **{counts['train_sources']:,}/{counts['validation_sources']:,} train/validation source IDs** "
        f"and **{counts['train_groups']:,}/{counts['validation_groups']:,} equal-content groups**. Source-ID, group-ID "
        "and response-ID overlaps are zero; exact and normalized source-text overlaps are zero. "
        f"Shared answer boilerplate occurs {boilerplate['train_responses']} times in train and {boilerplate['validation_responses']} "
        f"in validation; all shared gold labels empty: **{str(boilerplate['all_shared_gold_empty']).lower()}**. "
        "The answer 'Unable to answer based on given passages.' is retained across different contexts; "
        "we do not claim zero answer overlap or silently remove rows.", "",
        "| Type | Train spans | Validation spans | Positive train word tokens (%) |", "|---|---:|---:|---:|"]
    for kind in LABEL_TYPES:
        train, val = errors["support"]["train"][kind], errors["support"]["validation"][kind]
        lines.append(f"| {kind} | {train['raw_spans']} | {val['raw_spans']} | {100 * train['positive_word_fraction']:.4f} |")
    lines += ["", "## Models and bounded development", "",
        "**CRF:** four independent O/B/I linear-chain CRFs, with word/shape/neighbour features plus lexical source "
        "presence/novelty. Three regularization settings and Viterbi/posterior decoders are selected by validation "
        "any-character F1. The response-only control keeps the selected setting/decoder. "
        f"Choice: `{md(crf['selected']['name'])}`, `{md(crf['selected']['decoding']['name'])}`; runtime {crf['total_seconds']/60:.1f} minutes. "
        f"At the {crf['config']['training']['max_iterations']}-iteration L-BFGS cap, {capped(grid_taggers)}/{len(grid_taggers)} "
        f"grid taggers and {capped(control_taggers)}/{len(control_taggers)} control taggers stopped at the cap; convergence is not asserted.", "",
        "**Transformer:** pretrained ModernBERT-base plus dropout/four independent sigmoid logits. Multi-hot targets "
        "retain cross-type overlaps; masked BCE supervises only answer tokens intersected with attention_mask. "
        "Both training families use the same source-group split, seed 42, three epochs, FP32/SDPA, batch 1, "
        "accumulation 8, gradient checkpointing, lr 2e-5 and 2048-token cap. About 99.1% of development responses "
        "fit one window. Training selects the lexical best source window per answer chunk without gold; "
        "inference evaluates all windows, takes per-type minimum scores, then averages overlapping answer chunks.", "",
        "The unweighted baseline originally chose epoch 2 by validation BCE and used threshold 0.5. "
        "Its bounded calibration compares saved epochs 1/2 with 11 thresholds per type. One additional "
        "train-only weighting trial uses `min(50,max(1,sqrt(negative/positive)))` over response-token counts. "
        "Its 15-threshold grid evaluates only saved improving checkpoints (one to three). Each type maximizes "
        "validation character F1, ties prefer the higher threshold; checkpoint ties prefer higher any F1, "
        "then earlier declared candidate. These are independent per-type searches, not exhaustive joint grids.", "",
        "| Type | Actual positive-loss weight |", "|---|---:|"]
    for kind, weight in zip(LABEL_TYPES, weighted_summary["positive_weighting"]["weights_vector"]):
        lines.append(f"| {kind} | {weight:.4f} |")
    lines += ["", "| Validation development step | Any character F1 (%) | Typed macro F1 (%) |", "|---|---:|---:|"]
    for title, value in (("Unweighted, original 0.5", base_default), ("Unweighted, calibrated", base["validation_metrics"]),
                         ("Weighted trial, calibrated", weighted["validation_metrics"])):
        lines.append(f"| {title} | {100*value['any']['f1']:.3f} | {100*value['macro_f1_types']:.3f} |")
    lines += ["", f"The frozen primary `transformer` is **{state['family']}**, selected by validation typed macro F1, "
        "then any F1, with exact family ties retaining the unweighted baseline. "
        f"The weighted trial has {len(inputs['weighted_calibration']['candidates'])} saved checkpoint candidates. "
        "No weighted improvement is assumed from the training loss alone. Primary checkpoint and thresholds:", "",
        f"`{md(selected['checkpoint'])}`", "", "| Type | Frozen threshold |", "|---|---:|"]
    lines += [f"| {kind} | {selected['thresholds_by_type'][kind]:g} |" for kind in LABEL_TYPES]
    lines += ["", "## Common metrics and results", "",
        f"Metric version: `{METRIC_VERSION}`. Character micro P/R/F1 counts original Unicode codepoints including "
        "whitespace; any uses the union across types, while typed macro averages all four fixed types. "
        "Exact typed spans require original boundaries and type. Identical duplicates count once, but overlapping/"
        "touching raw spans are not merged for exact matching. Empty denominators yield zero. All table metrics "
        "below are percentages, not token accuracy; [full definitions](../docs/EVALUATION.md).", "",
        "### Validation (used for selection)", "",
        "Development scores include decoder/checkpoint/family selection and are not independent performance estimates.", "",
        *metric_table(validation, list(state["models"]) + sorted(set(validation["models"]) - set(state["models"]))), "",
        "### Reserved official test", "",
        "The model set is fixed by [FINAL_PROTOCOL.json](../docs/FINAL_PROTOCOL.json). Final-test jobs require "
        "committed scientific decisions and exact model/code/data hashes before prediction/evaluation. These "
        "already-scored aggregates record frozen-job gold provenance; this report neither opens test records "
        "nor performs post-test model selection.", "", *metric_table(test, state["models"]), "",
        "| Model | Evident Conflict F1 (%) | Subtle Conflict F1 (%) | Evident Baseless F1 (%) | Subtle Baseless F1 (%) |",
        "|---|---:|---:|---:|---:|"]
    for name in state["models"]:
        c = test["models"][name]["metrics"]["character"]
        lines.append("| " + name + " | " + " | ".join(f"{100*c[kind]['f1']:.2f}" for kind in LABEL_TYPES) + " |")
    lines += ["", "## Interpretation, limitations and reproduction", "",
        "Subtle Conflict has little support; the original 0.5 decoder produced no validation prediction for it. "
        "Threshold calibration changes the precision/recall operating point, not reasoning ability. Rare-type "
        "reliability remains limited. Weighted/unweighted trials also have different declared threshold grids; "
        "their validation gap is not attributed solely to loss weighting. CRF versus Transformer differences "
        "combine architecture and input features; only the two CRF variants directly control source features.", "",
        "Best-window training can miss evidence; minimum-over-window inference can suppress real errors. "
        "Word/subword edges, whitespace bridging and same-type unions limit exact-boundary recovery. "
        "A single fixed seed and bounded trials do not support multi-seed or statistical confidence claims. "
        "Validation error examples follow declared categories and are not a representative random sample.", "",
        "[RUNNING.md](../docs/RUNNING.md), [CALIBRATION.md](../docs/CALIBRATION.md) and "
        "[DEMO.md](../docs/DEMO.md) document execution. The offline demo shows saved validation predictions "
        "beside separately labeled human annotations, without live inference or factuality guarantees. "
        "Raw data, full predictions and weights stay ignored; aggregate reports, scripts and fixed manifests are versioned.", "",
        "```powershell", "python -m unittest discover -s tests -v",
        "python scripts/run_job.py configs/jobs/crf_validation.json",
        "python scripts/run_job.py configs/jobs/transformer_train.json",
        "python scripts/run_job.py configs/jobs/transformer_weighted_train.json", "```", "",
        "Exact frozen inference commands are archived under `configs/final/`; test results must not drive retuning. "
        "The report builder consumes only the supplied scored summaries, calibration metadata and frozen decision. "
        "It refuses wrong splits/model sets and existing output files.", "",
        f"Frozen decision SHA-256: `{digest(paths['decision'])}`.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ARGUMENTS:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite technical report")
        paths = {name.replace("-", "_"): getattr(args, name.replace("-", "_")) for name in ARGUMENTS}
        require(args.output.resolve() not in {path.resolve() for path in paths.values()}, "report output collides with input")
        inputs = {name: read(path) for name, path in paths.items()}
        document = render_report(inputs, paths)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
    except (ValueError, KeyError, OSError, StopIteration) as exc:
        parser.exit(1, f"Technical report failed: {exc}\n")
    print(json.dumps({"output": str(args.output), "primary_model": "transformer", "metric_version": METRIC_VERSION}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
