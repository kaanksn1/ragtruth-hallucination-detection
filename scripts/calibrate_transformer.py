"""Bounded per-type threshold selection on validation only; never retrains weights."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragtruth_data.alignment import char_scores_to_spans
from ragtruth_data.data import sha256_file, write_jsonl
from ragtruth_data.schema import LABEL_TYPES
from scripts.baseline_crf import MIN_PRECISION_FLOOR, char_scores
from scripts.transformer import read_records, verify_data, verify_validation_identity


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_grid(grid):
    require(isinstance(grid, list) and grid and all(type(t) in (int, float) and math.isfinite(t) and 0 < t < 1 for t in grid),
            "threshold grid must contain finite values in (0,1)")
    require(grid == sorted(set(grid)), "threshold grid must be sorted and unique")


def load_calibration(path, checkpoint, saved):
    """Bind a separate decoder artifact to an unchanged inference checkpoint.

    The normal model loader independently verifies every weight/tokenizer file.
    This function does not edit or load model weights.
    """
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    require(calibration.get("schema_version") == 1 and calibration.get("status") == "selected_on_validation",
            "unsupported calibration artifact")
    require(calibration.get("validation_split") == "validation", "calibration must be selected on validation")
    require(calibration.get("label_types") == list(LABEL_TYPES), "calibration label order mismatch")
    require(calibration.get("checkpoint_sha256") == sha256_file(Path(checkpoint) / "checkpoint.json"),
            "calibration belongs to a different checkpoint")
    require(calibration.get("data_sha256") == saved.get("metadata", {}).get("data_sha256")
            and calibration.get("data_sha256"), "calibration training/validation identity mismatch")
    grid = calibration.get("threshold_grid")
    validate_grid(grid)
    thresholds = calibration.get("thresholds_by_type")
    require(isinstance(thresholds, dict) and set(thresholds) == set(LABEL_TYPES)
            and all(type(t) in (int, float) and t in grid for t in thresholds.values()),
            "calibration requires one selected grid threshold for each of four types")
    return calibration


def decode_cached(record, scores_by_type, thresholds):
    require(len(scores_by_type) == len(LABEL_TYPES)
            and all(len(values) == len(record["response"])
                    and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values)
                    for values in scores_by_type),
            "cache scores must align with four types and original response characters")
    spans = []
    for kind, values in zip(LABEL_TYPES, scores_by_type):
        spans.extend(char_scores_to_spans(record["response"], values, thresholds[kind], kind))
    spans.sort(key=lambda span: (span["start"], span["end"], LABEL_TYPES.index(span["label_type"])))
    return {"id": record["id"], "spans": spans}


def _characters(spans):
    return {index for span in spans for index in range(span["start"], span["end"])}


def _prf(tp, predicted, gold):
    precision, recall = tp / predicted if predicted else 0.0, tp / gold if gold else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (predicted + gold) if predicted + gold else 0.0,
            "true_positive_chars": tp, "predicted_chars": predicted, "gold_chars": gold}


def tune_thresholds(records, cached, grid, min_precision=MIN_PRECISION_FLOOR):
    """Each type maximizes its own character F1; exact ties prefer higher threshold.

    Decode with the real span policy first, including whitespace bridges. This
    prevents an approximate token/character objective from selecting the threshold.

    Thresholds below ``min_precision`` are excluded unless they predict zero
    characters (never a false-positive flood by construction). Without this
    floor, a very low threshold on a rare type can flood almost every
    character positive, posting a nonzero-but-useless F1 that still beats the
    exact 0.0 F1 of predicting nothing and would otherwise win the search —
    the mechanism behind an accepted false-positive explosion for a tiny
    score gain. If every grid threshold for a type fails the floor, the
    highest (most conservative) threshold is kept instead of the nominal
    best-F1 one.
    """
    validate_grid(grid)
    ids = [record["id"] for record in records]
    require(len(ids) == len(set(ids)) and set(ids) == set(cached), "calibration cache IDs differ from validation")
    totals = {kind: {threshold: [0, 0, 0] for threshold in grid} for kind in LABEL_TYPES}
    for record in records:
        values_by_type = cached[record["id"]]
        require(len(values_by_type) == len(LABEL_TYPES)
                and all(len(values) == len(record["response"])
                        and all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in values)
                        for values in values_by_type), "invalid cache character scores")
        for index, kind in enumerate(LABEL_TYPES):
            gold = _characters([label for label in record["labels"] if label["label_type"] == kind])
            for threshold in grid:
                predicted = _characters(char_scores_to_spans(record["response"], values_by_type[index], threshold, kind))
                count = totals[kind][threshold]
                count[0] += len(gold & predicted)
                count[1] += len(predicted)
                count[2] += len(gold)
    table, selected = {}, {}
    for kind in LABEL_TYPES:
        table[kind] = [{"threshold": threshold, **_prf(*totals[kind][threshold])} for threshold in grid]
        eligible = [row for row in table[kind] if row["predicted_chars"] == 0 or row["precision"] >= min_precision]
        pool = eligible if eligible else table[kind]
        best = (max(pool, key=lambda row: (row["f1"], row["threshold"])) if eligible
                else max(table[kind], key=lambda row: row["threshold"]))
        selected[kind] = best["threshold"]
    return selected, table


def tune_global_threshold(records, cached, grid):
    """One shared threshold for all four types, chosen by the combined ('any') detection F1.

    Unlike ``tune_thresholds`` (each type's own best F1 independently), this mirrors the
    CRF baseline's shared-threshold search: a single value applied to every type, picked
    directly on the primary detection metric. It is the 'global' point of comparison against
    independent per-type tuning and the un-calibrated default.
    """
    validate_grid(grid)
    ids = [record["id"] for record in records]
    require(len(ids) == len(set(ids)) and set(ids) == set(cached), "calibration cache IDs differ from validation")
    rows = []
    for threshold in grid:
        thresholds = {kind: threshold for kind in LABEL_TYPES}
        spans = {record["id"]: decode_cached(record, cached[record["id"]], thresholds)["spans"] for record in records}
        gold = {record["id"]: record["labels"] for record in records}
        rows.append({"threshold": threshold, "metrics": char_scores(gold, spans)})
    best = max(rows, key=lambda row: (row["metrics"]["any"]["f1"], row["threshold"]))
    return {kind: best["threshold"] for kind in LABEL_TYPES}, rows


def tune_joint_thresholds(records, cached, grid, min_precision=MIN_PRECISION_FLOOR):
    """Per-type thresholds chosen jointly to maximize the combined ('any') detection F1.

    Mirrors ``scripts/baseline_crf.py::tune_joint_thresholds``. ``tune_thresholds`` lets
    each type maximize its own F1 independently, which can leave the primary detection
    metric on the table (real measurement: on this project's validation split, independent
    per-type tuning scored lower combined F1 than the single shared threshold, despite a
    higher macro F1). This instead searches combinations of thresholds -- restricted to each
    type's own false-positive-floor-eligible values, so a flooding threshold is never admitted
    just because some combination nominally helps the union -- and keeps the combination with
    the highest combined 'any' F1. Ties prefer the higher macro F1, then higher thresholds.
    """
    validate_grid(grid)
    ids = [record["id"] for record in records]
    require(len(ids) == len(set(ids)) and set(ids) == set(cached), "calibration cache IDs differ from validation")
    gold_by_type = {record["id"]: {kind: _characters([label for label in record["labels"] if label["label_type"] == kind])
                                   for kind in LABEL_TYPES} for record in records}
    gold_any = {record["id"]: _characters(record["labels"]) for record in records}

    char_sets, independent_table, eligible = {}, {}, {}
    for index, kind in enumerate(LABEL_TYPES):
        char_sets[kind] = {}
        rows = []
        for threshold in sorted(grid):
            predicted_sets, tp, predicted_total, gold_total = {}, 0, 0, 0
            for record in records:
                predicted = _characters(char_scores_to_spans(record["response"], cached[record["id"]][index], threshold, kind))
                predicted_sets[record["id"]] = predicted
                g = gold_by_type[record["id"]][kind]
                tp += len(g & predicted)
                predicted_total += len(predicted)
                gold_total += len(g)
            char_sets[kind][threshold] = predicted_sets
            rows.append({"threshold": threshold, **_prf(tp, predicted_total, gold_total)})
        independent_table[kind] = rows
        values = [row["threshold"] for row in rows if row["predicted_chars"] == 0 or row["precision"] >= min_precision]
        eligible[kind] = sorted(values) if values else [max(grid)]

    best_combo, best_key, evaluated = None, None, 0
    for combo in itertools.product(*(eligible[kind] for kind in LABEL_TYPES)):
        tp = predicted_total = gold_total = 0
        type_tp = {kind: 0 for kind in LABEL_TYPES}
        type_predicted = {kind: 0 for kind in LABEL_TYPES}
        type_gold = {kind: 0 for kind in LABEL_TYPES}
        for record in records:
            rid = record["id"]
            predicted_any = set()
            for kind, threshold in zip(LABEL_TYPES, combo):
                predicted = char_sets[kind][threshold][rid]
                predicted_any |= predicted
                g = gold_by_type[rid][kind]
                type_tp[kind] += len(g & predicted)
                type_predicted[kind] += len(predicted)
                type_gold[kind] += len(g)
            g_any = gold_any[rid]
            tp += len(g_any & predicted_any)
            predicted_total += len(predicted_any)
            gold_total += len(g_any)
        any_scores = _prf(tp, predicted_total, gold_total)
        per_type = {kind: _prf(type_tp[kind], type_predicted[kind], type_gold[kind]) for kind in LABEL_TYPES}
        macro_f1 = sum(per_type[kind]["f1"] for kind in LABEL_TYPES) / len(LABEL_TYPES)
        key = (any_scores["f1"], macro_f1, combo)
        evaluated += 1
        if best_key is None or key > best_key:
            best_key = key
            best_combo = {"thresholds": dict(zip(LABEL_TYPES, combo)), "any": any_scores,
                          "macro_f1_types": macro_f1, "per_type": per_type}
    return best_combo["thresholds"], {"evaluated_combinations": evaluated, "eligible_per_type": eligible,
                                      "selected": best_combo, "independent_table": independent_table}


def select_best_configuration(candidates):
    """Highest validation detection ('any') F1 among named configurations; macro F1 only ties.

    ``candidates`` maps a configuration name (e.g. 'default', 'global', 'per_type') to its
    validation ``char_scores``-shaped metrics. This never lets a configuration with a macro-F1
    or per-class gain but a *lower* detection F1 win over one that is strictly better (or
    equal) on the primary metric.
    """
    if not candidates:
        raise ValueError("no configurations to select from")
    order = list(candidates)
    return max(order, key=lambda name: (candidates[name]["any"]["f1"], candidates[name]["macro_f1_types"],
                                        -order.index(name)))


def select_checkpoint(candidates):
    """Highest validation detection ('any') character F1; macro F1 only breaks ties.

    Detection F1 (the union across the four types) is the primary hallucination-detection
    metric; four-class macro F1 is a secondary tie-break only. Picking by macro F1 first
    can prefer a checkpoint/threshold set that regresses detection F1 for a small macro
    gain -- this ordering is what prevents that.
    """
    if not candidates:
        raise ValueError("no calibration candidates to select from")
    return max(candidates, key=lambda row: (row["calibrated_metrics"]["any"]["f1"],
                                           row["calibrated_metrics"]["macro_f1_types"], -row["candidate_index"]))


def _cache_identity(checkpoint, saved, identity):
    return {"schema_version": 1, "checkpoint_sha256": sha256_file(Path(checkpoint) / "checkpoint.json"),
            "data_sha256": identity, "label_types": list(LABEL_TYPES),
            "aggregation": saved["config"]["aggregation"], "scope": "validation only"}


def read_cache(path, expected, records):
    meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8"))
    require(meta.get("identity") == expected and meta.get("cache_sha256") == sha256_file(path), "probability cache identity/checksum mismatch")
    by_id = {record["id"]: record for record in records}
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rid = row["id"]
            require(rid in by_id and rid not in result, "duplicate or unexpected cache response ID")
            record = by_id[rid]
            require(row["response_sha256"] == text_sha256(record["response"]), "cache response text changed")
            values = row["scores_by_type"]
            require(isinstance(values, list) and len(values) == len(LABEL_TYPES)
                    and all(isinstance(column, list) and len(column) == len(record["response"])
                            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in column)
                            for column in values), "invalid cached probabilities")
            result[rid] = values
    require(set(result) == set(by_id), "probability cache is missing validation IDs")
    return result


def build_cache(path, checkpoint, saved, identity, records, device):
    import torch
    import transformer as tr

    require(not path.exists() and not path.with_suffix(path.suffix + ".meta.json").exists(), "refusing to overwrite a probability cache")
    model, tokenizer, _ = tr.load_checkpoint(checkpoint, device)
    config = dict(saved["config"])
    if device.type == "cpu":
        config["precision"] = "fp32"
    tr.seed_everything(config["seed"])
    tr.autocast_context(device, config["precision"])
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=2) as handle:
        for count, (record, scores) in enumerate(tr.predict_character_scores(model, records, tokenizer, config, device), 1):
            handle.write(json.dumps({"id": record["id"], "response_sha256": text_sha256(record["response"]),
                                     "scores_by_type": scores}, separators=(",", ":")) + "\n")
            if count % 250 == 0:
                print(json.dumps({"checkpoint": str(checkpoint), "cached_validation_responses": count}), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    meta = {"identity": _cache_identity(checkpoint, saved, identity), "cache_sha256": sha256_file(path),
            "responses": len(records), "seconds": time.monotonic() - started}
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/calibration.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--split", choices=("validation",), default="validation")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    require(config.get("schema_version") == 1 and config.get("split") == "validation", "calibration only supports validation")
    require(config.get("selection_metric") == "detection_f1_then_macro_f1_types",
            "calibration selection metric must be predeclared detection_f1_then_macro_f1_types")
    validate_grid(config.get("threshold_grid"))
    require(isinstance(config.get("checkpoints"), list) and 1 <= len(config["checkpoints"]) <= 3
            and len(set(config["checkpoints"])) == len(config["checkpoints"]), "declare one to three distinct checkpoint candidates")
    require(not any((args.output / name).exists() for name in ("calibration.json", "calibration_report.json", "validation_predictions.jsonl", "probabilities")),
            "refusing to overwrite calibration outputs")
    args.output.mkdir(parents=True, exist_ok=True)
    import transformer as tr
    device = tr.choose_device(args.device)
    candidates = []
    for index, checkpoint_name in enumerate(config["checkpoints"]):
        checkpoint = (ROOT / checkpoint_name).resolve()
        require(checkpoint.is_relative_to(ROOT / "outputs"), "checkpoint must be inside outputs/")
        saved = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
        base = saved["config"]
        identity = verify_data(base, ("validation",))
        verify_validation_identity(saved, identity)
        inputs = read_records(base["processed_dir"], "validation", base["split_manifest"], with_labels=False)
        labeled = read_records(base["processed_dir"], "validation", base["split_manifest"])
        require(inputs and len(inputs) == len(labeled), "empty/inconsistent validation data")
        cache_path = args.output / "probabilities" / f"candidate-{index + 1}.jsonl.gz"
        build_cache(cache_path, checkpoint, saved, identity, inputs, device)
        cached = read_cache(cache_path, _cache_identity(checkpoint, saved, identity), labeled)
        gold = {record["id"]: record["labels"] for record in labeled}

        per_type_thresholds, per_type_search = tune_thresholds(labeled, cached, config["threshold_grid"])
        per_type = [decode_cached(record, cached[record["id"]], per_type_thresholds) for record in labeled]
        global_thresholds, global_search = tune_global_threshold(labeled, cached, config["threshold_grid"])
        global_ = [decode_cached(record, cached[record["id"]], global_thresholds) for record in labeled]
        joint_thresholds, joint_search = tune_joint_thresholds(labeled, cached, config["threshold_grid"])
        joint = [decode_cached(record, cached[record["id"]], joint_thresholds) for record in labeled]
        default_thresholds = {kind: base["threshold"] for kind in LABEL_TYPES}
        default = [decode_cached(record, cached[record["id"]], default_thresholds) for record in labeled]

        configurations = {
            "default": {"thresholds": default_thresholds, "rows": default},
            "global": {"thresholds": global_thresholds, "rows": global_},
            "per_type": {"thresholds": per_type_thresholds, "rows": per_type},
            "joint": {"thresholds": joint_thresholds, "rows": joint},
        }
        configuration_metrics = {name: char_scores(gold, {row["id"]: row["spans"] for row in value["rows"]})
                                 for name, value in configurations.items()}
        # Never ship per-type (or global) tuning if it actually regresses the primary
        # detection metric relative to another eligible configuration -- see the real CRF
        # measurement in docs/BASELINE_CRF.md (2026-11-01) that motivated this comparison.
        winner = select_best_configuration(configuration_metrics)
        tuned = configurations[winner]["rows"]
        thresholds = configurations[winner]["thresholds"]

        candidate = {"candidate_index": index, "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "data_sha256": saved["metadata"]["data_sha256"], "thresholds_by_type": thresholds,
            "threshold_search": per_type_search, "global_threshold_search": global_search,
            "joint_threshold_search": {"evaluated_combinations": joint_search["evaluated_combinations"],
                                       "eligible_per_type": joint_search["eligible_per_type"]},
            "configuration_winner": winner, "configuration_metrics": configuration_metrics,
            "default_metrics": configuration_metrics["default"],
            "calibrated_metrics": configuration_metrics[winner],
            "cache_path": cache_path.relative_to(args.output).as_posix(), "validation_responses": len(labeled)}
        candidates.append(candidate)
        write_jsonl(args.output / f"validation_predictions_candidate_{index + 1}.jsonl", tuned)
        write_jsonl(args.output / f"validation_predictions_default_{index + 1}.jsonl", default)
        print(json.dumps({"candidate": index + 1, "configuration_winner": winner, "thresholds": thresholds,
                          "any_f1": configuration_metrics[winner]["any"]["f1"],
                          "macro_f1_types": candidate["calibrated_metrics"]["macro_f1_types"]}), flush=True)
        del cached, tuned, default, global_, per_type, joint, labeled, inputs
    selected = select_checkpoint(candidates)
    artifact = {"schema_version": 1, "status": "selected_on_validation", "validation_split": "validation",
                "label_types": list(LABEL_TYPES), "threshold_grid": config["threshold_grid"],
                "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
                "data_sha256": selected["data_sha256"], "thresholds_by_type": selected["thresholds_by_type"],
                "selection_metric": "detection_f1_then_macro_f1_types", "threshold_tie_break": "higher threshold",
                "checkpoint_tie_break": "higher macro character F1 (4-type average), then earlier declared checkpoint",
                "validation_metrics": selected["calibrated_metrics"], "official_test_evaluated": False}
    (args.output / "calibration.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    original = args.output / f"validation_predictions_candidate_{selected['candidate_index'] + 1}.jsonl"
    (args.output / "validation_predictions.jsonl").write_bytes(original.read_bytes())
    report = {"config": config, "candidates": candidates, "selected": artifact,
              "config_sha256": sha256_file(args.config), "code_sha256": sha256_file(Path(__file__)),
              "metric_note": "project character F1 including span whitespace; CRF helper, independently checked by shared scorer",
              "official_test_evaluated": False}
    (args.output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_checkpoint": selected["checkpoint"], "calibration": str(args.output / "calibration.json")}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from None
