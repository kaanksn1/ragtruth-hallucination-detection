"""Bounded per-type threshold selection on validation only; never retrains weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragtruth_data.alignment import char_scores_to_spans
from ragtruth_data.data import sha256_file, write_jsonl
from ragtruth_data.schema import LABEL_TYPES
from scripts.baseline_crf import MIN_PRECISION_FLOOR, char_scores
from scripts.transformer import read_records, verify_data, verify_validation_identity

LEGACY_METRIC = "macro_f1_types"
SUCCESSOR_METRIC = "detection_f1_then_macro_f1_types"
SELECTION_MODES = {"legacy_macro": LEGACY_METRIC, "successor_any": SUCCESSOR_METRIC}


class NoEligibleThreshold(ValueError):
    """No grid decoder satisfies the declared validation precision constraint."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_grid(grid):
    require(isinstance(grid, list) and grid and all(type(t) in (int, float) and math.isfinite(t) and 0 < t < 1 for t in grid),
            "threshold grid must contain finite values in (0,1)")
    require(grid == sorted(set(grid)), "threshold grid must be sorted and unique")


def validate_precision_floor(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            "precision floor must be finite and in [0,1]")


def selection_mode(config, requested=None):
    metric = config.get("selection_metric")
    require(metric in SELECTION_MODES.values(), "unsupported predeclared calibration selection metric")
    mode = next(name for name, value in SELECTION_MODES.items() if value == metric)
    require(requested is None or requested == mode, "selection mode conflicts with the predeclared config metric")
    return mode


def validate_search_inputs(records, cached, grid):
    validate_grid(grid)
    require(records, "calibration requires nonempty validation records")
    ids = [record["id"] for record in records]
    require(len(ids) == len(set(ids)) and set(ids) == set(cached), "calibration cache IDs differ from validation")
    for record in records:
        values = cached[record["id"]]
        require(len(values) == len(LABEL_TYPES)
                and all(len(column) == len(record["response"])
                        and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in column)
                        for column in values), "invalid cache character scores")


def configuration_precision(records, rows, min_precision):
    """Check every raw type; an empty prediction is an admissible abstention."""
    validate_precision_floor(min_precision)
    predicted = {row["id"]: row["spans"] for row in rows}
    require(len(predicted) == len(rows) and set(predicted) == {r["id"] for r in records},
            "configuration prediction IDs differ from validation")
    by_type = {}
    for kind in LABEL_TYPES:
        tp = count = gold = 0
        for record in records:
            g = _characters([s for s in record["labels"] if s["label_type"] == kind])
            p = _characters([s for s in predicted[record["id"]] if s["label_type"] == kind])
            tp += len(g & p)
            count += len(p)
            gold += len(g)
        by_type[kind] = _prf(tp, count, gold)
    return {"eligible": all(row["predicted_chars"] == 0 or row["precision"] >= min_precision
                            for row in by_type.values()), "by_type": by_type}


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
    if "selection_policy" in calibration:
        policy = calibration["selection_policy"]
        require(isinstance(policy, dict) and policy.get("mode") in SELECTION_MODES,
                "unsupported calibration selection policy")
        require(SELECTION_MODES[policy["mode"]] == calibration.get("selection_metric"),
                "calibration policy and selection metric differ")
        allowed = {"per_type"} if policy["mode"] == "legacy_macro" else {"default", "global", "per_type", "joint"}
        require(calibration.get("selected_configuration") in allowed, "unsupported selected calibration configuration")
        validate_precision_floor(policy.get("minimum_validation_type_precision"))
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
    score gain. If every grid threshold fails, reject this search instead of
    silently returning a still-ineligible maximum threshold. The historical
    macro protocol explicitly calls this with min_precision=0 (no floor).
    """
    validate_precision_floor(min_precision)
    validate_search_inputs(records, cached, grid)
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
        if not eligible:
            raise NoEligibleThreshold("no eligible precision-floor threshold for " + kind)
        best = max(eligible, key=lambda row: (row["f1"], row["threshold"]))
        selected[kind] = best["threshold"]
    return selected, table


def tune_global_threshold(records, cached, grid, min_precision=MIN_PRECISION_FLOOR):
    """One shared threshold for all four types, chosen by the combined ('any') detection F1.

    Unlike ``tune_thresholds`` (each type's own best F1 independently), this mirrors the
    CRF baseline's shared-threshold search: a single value applied to every type, picked
    directly on the primary detection metric. It is the 'global' point of comparison against
    independent per-type tuning and the un-calibrated default.
    """
    validate_precision_floor(min_precision)
    validate_search_inputs(records, cached, grid)
    rows = []
    for threshold in grid:
        thresholds = {kind: threshold for kind in LABEL_TYPES}
        decoded = [decode_cached(record, cached[record["id"]], thresholds) for record in records]
        spans = {row["id"]: row["spans"] for row in decoded}
        gold = {record["id"]: record["labels"] for record in records}
        rows.append({"threshold": threshold, "metrics": char_scores(gold, spans),
                     "precision_constraint": configuration_precision(records, decoded, min_precision)})
    eligible = [row for row in rows if row["precision_constraint"]["eligible"]]
    if not eligible:
        raise NoEligibleThreshold("no eligible shared precision-floor threshold")
    best = max(eligible, key=lambda row: (row["metrics"]["any"]["f1"], row["threshold"]))
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
    validate_precision_floor(min_precision)
    validate_search_inputs(records, cached, grid)
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
        if not values:
            raise NoEligibleThreshold("no eligible joint precision-floor threshold for " + kind)
        eligible[kind] = sorted(values)

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
    def key(name):
        metrics = candidates[name]
        values = metrics.get("any", {}).get("f1"), metrics.get("macro_f1_types")
        require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values),
                "invalid configuration selection metrics")
        return (*values, -order.index(name))
    return max(order, key=key)


def select_checkpoint(candidates, metric=SUCCESSOR_METRIC):
    """Highest validation detection ('any') character F1; macro F1 only breaks ties.

    Detection F1 (the union across the four types) is the primary hallucination-detection
    metric; four-class macro F1 is a secondary tie-break only. Picking by macro F1 first
    can prefer a checkpoint/threshold set that regresses detection F1 for a small macro
    gain -- this ordering is what prevents that.
    """
    if not candidates:
        raise ValueError("no calibration candidates to select from")
    require(metric in SELECTION_MODES.values(), "unsupported checkpoint selection metric")
    def key(row):
        metrics = row["calibrated_metrics"]
        any_f1, macro = metrics["any"]["f1"], metrics["macro_f1_types"]
        require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in (any_f1, macro)),
                "invalid calibration selection metrics")
        return (macro, any_f1, -row["candidate_index"]) if metric == LEGACY_METRIC else (any_f1, macro, -row["candidate_index"])
    return max(candidates, key=key)


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


def compare_configurations(records, cached, grid, default_threshold, mode):
    """Return diagnostics for all computed decoders, but select eligible ones only."""
    require(mode in SELECTION_MODES, "unsupported calibration selection mode")
    require(type(default_threshold) in (int, float) and math.isfinite(default_threshold)
            and 0 < default_threshold < 1, "invalid default checkpoint threshold")
    validate_search_inputs(records, cached, grid)
    floor = 0.0 if mode == "legacy_macro" else MIN_PRECISION_FLOOR
    default_thresholds = {kind: default_threshold for kind in LABEL_TYPES}
    configurations = {"default": {"thresholds": default_thresholds,
        "rows": [decode_cached(r, cached[r["id"]], default_thresholds) for r in records]}}
    searches, rejected, timings = {}, {}, {}
    functions = {"per_type": tune_thresholds} if mode == "legacy_macro" else {
        "per_type": tune_thresholds, "global": tune_global_threshold, "joint": tune_joint_thresholds}
    for name, function in functions.items():
        started = time.monotonic()
        try:
            thresholds, search = function(records, cached, grid, min_precision=floor)
        except NoEligibleThreshold as exc:
            rejected[name] = str(exc)
            continue
        finally:
            timings[name] = time.monotonic() - started
        searches[name] = search
        configurations[name] = {"thresholds": thresholds,
            "rows": [decode_cached(r, cached[r["id"]], thresholds) for r in records]}
    gold = {r["id"]: r["labels"] for r in records}
    metrics, constraints = {}, {}
    for name, value in configurations.items():
        metrics[name] = char_scores(gold, {r["id"]: r["spans"] for r in value["rows"]})
        constraints[name] = configuration_precision(records, value["rows"], floor)
        if any(t not in grid for t in value["thresholds"].values()):
            constraints[name]["eligible"] = False
            constraints[name]["reason"] = "decoder thresholds are outside the declared grid"
    if mode == "legacy_macro":
        # Historical configs requested only independent per-type calibration;
        # default was a diagnostic, not another selection candidate.
        winner = "per_type"
    else:
        eligible = {name: metrics[name] for name in ("default", "global", "per_type", "joint")
                    if name in metrics and constraints[name]["eligible"]}
        winner = select_best_configuration(eligible) if eligible else None
    return {"configurations": configurations, "metrics": metrics, "constraints": constraints,
            "searches": searches, "rejected_searches": rejected, "search_seconds": timings, "winner": winner}


def producer_identity():
    """Record the actual producer bytes, including an uncommitted intermediate state."""
    files = ("scripts/calibrate_transformer.py", "scripts/baseline_crf.py", "scripts/transformer.py",
             "transformer.py", "ragtruth_data/features.py", "ragtruth_data/alignment.py", "ragtruth_data/data.py", "ragtruth_data/schema.py")
    identity = {"code_sha256": {name: sha256_file(ROOT / name) for name in files}}
    try:
        base = ["git", "-c", "safe.directory=" + str(ROOT)]
        commit = subprocess.run([*base, "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                capture_output=True, text=True, encoding="utf-8").stdout.strip()
        dirty = subprocess.run([*base, "status", "--porcelain"], cwd=ROOT, check=True,
                               capture_output=True, text=True, encoding="utf-8").stdout
        identity["git"] = {"available": True, "commit": commit, "dirty": bool(dirty.strip()),
                           "status_sha256": hashlib.sha256(dirty.encode("utf-8")).hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        identity["git"] = {"available": False}
    return identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/calibration.json"))
    parser.add_argument("--selection-mode", choices=tuple(SELECTION_MODES),
                        help="Optional explicit mode; must match the supplied versioned config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--split", choices=("validation",), default="validation")
    args = parser.parse_args(argv)
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    require(config.get("schema_version") == 1 and config.get("split") == "validation", "calibration only supports validation")
    mode = selection_mode(config, args.selection_mode)
    validate_grid(config.get("threshold_grid"))
    require(isinstance(config.get("checkpoints"), list) and 1 <= len(config["checkpoints"]) <= 3
            and all(isinstance(name, str) and name for name in config["checkpoints"])
            and len(set(config["checkpoints"])) == len(config["checkpoints"]), "declare one to three distinct checkpoint candidates")
    require(not any((args.output / name).exists() for name in ("calibration.json", "calibration_report.json", "validation_predictions.jsonl", "probabilities")),
            "refusing to overwrite calibration outputs")
    producer = producer_identity()
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    policy = {"version": "legacy-per-type-macro-v1" if mode == "legacy_macro" else "constrained-any-v2",
              "mode": mode, "selection_metric": config["selection_metric"],
              "minimum_validation_type_precision": 0.0 if mode == "legacy_macro" else MIN_PRECISION_FLOOR,
              "empty_prediction_allowed": True, "no_eligible_threshold": "reject configuration; no maximum-grid fallback",
              "configuration_order": ["per_type"] if mode == "legacy_macro" else ["default", "global", "per_type", "joint"]}
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
        comparison = compare_configurations(labeled, cached, config["threshold_grid"], base["threshold"], mode)
        configurations, winner = comparison["configurations"], comparison["winner"]
        searches = comparison["searches"]
        candidate = {"candidate_index": index, "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "data_sha256": saved["metadata"]["data_sha256"],
            "threshold_search": searches.get("per_type"), "global_threshold_search": searches.get("global"),
            "joint_threshold_search": ({key: searches["joint"][key] for key in ("evaluated_combinations", "eligible_per_type")}
                                       if "joint" in searches else None),
            "configuration_winner": winner, "configuration_metrics": comparison["metrics"],
            "configuration_constraints": comparison["constraints"], "rejected_searches": comparison["rejected_searches"],
            "search_seconds": comparison["search_seconds"],
            "default_metrics": comparison["metrics"]["default"],
            "cache_path": cache_path.relative_to(args.output).as_posix(), "validation_responses": len(labeled)}
        if winner is not None:
            candidate.update(thresholds_by_type=configurations[winner]["thresholds"],
                             calibrated_metrics=comparison["metrics"][winner])
            write_jsonl(args.output / f"validation_predictions_candidate_{index + 1}.jsonl", configurations[winner]["rows"])
        candidates.append(candidate)
        write_jsonl(args.output / f"validation_predictions_default_{index + 1}.jsonl", configurations["default"]["rows"])
        print(json.dumps({"candidate": index + 1, "configuration_winner": winner,
                          "eligible": winner is not None}), flush=True)
        del cached, comparison, configurations, labeled, inputs
    require(producer_identity()["code_sha256"] == producer["code_sha256"], "calibration producer code changed during the run")
    require(sha256_file(args.config) == config_sha256, "calibration config changed during the run")
    finished_utc = datetime.now(timezone.utc).isoformat()
    report = {"config": config, "candidates": candidates, "selection_policy": policy, "producer": producer,
              "started_utc": started_utc, "finished_utc": finished_utc, "elapsed_seconds": time.monotonic() - started,
              "config_sha256": config_sha256, "code_sha256": sha256_file(Path(__file__)),
              "metric_note": "project character F1 including span whitespace; validation-selected, not independent performance",
              "official_test_evaluated": False}
    eligible = [candidate for candidate in candidates if candidate.get("configuration_winner") is not None]
    if not eligible:
        report["status"] = "rejected_no_eligible_configuration"
        (args.output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise NoEligibleThreshold("no eligible calibration configuration; default diagnostics retained in calibration_report.json")
    selected = select_checkpoint(eligible, metric=config["selection_metric"])
    artifact = {"schema_version": 1, "status": "selected_on_validation", "validation_split": "validation",
                "label_types": list(LABEL_TYPES), "threshold_grid": config["threshold_grid"],
                "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
                "data_sha256": selected["data_sha256"], "thresholds_by_type": selected["thresholds_by_type"],
                "selection_metric": config["selection_metric"], "threshold_tie_break": "higher threshold",
                "checkpoint_tie_break": ("higher any character F1, then earlier declared checkpoint" if mode == "legacy_macro" else
                                         "higher macro character F1 (4-type average), then earlier declared checkpoint"),
                "selected_configuration": selected["configuration_winner"], "selection_policy": policy,
                "producer": producer, "validation_metrics": selected["calibrated_metrics"], "official_test_evaluated": False}
    (args.output / "calibration.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    original = args.output / f"validation_predictions_candidate_{selected['candidate_index'] + 1}.jsonl"
    (args.output / "validation_predictions.jsonl").write_bytes(original.read_bytes())
    report.update(status="selected_on_validation", selected=artifact)
    (args.output / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_checkpoint": selected["checkpoint"], "calibration": str(args.output / "calibration.json")}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from None
