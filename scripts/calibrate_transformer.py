"""Bounded per-type threshold selection on validation only; never retrains weights."""
from __future__ import annotations

import argparse
import gzip
import hashlib
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
from scripts.baseline_crf import char_scores
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


def tune_thresholds(records, cached, grid):
    """Each type maximizes its own character F1; exact ties prefer higher threshold.

    Decode with the real span policy first, including whitespace bridges. This
    prevents an approximate token/character objective from selecting the threshold.
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
        selected[kind] = max(table[kind], key=lambda row: (row["f1"], row["threshold"]))["threshold"]
    return selected, table


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
    require(config.get("selection_metric") == "macro_f1_types", "calibration selection metric must be predeclared macro_f1_types")
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
        thresholds, search = tune_thresholds(labeled, cached, config["threshold_grid"])
        gold = {record["id"]: record["labels"] for record in labeled}
        tuned = [decode_cached(record, cached[record["id"]], thresholds) for record in labeled]
        default_thresholds = {kind: base["threshold"] for kind in LABEL_TYPES}
        default = [decode_cached(record, cached[record["id"]], default_thresholds) for record in labeled]
        candidate = {"candidate_index": index, "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "data_sha256": saved["metadata"]["data_sha256"], "thresholds_by_type": thresholds, "threshold_search": search,
            "default_metrics": char_scores(gold, {row["id"]: row["spans"] for row in default}),
            "calibrated_metrics": char_scores(gold, {row["id"]: row["spans"] for row in tuned}),
            "cache_path": cache_path.relative_to(args.output).as_posix(), "validation_responses": len(labeled)}
        candidates.append(candidate)
        write_jsonl(args.output / f"validation_predictions_candidate_{index + 1}.jsonl", tuned)
        write_jsonl(args.output / f"validation_predictions_default_{index + 1}.jsonl", default)
        print(json.dumps({"candidate": index + 1, "thresholds": thresholds,
                          "macro_f1_types": candidate["calibrated_metrics"]["macro_f1_types"]}), flush=True)
        del cached, tuned, default, labeled, inputs
    selected = max(candidates, key=lambda row: (row["calibrated_metrics"]["macro_f1_types"],
                                               row["calibrated_metrics"]["any"]["f1"], -row["candidate_index"]))
    artifact = {"schema_version": 1, "status": "selected_on_validation", "validation_split": "validation",
                "label_types": list(LABEL_TYPES), "threshold_grid": config["threshold_grid"],
                "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
                "data_sha256": selected["data_sha256"], "thresholds_by_type": selected["thresholds_by_type"],
                "selection_metric": "macro_f1_types", "threshold_tie_break": "higher threshold",
                "checkpoint_tie_break": "higher any character F1, then earlier declared checkpoint",
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
