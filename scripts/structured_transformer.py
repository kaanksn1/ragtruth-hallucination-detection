"""Train, validation-calibrate, or frozen-holdout-predict the word BIO model."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data.data import sha256_file, write_jsonl
from ragtruth_data.schema import LABEL_TYPES
from scripts.run_job import require_frozen_test_job
from scripts.transformer import read_records, verify_data, verify_validation_identity, checkpoint_files

CODE_PATHS = ("structured_transformer.py", "scripts/structured_transformer.py", "transformer.py",
              "scripts/transformer.py", "scripts/run_job.py", "ragtruth_data/data.py",
              "ragtruth_data/features.py", "ragtruth_data/alignment.py", "ragtruth_data/schema.py",
              "scripts/baseline_crf.py", "requirements-transformer.txt")
BIAS_GRID = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def checkpoint_identity(checkpoint, saved):
    return {"schema_version": 1, "architecture": saved["architecture"],
            "checkpoint_sha256": sha256_file(Path(checkpoint) / "checkpoint.json"),
            "data_sha256": saved["metadata"]["data_sha256"], "label_types": list(LABEL_TYPES),
            "bio_order": ["O", "B", "I"], "aggregation": "context min-positive whole vector, answer-chunk mean"}


def load_calibration(path, checkpoint, saved):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    require(artifact.get("schema_version") == 1 and artifact.get("status") == "selected_on_validation"
            and artifact.get("validation_split") == "validation", "invalid structured calibration artifact")
    require(artifact.get("architecture") == saved["architecture"] and artifact.get("label_types") == list(LABEL_TYPES),
            "calibration architecture/label order mismatch")
    require(artifact.get("checkpoint_sha256") == sha256_file(Path(checkpoint) / "checkpoint.json"),
            "calibration belongs to another checkpoint")
    require(artifact.get("data_sha256") == saved["metadata"].get("data_sha256")
            and artifact.get("data_sha256"), "calibration data identity mismatch")
    require(artifact.get("o_bias_grid") == BIAS_GRID, "calibration requires the frozen nine-value O-bias grid")
    biases = artifact.get("o_bias_by_type")
    require(isinstance(biases, dict) and set(biases) == set(LABEL_TYPES)
            and all(type(value) in (int, float) and value in BIAS_GRID for value in biases.values()),
            "calibration requires four selected grid O biases")
    return artifact


def verify_frozen_inputs(args, inputs):
    # Membership is checked before any test data or checkpoint weight reads.
    required = [*(ROOT / name for name in CODE_PATHS), args.processed_dir / "test.jsonl",
                args.manifest, args.split_summary, args.checkpoint / "checkpoint.json",
                *checkpoint_files(args.checkpoint)]
    if args.calibration is not None:
        required.append(args.calibration)
    for path in required:
        resolved = path.resolve()
        require(resolved.is_relative_to(ROOT) and resolved.relative_to(ROOT).as_posix() in inputs,
                "frozen structured prediction must declare all reused code, checkpoint and holdout inputs")
    for path in required:
        relative = path.resolve().relative_to(ROOT).as_posix()
        require(inputs[relative] == sha256_file(path), f"frozen structured input changed: {relative}")


def verify_prediction_data(config, split, frozen_inputs=None):
    """A frozen holdout can add its hash without mutating training identity.

    Only an ABSENT test key uses the exact frozen-job hash. A present wrong
    value (including null) is an integrity failure, never a fallback request.
    Development prediction always uses the strict shared summary checker.
    """
    if split != "test":
        return verify_data(config, (split,))
    require(frozen_inputs is not None, "holdout data identity requires the frozen job gate first")
    summary = json.loads(Path(config["split_summary"]).read_text(encoding="utf-8"))
    if "test.jsonl" in summary["processed_sha256"]:
        return verify_data(config, ("test",))
    path = (Path(config["processed_dir"]) / "test.jsonl").resolve()
    require(path.is_relative_to(ROOT), "frozen holdout data must remain inside the repository")
    relative = path.relative_to(ROOT).as_posix()
    require(relative in frozen_inputs, "frozen holdout job must declare the materialized test hash")
    actual = sha256_file(path)
    require(actual == frozen_inputs[relative], "materialized holdout differs from its frozen job hash")
    return dict(verify_data(config, ()), **{"test.jsonl": actual})


def cmd_train(args):
    import torch
    import structured_transformer as tr

    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("device", "seed", "epochs"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    tr.validate_config(config)
    identity = verify_data(config, ("train", "validation"))
    records = {split: read_records(config["processed_dir"], split, config["split_manifest"], limit=limit)
               for split, limit in (("train", args.limit_train), ("validation", args.limit_validation))}
    require(all(records.values()), "train and validation require good responses")
    for field in ("id", "source_id", "group_id"):
        require(not ({row[field] for row in records["train"]} & {row[field] for row in records["validation"]}),
                f"train/validation leakage: {field}")
    out = args.output
    require(not any((out / name).exists() for name in ("checkpoints", "structured_summary.json", "validation_predictions.jsonl")),
            "refusing to overwrite structured training outputs")
    device = tr.core.choose_device(config["device"])
    tr.core.autocast_context(device, config["precision"])
    tr.core.seed_everything(config["seed"])
    tokenizer = tr.AutoTokenizer.from_pretrained(config["model_name"], revision=config["revision"], use_fast=True)
    tr.core.validate_tokenizer(tokenizer, config)
    # Always initialize the encoder from the pinned original Hub revision.
    model = tr.StructuredTransformer.from_pretrained(config).to(device)
    tr.core.validate_tokenizer(tokenizer, config, model.encoder.config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    out.mkdir(parents=True, exist_ok=True)
    selection = "minimum validation word/type-normalized CRF NLL; earliest epoch on ties; ALL epochs retained for validation bias calibration"
    summary = {"architecture": tr.ARCHITECTURE, "config": config, "data_sha256": identity,
        "label_types": list(LABEL_TYPES), "bio_order": ["O", "B", "I"], "loss_mode": "crf_nll",
        "selection_metric": selection, "save_all_epochs": True, "saved_checkpoints": [],
        "initialization": "original pinned pretrained encoder; fresh classifier and CRFs; no V1 task weights",
        "counts": {split: len(values) for split, values in records.items()},
        "limits": {"train": args.limit_train, "validation": args.limit_validation},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"), "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_PATHS}, "epochs": [], "status": "running"}
    started, best = time.monotonic(), float("inf")
    for epoch in range(config["epochs"]):
        print(json.dumps({"epoch_started": epoch + 1, "architecture": tr.ARCHITECTURE}), flush=True)
        train = tr.run_epoch(model, tr.make_loader(records["train"], tokenizer, config, True, epoch), device, config, optimizer)
        validation = tr.run_epoch(model, tr.make_loader(records["validation"], tokenizer, config), device, config)
        require(train["loss"] is not None and validation["loss"] is not None, "no supervised original answer words")
        row = {"epoch": epoch + 1, "train": train, "validation": validation}
        summary["epochs"].append(row)
        relative = f"checkpoints/epoch-{epoch + 1}"
        tr.save_checkpoint(out / relative, model, tokenizer, config,
                           {"epoch": epoch + 1, "validation_loss": validation["loss"], "data_sha256": identity,
                            "initialization": summary["initialization"], "loss_mode": "crf_nll", "selection_metric": selection})
        summary["saved_checkpoints"].append(relative)
        if validation["loss"] < best:
            best = validation["loss"]
            summary["best_checkpoint"] = relative
        write_json(out / "structured_summary.json", summary)
        print(json.dumps(row), flush=True)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, tokenizer, _ = tr.load_checkpoint(out / summary["best_checkpoint"], device)
    write_jsonl(out / "validation_predictions.jsonl", tr.predict_records(model, records["validation"], tokenizer, config, device))
    summary["checkpoint_sha256"] = {file.relative_to(out).as_posix(): sha256_file(file)
                                      for relative in summary["saved_checkpoints"] for file in checkpoint_files(out / relative)}
    summary.update(status="complete", total_seconds=time.monotonic() - started)
    write_json(out / "structured_summary.json", summary)
    return summary


def score_and_cache(model, tokenizer, records, config, device, cache_path, checkpoint, saved):
    import structured_transformer as tr
    require(not cache_path.exists() and not cache_path.with_suffix(cache_path.suffix + ".meta.json").exists(),
            "refusing to overwrite word probability cache")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = []
    with cache_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(tr.predict_word_scores(model, records, tokenizer, config, device), 1):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            cached.append(row)
            if index % 250 == 0:
                print(json.dumps({"cached_validation_responses": index, "checkpoint": str(checkpoint)}), flush=True)
    write_json(cache_path.with_suffix(cache_path.suffix + ".meta.json"),
               {"identity": checkpoint_identity(checkpoint, saved), "cache_sha256": sha256_file(cache_path),
                "responses": len(cached), "crf_state": {key: value.detach().cpu().tolist()
                    for key, value in model.crf.state_dict().items()}})
    return cached


def decode_batch(cached, crf, biases, batch_size=32):
    """Pad small CPU batches for calibration; classes remain vectorized."""
    import torch
    from ragtruth_data.alignment import tags_to_char_spans
    result = []
    for start in range(0, len(cached), batch_size):
        group = cached[start:start + batch_size]
        words = max(len(row["word_offsets"]) for row in group)
        values = torch.zeros((len(group), words, 4, 3))
        mask = torch.zeros((len(group), words), dtype=torch.bool)
        for index, row in enumerate(group):
            count = len(row["word_offsets"])
            if count:
                values[index, :count] = torch.tensor(row["probabilities"]).clamp_min(torch.finfo(torch.float32).tiny).log()
                mask[index, :count] = True
        decoded = crf.decode(values, mask, biases)
        for index, row in enumerate(group):
            spans = []
            for type_index, kind in enumerate(LABEL_TYPES):
                tags = [["O", "B", "I"][value] for value in decoded[index, :len(row["word_offsets"]), type_index].tolist()]
                spans.extend(tags_to_char_spans(row["word_offsets"], tags, kind))
            spans.sort(key=lambda span: (span["start"], span["end"], LABEL_TYPES.index(span["label_type"])))
            result.append({"id": row["id"], "spans": spans})
    return result


def select_biases(search):
    return {kind: max(search, key=lambda row: (row["metrics"][kind]["f1"], row["o_bias"]))["o_bias"]
            for kind in LABEL_TYPES}


def select_candidate(candidates):
    return max(candidates, key=lambda row: (row["calibrated_metrics"]["macro_f1_types"],
        row["calibrated_metrics"]["any"]["f1"], -row["epoch"]))


def cmd_calibrate(args):
    import torch
    import structured_transformer as tr
    from scripts.baseline_crf import char_scores

    declaration = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    require(declaration.get("split", "validation") == "validation", "calibration is validation-only")
    require(declaration.get("o_bias_grid", BIAS_GRID) == BIAS_GRID, "the O-bias grid is fixed in advance")
    names = args.checkpoints or declaration.get("checkpoints")
    require(isinstance(names, list) and 1 <= len(names) <= 3 and len(set(map(str, names))) == len(names),
            "declare one to three distinct structured checkpoints")
    out = args.output
    require(not any((out / name).exists() for name in ("calibration.json", "calibration_report.json", "validation_predictions.jsonl", "probabilities")),
            "refusing to overwrite structured calibration outputs")
    out.mkdir(parents=True, exist_ok=True)
    device = tr.core.choose_device(args.device)
    candidates, family_identity, family_parent = [], None, None
    for index, name in enumerate(names):
        checkpoint = (ROOT / name).resolve()
        require(checkpoint.is_relative_to(ROOT / "outputs"), "calibration checkpoints must be inside outputs/")
        saved = tr.read_checkpoint(checkpoint)
        config = dict(saved["config"])
        identity = verify_data(config, ("validation",))
        verify_validation_identity(saved, identity)
        training_identity = saved["metadata"]["data_sha256"]
        if family_identity is None:
            family_identity, family_parent = training_identity, checkpoint.parent
        require(training_identity == family_identity and checkpoint.parent == family_parent,
                "calibration checkpoints must share one training run and data identity")
        records = read_records(config["processed_dir"], "validation", config["split_manifest"], with_labels=True)
        require(bool(records), "no good validation responses")
        inputs = [{key: value for key, value in row.items() if key != "labels"} for row in records]
        model, tokenizer, _ = tr.load_checkpoint(checkpoint, device)
        if device.type == "cpu":
            config["precision"] = "fp32"
        cache_path = out / "probabilities" / f"candidate-{index + 1}.jsonl"
        cached = score_and_cache(model, tokenizer, inputs, config, device, cache_path, checkpoint, saved)
        crf = tr.BatchedBIOCRF()
        crf.load_state_dict({key: value.detach().cpu() for key, value in model.crf.state_dict().items()})
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gold = {row["id"]: row["labels"] for row in records}
        require(set(gold) == {row["id"] for row in cached} and len(cached) == len(gold), "cache ID mismatch")
        search = []
        for bias in BIAS_GRID:
            predictions = decode_batch(cached, crf, bias)
            search.append({"o_bias": bias, "metrics": char_scores(gold, {row["id"]: row["spans"] for row in predictions})})
        biases = select_biases(search)
        tuned = decode_batch(cached, crf, biases)
        default = next(row["metrics"] for row in search if row["o_bias"] == 0)
        candidate = {"candidate_index": index, "epoch": saved["metadata"]["epoch"],
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "checkpoint_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "data_sha256": training_identity, "o_bias_by_type": biases, "o_bias_search": search,
            "default_metrics": default, "calibrated_metrics": char_scores(gold, {row["id"]: row["spans"] for row in tuned}),
            "validation_responses": len(records), "cache_path": cache_path.relative_to(out).as_posix(),
            "cache_sha256": sha256_file(cache_path)}
        candidates.append(candidate)
        write_jsonl(out / f"validation_predictions_candidate_{index + 1}.jsonl", tuned)
        print(json.dumps({"candidate": index + 1, "o_bias_by_type": biases,
                          "macro_f1_types": candidate["calibrated_metrics"]["macro_f1_types"]}), flush=True)
    selected = select_candidate(candidates)
    artifact = {"schema_version": 1, "architecture": tr.ARCHITECTURE, "status": "selected_on_validation",
        "validation_split": "validation", "label_types": list(LABEL_TYPES), "o_bias_grid": BIAS_GRID,
        "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"],
        "data_sha256": selected["data_sha256"], "o_bias_by_type": selected["o_bias_by_type"],
        "selection_metric": "macro_f1_types", "bias_tie_break": "larger O bias (less positive emissions)",
        "checkpoint_tie_break": "higher any character F1, then earlier epoch", "validation_metrics": selected["calibrated_metrics"],
        "holdout_evaluated": False}
    write_json(out / "calibration.json", artifact)
    original = out / f"validation_predictions_candidate_{selected['candidate_index'] + 1}.jsonl"
    (out / "validation_predictions.jsonl").write_bytes(original.read_bytes())
    write_json(out / "calibration_report.json", {"config": {"checkpoints": [str(name) for name in names],
        "split": "validation", "o_bias_grid": BIAS_GRID}, "candidates": candidates, "selected": artifact,
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE_PATHS}, "holdout_evaluated": False})
    return artifact


def cmd_predict(args):
    inputs = None
    if args.split == "test":
        require(args.limit is None, "frozen holdout predictions cannot be limited")
        inputs = require_frozen_test_job(ROOT, args.invocation)
        verify_frozen_inputs(args, inputs)
    require(not args.output.exists(), "refusing to overwrite structured predictions")
    if args.cache_output:
        require(not args.cache_output.exists() and not args.cache_output.with_suffix(args.cache_output.suffix + ".meta.json").exists(),
                "refusing to overwrite structured prediction cache")
    import structured_transformer as tr

    saved = tr.read_checkpoint(args.checkpoint)
    calibration = load_calibration(args.calibration, args.checkpoint, saved) if args.calibration else None
    config = dict(saved["config"])
    identity = verify_prediction_data(dict(config, processed_dir=str(args.processed_dir), split_manifest=str(args.manifest),
                                split_summary=str(args.split_summary)), args.split, inputs)
    if args.split == "validation":
        verify_validation_identity(saved, identity)
    else:
        require(all(identity[key] == saved["metadata"]["data_sha256"].get(key)
                    for key in ("split_manifest", "split_summary")), "holdout split identity differs from training")
    records = read_records(args.processed_dir, args.split, args.manifest, with_labels=False, limit=args.limit)
    device = tr.core.choose_device(args.device or "auto")
    model, tokenizer, saved = tr.load_checkpoint(args.checkpoint, device)
    if device.type == "cpu":
        config["precision"] = "fp32"
    biases = calibration["o_bias_by_type"] if calibration else 0.0
    if args.cache_output:
        cached = score_and_cache(model, tokenizer, records, config, device, args.cache_output, args.checkpoint, saved)
        crf = tr.load_crf(args.checkpoint)
        write_jsonl(args.output, (tr.decode_word_scores(row, crf, biases) for row in cached))
    else:
        write_jsonl(args.output, tr.predict_records(model, records, tokenizer, config, device, biases))
    return len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def positive(text):
        value = int(text)
        if value <= 0:
            raise argparse.ArgumentTypeError("must be positive")
        return value

    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, default=ROOT / "configs/v2/structured.json")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"))
    train.add_argument("--seed", type=int)
    train.add_argument("--epochs", type=positive)
    train.add_argument("--limit-train", type=positive)
    train.add_argument("--limit-validation", type=positive)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--config", type=Path)
    calibrate.add_argument("--checkpoints", type=Path, nargs="+")
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--split", choices=("validation", "test"), required=True)
    predict.add_argument("--processed-dir", type=Path, default=Path("data/processed_v2"))
    predict.add_argument("--manifest", type=Path, default=Path("data/splits/split_manifest_v2.csv"))
    predict.add_argument("--split-summary", type=Path, default=Path("data/splits/split_summary_v2.json"))
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"))
    predict.add_argument("--calibration", type=Path)
    predict.add_argument("--cache-output", type=Path)
    predict.add_argument("--limit", type=positive)
    args = parser.parse_args(argv)
    args.invocation = [str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    try:
        if args.command == "train":
            result = cmd_train(args)
            print(json.dumps({"best_checkpoint": str(args.output / result["best_checkpoint"])}), flush=True)
        elif args.command == "calibrate":
            result = cmd_calibrate(args)
            print(json.dumps({"selected_checkpoint": result["checkpoint"], "calibration": str(args.output / "calibration.json")}), flush=True)
        else:
            print(json.dumps({"predictions": cmd_predict(args), "output": str(args.output)}), flush=True)
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
