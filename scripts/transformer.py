"""Train/predict the person-4 Transformer. Full runs belong on the owner's PC."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data.data import read_jsonl, sha256_file, write_jsonl
from scripts.run_job import require_frozen_test_job


def read_records(processed_dir, split, manifest, with_labels=True, limit=None):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    membership, group_splits = {}, {}
    with Path(manifest).open(encoding="utf-8", newline="") as handle:
        for member in csv.DictReader(handle):
            source, group, side = member["source_id"], member["group_id"], member["split"]
            if not source or not group or side not in {"train", "validation", "test"}:
                raise ValueError("invalid source/group/split in manifest")
            if source in membership:
                raise ValueError("duplicate source_id in manifest")
            if group in group_splits and group_splits[group] != side:
                raise ValueError(f"manifest group {group} occurs in multiple splits")
            membership[source], group_splits[group] = member, side
    result, seen = [], set()
    for row in read_jsonl(Path(processed_dir) / f"{split}.jsonl"):
        if not isinstance(row["id"], str) or row["id"] in seen:
            raise ValueError("response IDs must be unique strings")
        seen.add(row["id"])
        member = membership.get(row["source_id"])
        if member is None or member["split"] != split or row["split"] != split:
            raise ValueError("record split/source_id differs from the committed manifest")
        if row["quality"] != "good":
            continue
        # Do not pass annotation metadata into the feature builder or predictor.
        selected = {key: row[key] for key in ("id", "source_id", "source_text", "response")}
        selected["group_id"] = member["group_id"]
        if with_labels:
            selected["labels"] = row["labels"]
        result.append(selected)
    return result[:limit] if limit else result


def verify_data(config, splits):
    summary_path = Path(config["split_summary"])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = {str(summary_path): sha256_file(summary_path),
                config["split_manifest"]: sha256_file(Path(config["split_manifest"]))}
    for split in splits:
        path = Path(config["processed_dir"]) / f"{split}.jsonl"
        actual = sha256_file(path)
        if summary["processed_sha256"].get(path.name) != actual:
            raise ValueError(f"{path} differs from split summary; run shared data preparation")
        identity[str(path)] = actual
    return identity


def checkpoint_files(path):
    return [p for p in Path(path).rglob("*") if p.is_file()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cmd_train(args):
    import torch
    import transformer as tr

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.device:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    tr.validate_config(config)
    identity = verify_data(config, ("train", "validation"))
    records = {split: read_records(config["processed_dir"], split, config["split_manifest"], limit=limit)
               for split, limit in (("train", args.limit_train), ("validation", args.limit_validation))}
    if not all(records.values()):
        raise ValueError("train and validation must both contain good responses")
    for field in ("id", "source_id", "group_id"):
        if {r[field] for r in records["train"]} & {r[field] for r in records["validation"]}:
            raise ValueError(f"train/validation leakage in {field}")
    out = args.output
    # run_job creates its run directory first. Permit that but never overwrite model outputs.
    if any((out / name).exists() for name in ("checkpoints", "transformer_summary.json", "validation_predictions.jsonl")):
        raise FileExistsError("refusing to overwrite existing Transformer outputs")
    device = tr.choose_device(config["device"])
    tr.autocast_context(device, config["precision"])
    tr.seed_everything(config["seed"])
    tokenizer = tr.AutoTokenizer.from_pretrained(config["model_name"], revision=config["revision"], use_fast=True)
    tr.validate_tokenizer(tokenizer, config)
    model = tr.HallucinationTransformer.from_pretrained(config).to(device)
    tr.validate_tokenizer(tokenizer, config, model.encoder.config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": config, "label_types": list(tr.LABEL_TYPES), "data_sha256": identity,
        "selection_metric": "minimum validation masked BCE over all windows; earliest epoch breaks ties",
        "threshold_note": "fixed config threshold, not tuned on test",
        "counts": {split: len(rows) for split, rows in records.items()},
        "limits": {"train": args.limit_train, "validation": args.limit_validation},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": importlib.metadata.version("transformers"),
                        "tokenizers": importlib.metadata.version("tokenizers"), "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                        (ROOT / "transformer.py", Path(__file__), ROOT / "ragtruth_data/features.py",
                         ROOT / "ragtruth_data/alignment.py", ROOT / "ragtruth_data/schema.py")},
        "epochs": [], "status": "running",
    }
    started, best = time.monotonic(), float("inf")
    for epoch in range(config["epochs"]):
        train = tr.run_epoch(model, tr.make_loader(records["train"], tokenizer, config, True, epoch),
                             device, config, optimizer)
        validation = tr.run_epoch(model, tr.make_loader(records["validation"], tokenizer, config), device, config)
        if train["loss"] is None or validation["loss"] is None:
            raise ValueError("train/validation contains no valid answer tokens")
        row = {"epoch": epoch + 1, "train": train, "validation": validation}
        summary["epochs"].append(row)
        if validation["loss"] < best:
            best = validation["loss"]
            relative = f"checkpoints/epoch-{epoch + 1}"
            tr.save_checkpoint(out / relative, model, tokenizer, config,
                               {"epoch": epoch + 1, "validation_loss": best, "data_sha256": identity})
            summary["best_checkpoint"] = relative
        write_json(out / "transformer_summary.json", summary)
        print(json.dumps(row), flush=True)
    # Release Adam state and the training model before loading the selected checkpoint.
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model, tokenizer, _ = tr.load_checkpoint(out / summary["best_checkpoint"], device)
    write_jsonl(out / "validation_predictions.jsonl",
                tr.predict_records(model, records["validation"], tokenizer, config, device))
    summary["checkpoint_sha256"] = {str(p.relative_to(out)): sha256_file(p)
                                     for p in checkpoint_files(out / summary["best_checkpoint"])}
    summary.update(status="complete", total_seconds=time.monotonic() - started)
    write_json(out / "transformer_summary.json", summary)
    return summary


def cmd_predict(args):
    # Check official-test authorization before reading data, labels or weights.
    frozen_inputs = None
    if args.split == "test":
        if args.limit is not None:
            raise ValueError("official test predictions cannot be limited")
        frozen_inputs = require_frozen_test_job(ROOT, args.invocation)
        required = [args.processed_dir / "test.jsonl", args.manifest, *checkpoint_files(args.checkpoint)]
        for path in required:
            relative = path.resolve().relative_to(ROOT).as_posix()
            if frozen_inputs.get(relative) != sha256_file(path):
                raise ValueError(f"frozen job must declare unchanged input {relative}")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite existing predictions")
    import transformer as tr

    saved = json.loads((args.checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
    config = saved["config"]
    # Runtime device is portable; inference behavior/threshold always comes from the checkpoint.
    device = tr.choose_device(args.device or "auto")
    if args.split == "validation":
        verify_data(dict(config, processed_dir=str(args.processed_dir), split_manifest=str(args.manifest),
                         split_summary=str(args.split_summary)), ("validation",))
    records = read_records(args.processed_dir, args.split, args.manifest, with_labels=False, limit=args.limit)
    model, tokenizer, saved = tr.load_checkpoint(args.checkpoint, device)
    config = dict(saved["config"])
    if device.type == "cpu":
        config["precision"] = "fp32"
    tr.autocast_context(device, config["precision"])
    write_jsonl(args.output, tr.predict_records(model, records, tokenizer, config, device))
    return len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, default=ROOT / "configs/transformer.json")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"))
    train.add_argument("--seed", type=int)

    def positive(value):
        number = int(value)
        if number <= 0:
            raise argparse.ArgumentTypeError("limit must be positive")
        return number

    train.add_argument("--limit-train", type=positive)
    train.add_argument("--limit-validation", type=positive)
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--split", choices=("validation", "test"), required=True)
    predict.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    predict.add_argument("--manifest", type=Path, default=Path("data/splits/split_manifest_v1.csv"))
    predict.add_argument("--split-summary", type=Path, default=Path("data/splits/split_summary_v1.json"))
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--device", choices=("auto", "cpu", "cuda"))
    predict.add_argument("--limit", type=positive)
    args = parser.parse_args(argv)
    args.invocation = [str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    try:
        if args.command == "train":
            summary = cmd_train(args)
            print(json.dumps({"best_checkpoint": str(args.output / summary["best_checkpoint"])}))
        else:
            print(json.dumps({"predictions": cmd_predict(args), "output": str(args.output)}))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
