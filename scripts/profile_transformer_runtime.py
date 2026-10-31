"""Owner's length-stratified GPU timing; no official test or quality selection.

Inventory all training/validation windows, select input-only length reservoirs,
and time the real model/optimizer on small batches. Disposable profiling updates
are never saved or used as a trained detector. Estimates are not guarantees.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data.data import sha256_file
from scripts.transformer import read_records, verify_data

BOUNDS = (256, 512, 768, 1024, 1536, 2048)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/transformer.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite runtime profile")
    import torch
    import transformer as tr

    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["device"] = "cuda"
    tr.validate_config(config)
    if config["batch_size"] != 1 or config["max_length"] != 2048:
        parser.error("this bounded profiler supports the current batch-1 / 2048 recipe")
    device = tr.choose_device("cuda")
    identity = verify_data(config, ("train", "validation"))
    tr.seed_everything(config["seed"])
    tokenizer = tr.AutoTokenizer.from_pretrained(config["model_name"], revision=config["revision"], use_fast=True)
    tr.validate_tokenizer(tokenizer, config)
    rng = random.Random(config["seed"])
    counts, samples, longest = defaultdict(Counter), defaultdict(lambda: defaultdict(list)), {}
    started = time.monotonic()
    record_counts, empty = {}, Counter()
    for split in ("train", "validation"):
        rows = read_records(config["processed_dir"], split, config["split_manifest"])
        record_counts[split] = len(rows)
        for number, row in enumerate(rows, 1):
            # Length/window selection never inspects labels. They are only used
            # later by the existing training/validation loss on sampled rows.
            features = tr.record_features(row, tokenizer, config, training=split == "train", with_labels=False)
            for index, feature in enumerate(features):
                if not any(feature["response_mask"]):
                    empty[split] += 1
                    continue
                length = len(feature["input_ids"])
                bucket = next(bound for bound in BOUNDS if length <= bound)
                counts[split][bucket] += 1
                entry = (row, index, length)
                reservoir = samples[split][bucket]
                if len(reservoir) < 8:
                    reservoir.append(entry)
                else:
                    candidate = rng.randrange(counts[split][bucket])
                    if candidate < 8:
                        reservoir[candidate] = entry
                if split not in longest or length > longest[split][2]:
                    longest[split] = entry
            if number % 3000 == 0:
                print(json.dumps({"inventory": split, "records": number}), flush=True)
        print(json.dumps({"inventory": split, "counts": counts[split]}), flush=True)
    inventory_seconds = time.monotonic() - started
    # Include the actual longest window, so the pilot also exercises the cap.
    for split, entry in longest.items():
        bucket = next(bound for bound in BOUNDS if entry[2] <= bound)
        samples[split][bucket][-1] = entry

    model = tr.HallucinationTransformer.from_pretrained(config).to(device)
    tr.validate_tokenizer(tokenizer, config, model.encoder.config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    def loader(entries, training):
        # Include tokenization, target construction and transfer just like the
        # normal loader. Rare multi-window rows may be prepared repeatedly here.
        for row, index, _ in entries:
            features = tr.record_features(row, tokenizer, config, training=training)
            yield tr.collate_features([features[index]], tokenizer.pad_token_id)

    warm_entries = next(iter(samples["train"].values()))
    warm = tr.run_epoch(model, loader(warm_entries, True), device, config, optimizer)
    measurements, estimates = {}, {}
    for split in ("train", "validation"):
        training = split == "train"
        measurements[split] = {}
        total_seconds = 0.0
        for bucket in sorted(samples[split]):
            entries = samples[split][bucket]
            # Two fixed repeats; usually 16 windows and two optimizer steps.
            repeated = [entries[i % len(entries)] for i in range(16)]
            result = tr.run_epoch(model, loader(repeated, training), device, config, optimizer if training else None)
            assert result["windows"] == 16 and result["loss"] is not None
            seconds_per_window = result["elapsed_seconds"] / result["windows"]
            total_seconds += counts[split][bucket] * seconds_per_window
            measurements[split][bucket] = dict(result, seconds_per_window=seconds_per_window,
                lengths=[entry[2] for entry in entries], sample_ids=[entry[0]["id"] for entry in entries])
            print(json.dumps({"timing": split, "length_upper": bucket, **result}), flush=True)
        estimates[split] = total_seconds
    hours = config["epochs"] * sum(estimates.values()) / 3600
    report = {"kind": "length_stratified_runtime_projection", "status": "complete",
        "config": config, "data_sha256": identity, "record_counts": record_counts,
        "window_counts": dict(counts), "empty_windows": dict(empty), "inventory_seconds": inventory_seconds,
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "warmup_excluded": warm, "measurements": measurements,
        "projected_seconds_per_epoch": estimates, "projected_epoch_count": config["epochs"],
        "projected_train_and_validation_hours": hours,
        "planning_range_hours": [hours * 0.8, hours * 1.5 + 0.1],
        "range_note": "planning allowance, not a statistical confidence interval",
        "limitations": ["small samples within length bins, not a complete timed epoch",
            "excludes initial loading/checkpoints/final JSON decoding; allow extra time",
            "profiling losses are not model quality results; profiling weights are discarded",
            "no official test data or performance-based model selection"],
        "code_sha256": {name: sha256_file(ROOT / name) for name in
            ("scripts/profile_transformer_runtime.py", "transformer.py", "scripts/transformer.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"projection_hours": hours, "planning_range_hours": report["planning_range_hours"]}), flush=True)


if __name__ == "__main__":
    main()
