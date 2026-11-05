"""Fit post-feedback supplementary RNN/CNN baselines on frozen V1 train/dev.

No official-test labels or files are accessed by this entry point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from neural_baselines import (ChunkDataset, WordTagger, build_vocabulary, collate_chunks,
                              load_selected_model, make_examples, masked_bce,
                              predict_scores, scores_to_predictions)
from ragtruth_data.data import sha256_file
from ragtruth_data.schema import LABEL_TYPES
from ragtruth_eval import METRIC_VERSION, evaluate_records, load_gold

THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.5)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_predictions(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_threshold(records, examples, scores):
    candidates = []
    for threshold in THRESHOLDS:
        predictions = scores_to_predictions(examples, scores, threshold)
        metrics = evaluate_records(records, predictions)
        candidates.append((threshold, metrics))
    # Stable ties prefer the earlier declared threshold, then earlier epoch.
    selected = max(candidates, key=lambda item: (item[1]["character"]["macro_f1_types"],
                                                  item[1]["character"]["any"]["f1"]))
    return selected, [{"threshold": threshold, "metrics": metrics} for threshold, metrics in candidates]


def load_side(args, split):
    return load_gold(args.processed_dir / f"{split}.jsonl", split,
                     manifest=args.manifest, summary=args.split_summary)


def train(args):
    started = time.perf_counter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA/ROCm device unavailable")
    records, train_identity = load_side(args, "train")
    validation, dev_identity = load_side(args, "validation")
    if set(row["source_id"] for row in records) & set(row["source_id"] for row in validation):
        raise ValueError("train/validation source IDs overlap")
    vocabulary = build_vocabulary(records, args.max_vocabulary, args.min_frequency)
    training_examples = make_examples(records, vocabulary, include_targets=True)
    validation_examples = make_examples(validation, vocabulary, include_targets=False)
    dataset = ChunkDataset(training_examples, args.chunk_size)
    if not dataset:
        raise ValueError("no train words")
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                         generator=generator, collate_fn=collate_chunks)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    configuration = {"schema_version": 1, "study": "post_feedback_supplement_2026-11-05",
        "architecture": args.model, "input_policy": "response_only", "label_types": list(LABEL_TYPES),
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "embedding_size": args.embedding_size, "hidden_size": args.hidden_size, "dropout": args.dropout,
        "learning_rate": args.learning_rate, "optimizer": "Adam", "loss": "masked_unweighted_BCE",
        "chunk_size": args.chunk_size, "chunk_policy": "nonoverlapping_all_words_no_truncation",
        "threshold_grid": list(THRESHOLDS),
        "selection": "validation typed character macro F1, then any character F1, earlier threshold/epoch",
        "official_test_used": False, "vocabulary": vocabulary,
        "train_data_identity": train_identity, "validation_data_identity": dev_identity,
        "source_sha256": {name: sha256_file(ROOT / name) for name in
            ("neural_baselines.py", "scripts/baseline_neural.py", "ragtruth_data/alignment.py",
             "ragtruth_data/schema.py", "ragtruth_eval/metrics.py")},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "platform": platform.platform(), "threads": args.threads},
        "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output_dir / "training_config.json", configuration)
    model = WordTagger(args.model, len(vocabulary), args.embedding_size, args.hidden_size, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history, best_key, best_epoch, selected_threshold = [], None, None, None
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum = word_count = 0
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            mask = batch["mask"].to(device)
            logits = model(batch["ids"].to(device), batch["lengths"])
            loss = masked_bce(logits, batch["targets"].to(device), mask)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(mask.sum())
            loss_sum += float(loss.detach()) * count
            word_count += count
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "steps": len(loader),
                                  "train_loss": loss_sum / word_count}), flush=True)
        checkpoint = args.output_dir / f"epoch-{epoch}.pt"
        with checkpoint.open("xb") as handle:
            torch.save(model.state_dict(), handle)
        scores = predict_scores(model, validation_examples, device=device,
                                batch_size=args.batch_size, chunk_size=args.chunk_size)
        (threshold, metrics), candidates = choose_threshold(validation, validation_examples, scores)
        item = {"epoch": epoch, "train_loss": loss_sum / word_count,
                "threshold": threshold, "metrics": metrics, "threshold_candidates": candidates,
                "checkpoint": checkpoint.name, "checkpoint_sha256": sha256_file(checkpoint),
                "seconds": time.perf_counter() - epoch_started}
        history.append(item)
        write_json(args.output_dir / f"epoch-{epoch}.json", item)
        key = metrics["character"]["macro_f1_types"], metrics["character"]["any"]["f1"]
        if best_key is None or key > best_key:
            best_key, best_epoch, selected_threshold = key, epoch, threshold
        print(json.dumps({"epoch": epoch, "train_loss": item["train_loss"], "threshold": threshold,
                          "macro_f1": key[0], "any_f1": key[1], "seconds": item["seconds"]}), flush=True)
    selected = history[best_epoch - 1]
    selected_config = {**configuration, "selected_epoch": best_epoch, "threshold": selected_threshold,
                        "checkpoint": selected["checkpoint"], "checkpoint_sha256": selected["checkpoint_sha256"]}
    write_json(args.output_dir / "model.json", selected_config)
    # Reload the normal inference artifact; saved validation predictions therefore
    # exercise exactly the selected checkpoint/config path rather than live weights.
    selected_model, saved = load_selected_model(args.output_dir, device)
    scores = predict_scores(selected_model, validation_examples, device=device,
                            batch_size=args.batch_size, chunk_size=args.chunk_size)
    predictions = scores_to_predictions(validation_examples, scores, saved["threshold"])
    replay_metrics = evaluate_records(validation, predictions)
    if replay_metrics != selected["metrics"]:
        raise ValueError("selected checkpoint replay differs from validation selection")
    pred_path = args.output_dir / "validation_predictions.jsonl"
    write_predictions(pred_path, predictions)
    summary = {"schema_version": 1, "study": configuration["study"], "model": args.model,
               "input_policy": "response_only", "split": "validation", "metric_version": METRIC_VERSION,
               "official_test_used": False, "train_responses": len(records),
               "validation_responses": len(validation), "train_words": sum(len(ex["ids"]) for ex in training_examples),
               "validation_words": sum(len(ex["ids"]) for ex in validation_examples),
               "selected_epoch": best_epoch, "threshold": selected_threshold,
               "metrics": replay_metrics, "predictions_sha256": sha256_file(pred_path),
               "model_config_sha256": sha256_file(args.output_dir / "model.json"),
               "history": history, "seconds": time.perf_counter() - started,
               "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"completed": str(args.output_dir), "selected_epoch": best_epoch,
                      "macro_f1": best_key[0], "any_f1": best_key[1], "seconds": summary["seconds"]}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rnn", "cnn"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/split_manifest_v1.csv")
    parser.add_argument("--split-summary", type=Path, default=ROOT / "data/splits/split_summary_v1.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-vocabulary", type=int, default=20000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if (any(value < 1 for value in (args.epochs, args.batch_size, args.chunk_size, args.embedding_size,
                                   args.hidden_size, args.max_vocabulary, args.min_frequency, args.threads))
            or args.hidden_size % 2 or args.max_vocabulary < 2
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.dropout) or not 0 <= args.dropout < 1):
        parser.error("invalid training hyperparameters")
    train(args)


if __name__ == "__main__":
    main()
