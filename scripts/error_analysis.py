"""Validation-only support/error report from fixed, commonly scored predictions."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragtruth_data.data import read_jsonl, sha256_file
from ragtruth_data.schema import LABEL_TYPES
from ragtruth_eval import METRIC_VERSION, evaluate_records, load_gold, read_predictions


def chars(spans, kind=None):
    return {i for s in spans if kind is None or s["label_type"] == kind for i in range(s["start"], s["end"])}


def support(rows, word_rows):
    counts = {kind: Counter() for kind in LABEL_TYPES}
    ids = {r["id"] for r in rows}
    for row in rows:
        for kind in LABEL_TYPES:
            labels = [s for s in row["labels"] if s["label_type"] == kind]
            c = counts[kind]
            c["raw_spans"] += len(labels)
            c["responses"] += bool(labels)
            c["characters"] += len(chars(labels))
            c["implicit_true_spans"] += sum(bool(s.get("implicit_true")) for s in labels)
            c["due_to_null_spans"] += sum(bool(s.get("due_to_null")) for s in labels)
    seen, total = set(), 0
    for row in word_rows:
        if row["id"] not in ids:
            continue
        if row["id"] in seen:
            raise ValueError("duplicate word-view response ID")
        seen.add(row["id"])
        total += len(row["tokens"])
        for kind in LABEL_TYPES:
            counts[kind]["positive_word_tokens"] += sum(tag != "O" for tag in row["bio_by_type"][kind])
    if seen != ids:
        raise ValueError("word-view IDs do not match selected records")
    return {kind: dict(c, word_tokens=total,
                      positive_word_fraction=c["positive_word_tokens"] / total if total else 0.0)
            for kind, c in counts.items()}


def select_examples(records, predictions, limit=3):
    """Deterministic failure categories, then largest remaining char error.

    These illustrative worst-case examples are not a representative sample.
    Selection never changes predictions or model settings.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("example limit must be a positive integer")
    predictions = list(predictions)
    evaluate_records(records, predictions, breakdowns=False)
    pred = {p["id"]: p["spans"] for p in predictions}
    scored = []
    for row in records:
        gold, guessed = chars(row["labels"]), chars(pred[row["id"]])
        fp, fn = len(guessed - gold), len(gold - guessed)
        rare = len(chars(row["labels"], "Subtle Conflict") - chars(pred[row["id"]], "Subtle Conflict"))
        typed_error = sum(len(chars(row["labels"], kind) ^ chars(pred[row["id"]], kind)) for kind in LABEL_TYPES)
        scored.append({"id": row["id"], "false_positive_chars": fp, "false_negative_chars": fn,
                       "missed_subtle_conflict_chars": rare, "typed_error_chars": typed_error})
    chosen, seen = [], set()
    priorities = [("missed_subtle_conflict", "missed_subtle_conflict_chars"),
                  ("false_negative", "false_negative_chars"), ("false_positive", "false_positive_chars"),
                  ("typed_or_boundary_error", "typed_error_chars")]
    for category, key in priorities:
        for entry in sorted(scored, key=lambda x: (-x[key], x["id"])):
            if entry[key] and entry["id"] not in seen:
                chosen.append(dict(entry, category=category))
                seen.add(entry["id"])
                break
        if len(chosen) == limit:
            break
    if len(chosen) < limit:
        for entry in sorted(scored, key=lambda x: (-x["typed_error_chars"], x["id"])):
            if entry["typed_error_chars"] and entry["id"] not in seen:
                chosen.append(dict(entry, category="remaining_largest_error"))
                seen.add(entry["id"])
            if len(chosen) == limit:
                break
    by_id = {r["id"]: r for r in records}
    for entry in chosen:
        row = by_id[entry["id"]]
        entry.update(task_type=row["task_type"], generator=row["generator"], response=row["response"],
                     source_excerpt=row["source_text"][:2000], source_excerpt_truncated=len(row["source_text"]) > 2000,
                     gold=row["labels"], predicted=pred[row["id"]])
    return chosen


def fenced(value):
    runs = re.findall(r"`+", value)
    fence = "`" * max(3, 1 + max(map(len, runs), default=0))
    return f"{fence}text\n{value}\n{fence}"


def markdown(report):
    lines = ["# Validation technical error analysis", "",
        "All examples are from validation, selected deterministically by failure category and character error size. "
        "They illustrate failures rather than estimate their frequency. Full metrics are in the common comparison. "
        "No official-test data is read. Source excerpts may be truncated; the demo retains full context.", "",
        "## Class support (quality=good)", "",
        "| Type | Train spans | Validation spans | Train positive word-token % | Train implicit_true spans | Train due_to_null spans |",
        "|---|---:|---:|---:|---:|---:|"]
    for kind in LABEL_TYPES:
        train, validation = report["support"]["train"][kind], report["support"]["validation"][kind]
        lines.append(f"| {kind} | {train['raw_spans']} | {validation['raw_spans']} | "
                     f"{100*train['positive_word_fraction']:.4f} | {train['implicit_true_spans']} | {train['due_to_null_spans']} |")
    lines += ["", "Low support, decision thresholds and incomplete/misleading context are hypotheses for errors, "
              "not established causes. implicit_true annotations concern information absent from source even when "
              "true in the world; due_to_null is retained. Metadata is never supplied to the model."]
    for name, examples in report["models"].items():
        lines += ["", f"## {name}"]
        for example in examples:
            lines += ["", f"### Response {example['id']}: {example['category']}", "",
                f"Task {example['task_type']}; generator {example['generator']}. "
                f"Unmatched characters: FP={example['false_positive_chars']}, FN={example['false_negative_chars']}.",
                "", "Source excerpt:", fenced(example["source_excerpt"]), "", "Response:", fenced(example["response"]),
                "", "| Gold type | Offsets | Annotated text |", "|---|---|---|"]
            for span in example["gold"]:
                text = example["response"][span["start"]:span["end"]]
                safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {span['label_type']} | [{span['start']}, {span['end']}) | {safe} |")
            lines += ["", "Predicted spans:", fenced(json.dumps(example["predicted"], ensure_ascii=False))]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    if comparison["split"] != "validation":
        parser.error("development error analysis accepts validation only")
    if args.output.exists() or args.report.exists() or args.output.resolve() == args.report.resolve():
        parser.error("use distinct new output paths")
    if comparison.get("metric_version") != METRIC_VERSION:
        raise ValueError("comparison metric version differs from current scorer")
    loaded = {split: load_gold(ROOT / f"data/processed/{split}.jsonl", split,
                              manifest=ROOT / "data/splits/split_manifest_v1.csv",
                              summary=ROOT / "data/splits/split_summary_v1.json")
              for split in ("train", "validation")}
    rows = {split: value[0] for split, value in loaded.items()}
    if (comparison.get("data", {}).get("sha256") != loaded["validation"][1]["sha256"]
            or comparison["data"].get("quality_filter") != "good"):
        raise ValueError("comparison does not match current validation data identity")
    summary = json.loads((ROOT / "data/splits/split_summary_v1.json").read_text(encoding="utf-8"))
    for split in rows:
        path = ROOT / f"data/processed/crf_{split}.jsonl"
        if sha256_file(path) != summary["processed_sha256"].get(path.name):
            raise ValueError("word-view support file differs from split summary")
    report = {"split": "validation", "comparison_sha256": sha256_file(args.comparison),
              "data_sha256": {split: value[1]["sha256"] for split, value in loaded.items()},
              "support": {split: support(records, read_jsonl(ROOT / f"data/processed/crf_{split}.jsonl"))
                          for split, records in rows.items()}, "models": {}}
    for name, model in comparison["models"].items():
        path = Path(model["predictions_path"])
        if sha256_file(path) != model["predictions_sha256"]:
            raise ValueError("comparison prediction bytes changed")
        report["models"][name] = select_examples(rows["validation"], read_predictions(path))
    for path in (args.output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": list(report["models"])}))


if __name__ == "__main__":
    main()
