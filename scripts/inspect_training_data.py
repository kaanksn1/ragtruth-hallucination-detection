"""Read-only training-data integrity check; never inspect official test labels.

This script validates the raw training annotations and describes their shape.
It does not clean data, train/select models, or evaluate predictions.
Only test IDs and split membership are used to check group leakage.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TYPES = {
    "Evident Conflict", "Subtle Conflict",
    "Evident Baseless Info", "Subtle Baseless Info",
}


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}, line {line_number}: {exc}") from exc


def lengths(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": values[0],
        "median": statistics.median(values),
        "p95": values[int((len(values) - 1) * 0.95)],
        "max": values[-1],
    }


def inspect(raw_dir):
    source_rows = list(read_jsonl(raw_dir / "source_info.jsonl"))
    sources = {str(row["source_id"]): row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("Duplicate source_id in source_info.jsonl")
    counts = Counter()
    types = Counter()
    quality = Counter()
    tasks = Counter()
    generators = Counter()
    group_splits = defaultdict(set)
    group_sizes = Counter()
    ids = set()
    missing_sources = []
    invalid_spans = []
    response_lengths = []
    prompt_lengths = []
    task_response_lengths = defaultdict(list)
    task_prompt_lengths = defaultdict(list)
    duplicate_responses = Counter()
    duplicate_groups = defaultdict(set)

    for row in read_jsonl(raw_dir / "response.jsonl"):
        response_id = str(row["id"])
        source_id = str(row["source_id"])
        split = row["split"]
        if response_id in ids:
            raise ValueError(f"Duplicate response id: {response_id}")
        ids.add(response_id)
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")
        counts[f"{split}_responses"] += 1
        group_splits[source_id].add(split)
        if source_id not in sources:
            missing_sources.append(response_id)
            continue

        # Do not access response text, quality, labels, or annotations for test.
        if split == "test":
            continue

        group_sizes[source_id] += 1
        source = sources[source_id]
        task = source["task_type"]
        text = row["response"]
        labels = row["labels"]
        counts["train_responses_with_spans"] += bool(labels)
        quality[row.get("quality", "missing")] += 1
        tasks[task] += 1
        generators[row["model"]] += 1
        response_words = len(text.split())
        prompt_words = len(source.get("prompt", "").split())
        response_lengths.append(response_words)
        prompt_lengths.append(prompt_words)
        task_response_lengths[task].append(response_words)
        task_prompt_lengths[task].append(prompt_words)
        duplicate_responses[text] += 1
        duplicate_groups[text].add(source_id)
        spans = []
        for span in labels:
            counts["train_spans"] += 1
            label = span["label_type"]
            types[label] += 1
            start, end = span["start"], span["end"]
            valid_offsets = (
                type(start) is int and type(end) is int
                and 0 <= start < end <= len(text)
            )
            if not valid_offsets or text[start:end] != span["text"]:
                invalid_spans.append({
                    "id": response_id, "start": start, "end": end,
                    "reason": "invalid_offsets" if not valid_offsets else "text_mismatch",
                })
            if label not in EXPECTED_TYPES:
                counts["train_spans_with_unknown_types"] += 1
            if span.get("implicit_true"):
                counts["train_spans_implicit_true"] += 1
            if span.get("due_to_null"):
                counts["train_spans_due_to_null"] += 1
            if valid_offsets:
                spans.append((start, end, label))

        overlap = any(
            max(a[0], b[0]) < min(a[1], b[1])
            for i, a in enumerate(spans) for b in spans[i + 1:]
        )
        counts["train_responses_with_overlapping_spans"] += overlap

    leaked_groups = sorted(k for k, splits in group_splits.items() if len(splits) > 1)
    source_config = json.loads((ROOT / "configs/source.json").read_text(encoding="utf-8"))
    report = {
        "source": source_config,
        "inspection_scope": "Training annotations only; official test IDs/splits only.",
        "length_unit": "Whitespace-separated words, NOT model tokens.",
        "counts": dict(counts),
        "train_source_groups": len(group_sizes),
        "test_source_groups": sum("test" in splits for splits in group_splits.values()),
        "source_groups_shared_between_official_train_test": leaked_groups,
        "missing_source_response_ids": missing_sources,
        "train_responses_by_task": dict(tasks),
        "train_responses_by_generator": dict(generators),
        "train_quality_flags": dict(quality),
        "train_span_types": dict(types),
        "train_invalid_span_count": len(invalid_spans),
        "train_invalid_span_examples": invalid_spans[:20],
        "train_response_words": lengths(response_lengths),
        "train_prompt_words": lengths(prompt_lengths),
        "train_lengths_by_task": {
            task: {"response_words": lengths(task_response_lengths[task]),
                   "prompt_words": lengths(task_prompt_lengths[task])}
            for task in sorted(tasks)
        },
        "train_exact_duplicate_response_excess": sum(n - 1 for n in duplicate_responses.values()),
        "train_exact_response_texts_shared_between_source_groups": sum(
            len(groups) > 1 for groups in duplicate_groups.values()
        ),
        "sha256": {
            name: hashlib.sha256((raw_dir / name).read_bytes()).hexdigest()
            for name in ("response.jsonl", "source_info.jsonl", "LICENSE")
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/training_data_audit.json")
    args = parser.parse_args()
    report = inspect(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "counts": report["counts"],
        "invalid_training_spans": report["train_invalid_span_count"],
        "shared_official_source_groups": len(report["source_groups_shared_between_official_train_test"]),
        "report": str(args.output),
    }, indent=2))
    if (report["missing_source_response_ids"]
            or report["source_groups_shared_between_official_train_test"]
            or report["train_invalid_span_count"]
            or report["counts"].get("train_spans_with_unknown_types", 0)):
        raise SystemExit("Integrity issues found; review report before preprocessing.")


if __name__ == "__main__":
    main()
