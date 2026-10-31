"""Read-only train/validation leakage and shared-response audit; no test labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragtruth_data.data import sha256_file
from ragtruth_eval import load_gold


def normalize(text):
    """Comparison-only whitespace collapse + Unicode casefold; never rewrite data."""
    return " ".join(text.split()).casefold()


def _shared_texts(train, validation, field, normalized=False):
    groups = []
    for rows in (train, validation):
        by_text = defaultdict(list)
        for row in rows:
            value = normalize(row[field]) if normalized else row[field]
            by_text[value].append(row)
        groups.append(by_text)
    overlaps = []
    for text in sorted(set(groups[0]) & set(groups[1])):
        first, second = groups[0][text], groups[1][text]
        overlaps.append({"text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text_length": len(text), "text_excerpt": text[:400], "excerpt_truncated": len(text) > 400,
            "train_count": len(first), "validation_count": len(second),
            "train_response_ids": sorted(row["id"] for row in first),
            "validation_response_ids": sorted(row["id"] for row in second),
            "train_source_ids": sorted({row["source_id"] for row in first}),
            "validation_source_ids": sorted({row["source_id"] for row in second}),
            "all_gold_empty": all(not row["labels"] for row in (*first, *second))})
    return {"distinct_texts": len(overlaps), "train_responses": sum(item["train_count"] for item in overlaps),
            "validation_responses": sum(item["validation_count"] for item in overlaps),
            "all_shared_gold_empty": all(item["all_gold_empty"] for item in overlaps), "items": overlaps}


def audit_development_records(train, validation, group_by_source):
    """Audit already-filtered development records without modifying any row.

    Source/group or response-ID overlap is an integrity failure. The same answer
    across different contexts is reported separately; it is not source leakage.
    """
    train, validation = list(train), list(validation)
    for split, rows in (("train", train), ("validation", validation)):
        if not rows:
            raise ValueError(split + " must not be empty")
        ids = set()
        for row in rows:
            if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
                raise ValueError("response IDs must be unique nonempty strings within each split")
            ids.add(row["id"])
            source = row.get("source_id")
            if not isinstance(source, str) or not source or source not in group_by_source:
                raise ValueError("every source_id must have a manifest group")
            if not isinstance(group_by_source[source], str) or not group_by_source[source]:
                raise ValueError("group IDs must be nonempty strings")
            if not isinstance(row.get("source_text"), str) or not isinstance(row.get("response"), str):
                raise ValueError("source_text and response must be original strings")
            if not isinstance(row.get("labels"), list):
                raise ValueError("labels must be a list")
    source_ids = [{row["source_id"] for row in rows} for rows in (train, validation)]
    groups = [{group_by_source[source] for source in side} for side in source_ids]
    response_ids = [{row["id"] for row in rows} for rows in (train, validation)]
    shared_sources = sorted(source_ids[0] & source_ids[1])
    shared_groups = sorted(groups[0] & groups[1])
    shared_ids = sorted(response_ids[0] & response_ids[1])
    source_exact = _shared_texts(train, validation, "source_text")
    source_normalized = _shared_texts(train, validation, "source_text", True)
    response_exact = _shared_texts(train, validation, "response")
    response_normalized = _shared_texts(train, validation, "response", True)
    bad = bool(shared_sources or shared_groups or shared_ids or source_exact["distinct_texts"]
               or source_normalized["distinct_texts"])
    warnings = []
    if not response_normalized["all_shared_gold_empty"]:
        warnings.append("Some shared answer texts have annotations; review their source contexts separately.")
    return {"schema_version": 1, "status": "failed_source_or_id_leakage" if bad else "passed",
        "splits": ["train", "validation"], "quality_filter": "good",
        "normalization": "comparison only: Unicode whitespace collapse, then casefold",
        "counts": {"train_responses": len(train), "validation_responses": len(validation),
                   "train_sources": len(source_ids[0]), "validation_sources": len(source_ids[1]),
                   "train_groups": len(groups[0]), "validation_groups": len(groups[1])},
        "source_id_overlap": {"count": len(shared_sources), "values": shared_sources},
        "group_id_overlap": {"count": len(shared_groups), "values": shared_groups},
        "response_id_overlap": {"count": len(shared_ids), "values": shared_ids},
        "source_text_overlap": {"exact": source_exact, "normalized": source_normalized},
        "response_text_overlap": {"exact": response_exact, "normalized": response_normalized},
        "response_overlap_policy": "Shared answer text alone is not source leakage. Empty-gold boilerplate remains unchanged.",
        "warnings": warnings, "rows_removed": 0, "official_test_read": False}


def audit_files(processed_dir, manifest, summary):
    processed_dir, manifest, summary = Path(processed_dir), Path(manifest), Path(summary)
    # Fixed development split names: this utility has no official-test option.
    train, train_meta = load_gold(processed_dir / "train.jsonl", "train", manifest=manifest, summary=summary)
    validation, val_meta = load_gold(processed_dir / "validation.jsonl", "validation", manifest=manifest, summary=summary)
    with manifest.open(encoding="utf-8", newline="") as handle:
        group_by_source = {row["source_id"]: row["group_id"] for row in csv.DictReader(handle)}
    report = audit_development_records(train, validation, group_by_source)
    report["input_sha256"] = {"train.jsonl": train_meta["sha256"]["gold"],
        "validation.jsonl": val_meta["sha256"]["gold"], "manifest": sha256_file(manifest), "summary": sha256_file(summary)}
    report["quality_selection"] = {"train": train_meta, "validation": val_meta}
    report["code_sha256"] = {name: sha256_file(ROOT / name) for name in
        ("scripts/audit_development_data.py", "ragtruth_eval/io.py", "ragtruth_data/data.py")}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/split_manifest_v1.csv"))
    parser.add_argument("--split-summary", type=Path, default=Path("data/splits/split_summary_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite development audit")
        report = audit_files(args.processed_dir, args.manifest, args.split_summary)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Development audit failed: {exc}\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
