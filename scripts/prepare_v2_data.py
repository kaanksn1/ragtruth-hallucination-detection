"""Isolated V2 development data with a fresh original-train source-group holdout.

Only the original project's prepared train/validation files and ID-only manifest
are read. Original official-test files are never opened. Fresh study-holdout
labels remain unmaterialized until --include-holdout and a committed V2 freeze.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT.parent.parent
sys.path.insert(0, str(ROOT))
from ragtruth_data.alignment import word_level_example
from ragtruth_data.data import read_jsonl, sha256_file
from ragtruth_data.schema import FORMAT_VERSION
from scripts.run_job import git

STUDY_ID = "ragtruth-v2-fresh-original-train-holdout"
SEED = 20261101
FRACTION = .10
MANIFEST = "data/splits/split_manifest_v2.csv"
SUMMARY = "data/splits/split_summary_v2.json"
MANIFEST_COLUMNS = ("source_id", "group_id", "task_type", "source_dataset", "official_split", "split",
                    "n_responses", "original_project_split", "study_id")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _order(value):
    return (0, int(value)) if value.isdigit() else (1, value)


def assign_v2_manifest(original_members, *, seed=SEED, fraction=FRACTION):
    """Select by ID/task/group only; annotations are not accepted as input."""
    require(type(seed) is int and 0 < fraction < 1, "invalid V2 split settings")
    members, seen, group_sides, tasks = [], set(), {}, defaultdict(set)
    for member in original_members:
        source, group, side = member["source_id"], member["group_id"], member["split"]
        require(source and group and source not in seen and side in {"train", "validation", "test"}, "invalid original manifest")
        seen.add(source)
        require(group not in group_sides or group_sides[group] == side, "original group crosses splits")
        group_sides[group] = side
        if side == "test":
            continue  # ID-only exclusion; no official-test source/response file is opened.
        require(member.get("official_split") == "train", "development member is not from original official train")
        require(member.get("task_type") and int(member["n_responses"]) > 0, "invalid original source metadata")
        members.append(dict(member))
        tasks[group].add(member["task_type"])
    require(members and all(len(values) == 1 for values in tasks.values()), "mixed/missing source-group tasks")
    strata = defaultdict(list)
    for group, values in tasks.items():
        if group_sides[group] == "train":
            strata[next(iter(values))].append(group)
    rng, holdout = random.Random(seed), set()
    for task in sorted(strata):
        groups = sorted(strata[task], key=_order)
        rng.shuffle(groups)
        number = min(len(groups) - 1, max(1, round(fraction * len(groups)))) if len(groups) > 1 else 0
        holdout.update(groups[:number])
    require(holdout, "not enough original-train groups for a fresh holdout")
    output = []
    for member in sorted(members, key=lambda row: _order(row["source_id"])):
        member["original_project_split"] = member["split"]
        member["study_id"] = STUDY_ID
        if member["split"] == "train" and member["group_id"] in holdout:
            member["split"] = "test"
        output.append({name: member[name] for name in MANIFEST_COLUMNS})
    return output


def _manifest_bytes(members):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(members)
    return handle.getvalue().encode("utf-8")


def _jsonl_bytes(rows):
    return ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8")


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _holdout_gate(root):
    name = "docs/V2_FINAL_PROTOCOL.json"
    path = root / name
    require(path.is_file(), "fresh holdout requires committed docs/V2_FINAL_PROTOCOL.json")
    try:
        committed = git(root, "show", "HEAD:" + name)
    except Exception as exc:
        raise ValueError("fresh holdout protocol must be committed before materialization") from exc
    require(path.read_bytes() == committed, "fresh holdout protocol differs from committed bytes")
    require(json.loads(committed).get("status") == "frozen", "fresh holdout protocol must be frozen")
    require((root / MANIFEST).is_file() and (root / SUMMARY).is_file(), "prepare V2 train/dev before freezing holdout")


def prepare_v2(source_root, *, root=ROOT, include_holdout=False):
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    require(root != source_root, "V2 output root must differ from the original project")
    require(not root.is_relative_to(source_root / "data"), "V2 worktree cannot be inside original data")
    if include_holdout:
        _holdout_gate(root)  # Before original prepared annotations are parsed.
    inputs = {"original_train.jsonl": source_root / "data/processed/train.jsonl",
              "original_validation.jsonl": source_root / "data/processed/validation.jsonl",
              "original_manifest.csv": source_root / "data/splits/split_manifest_v1.csv"}
    source_hashes = {name: sha256_file(path) for name, path in inputs.items()}
    with inputs["original_manifest.csv"].open(encoding="utf-8", newline="") as handle:
        members = assign_v2_manifest(list(csv.DictReader(handle)))
    membership = {row["source_id"]: row for row in members}
    records, seen, source_counts = defaultdict(list), set(), Counter()
    for original_side in ("train", "validation"):
        for row in read_jsonl(inputs[f"original_{original_side}.jsonl"]):
            rid, source = row["id"], row["source_id"]
            require(isinstance(rid, str) and rid and rid not in seen, "original response IDs are not unique strings")
            seen.add(rid)
            member = membership.get(source)
            require(member is not None and member["original_project_split"] == row["split"] == original_side,
                    "prepared development record disagrees with original manifest")
            require(row.get("official_split") == "train", "original official-test row encountered in development input")
            source_counts[source] += 1
            side = member["split"]
            if side == "test" and not include_holdout:
                continue  # No holdout annotations, quality/class stats, or example rows are output.
            selected = dict(row, split=side, study_id=STUDY_ID, original_project_split=original_side)
            records[side].append(selected)
    require(set(source_counts) == set(membership), "original development source coverage differs from manifest")
    require(all(source_counts[source] == int(member["n_responses"]) for source, member in membership.items()),
            "original source response count differs from manifest")
    outputs = {root / MANIFEST: _manifest_bytes(members)}
    processed_hashes = {}
    for side in ("train", "validation") + (("test",) if include_holdout else ()):
        rows = sorted(records[side], key=lambda row: _order(row["id"]))
        require(rows, "empty V2 split: " + side)
        full = _jsonl_bytes(rows)
        word = _jsonl_bytes(dict(word_level_example(row), split=side, quality=row["quality"], study_id=STUDY_ID,
                                 original_project_split=row["original_project_split"]) for row in rows)
        outputs[root / f"data/processed/{side}.jsonl"] = full
        outputs[root / f"data/processed/crf_{side}.jsonl"] = word
        if side != "test":
            processed_hashes[f"{side}.jsonl"] = hashlib.sha256(full).hexdigest()
            processed_hashes[f"crf_{side}.jsonl"] = hashlib.sha256(word).hexdigest()
    source_by_split = {side: Counter() for side in ("train", "validation", "test")}
    groups_by_split = {side: set() for side in source_by_split}
    response_by_split = Counter()
    for member in members:
        side = member["split"]
        source_by_split[side][member["task_type"]] += 1
        groups_by_split[side].add(member["group_id"])
        response_by_split[side] += int(member["n_responses"])
    summary = {"format_version": FORMAT_VERSION, "study_id": STUDY_ID, "seed": SEED, "holdout_fraction": FRACTION,
        "stratify_by": ["task_type"], "grouping": "original manifest group_id; source and same-content groups stay together",
        "selection_rule": "shuffle sorted original-train groups by task; round 10%, retaining at least one train group per stratum",
        "source_sha256": source_hashes,
        "sources_per_split": {side: dict(sorted(counts.items()), total=sum(counts.values())) for side, counts in source_by_split.items()},
        "groups_per_split": {side: len(groups) for side, groups in groups_by_split.items()},
        "responses_per_split": {side: response_by_split[side] for side in source_by_split},
        "processed_sha256": processed_hashes, "original_official_test_annotations_read": False,
        "holdout_role": "fresh study holdout carved only from original project's train; not original official test",
        "holdout_materialization_policy": "only after committed frozen docs/V2_FINAL_PROTOCOL.json; no holdout class stats"}
    outputs[root / SUMMARY] = _json_bytes(summary)
    # Validate the entire write set first. Identical reruns are byte-preserving;
    # any changed generated file requires a separate study/output directory.
    for target, payload in outputs.items():
        require(target.resolve().is_relative_to(root) and not target.resolve().is_relative_to(source_root / "data"),
                "output would touch original inputs")
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError("refusing to overwrite conflicting V2 output: " + str(target))
    for target, payload in outputs.items():
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
    require({name: sha256_file(path) for name, path in inputs.items()} == source_hashes, "original inputs changed during preparation")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--include-holdout", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = prepare_v2(args.source_root, include_holdout=args.include_holdout)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"V2 preparation failed: {exc}\n")
    print(json.dumps({"study_id": STUDY_ID, "responses": summary["responses_per_split"],
                      "holdout_materialized": args.include_holdout, "original_official_test_read": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
