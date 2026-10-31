"""Strict JSONL/manifest readers for the common evaluation contract."""
import csv
import json
from pathlib import Path

from ragtruth_data.data import read_jsonl, sha256_file
from .metrics import _id


def read_predictions(path):
    rows, seen = [], set()
    for row in read_jsonl(path):
        if not isinstance(row, dict) or set(row) != {"id", "spans"}:
            raise ValueError("each prediction must contain exactly id and spans")
        rid = _id(row["id"])
        if rid in seen:
            raise ValueError("duplicate prediction response ID")
        seen.add(rid)
        rows.append(row)
    return rows


def load_gold(path, split, *, manifest=None, summary=None, frozen_test_hash_verified=False):
    """Read all IDs before filtering to quality=good; enforce split/manifest.

    Callers of the library, unlike the gated CLI, are responsible for authorizing
    any official-test access before calling this generic file reader.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("unsupported split")
    path = Path(path)
    identity = {"gold": sha256_file(path)}
    summary_covers_split = False
    if summary is not None:
        data = json.loads(Path(summary).read_text(encoding="utf-8"))
        hashes = data.get("processed_sha256", {})
        name = f"{split}.jsonl"
        expected = hashes.get(name)
        if name not in hashes and split == "test" and frozen_test_hash_verified:
            # The original summary intentionally binds only train/validation.
            # A gated final-test job, rather than a mutated summary, binds test.
            pass
        elif expected != identity["gold"]:
            raise ValueError("gold file checksum differs from split summary")
        else:
            summary_covers_split = True
        identity["split_summary"] = sha256_file(summary)
    membership, groups = {}, {}
    if manifest is not None:
        with Path(manifest).open(encoding="utf-8", newline="") as handle:
            for member in csv.DictReader(handle):
                source, group, side = member["source_id"], member["group_id"], member["split"]
                if not source or not group or side not in {"train", "validation", "test"} or source in membership:
                    raise ValueError("invalid or duplicate source in split manifest")
                if group in groups and groups[group] != side:
                    raise ValueError("manifest group occurs in multiple splits")
                membership[source], groups[group] = side, side
        identity["split_manifest"] = sha256_file(manifest)
    rows, seen, excluded = [], set(), {}
    for record in read_jsonl(path):
        if not isinstance(record, dict):
            raise ValueError("gold records must be objects")
        rid = _id(record.get("id"))
        if rid in seen:
            raise ValueError("duplicate gold response ID")
        seen.add(rid)
        if record.get("split") != split:
            raise ValueError("gold record split differs from requested split")
        if manifest is not None and membership.get(record.get("source_id")) != split:
            raise ValueError("gold source/split differs from manifest")
        quality = record.get("quality")
        if quality not in {"good", "incorrect_refusal", "truncated"}:
            raise ValueError("unknown/missing gold quality flag")
        if quality != "good":
            excluded[quality] = excluded.get(quality, 0) + 1
        else:
            rows.append(record)
    if not rows:
        raise ValueError("no quality=good records to evaluate")
    return rows, {"sha256": identity, "quality_filter": "good", "file_responses": len(seen),
                  "included_responses": len(rows), "excluded_by_quality": excluded,
                  "summary_covers_split": summary_covers_split,
                  "gold_identity_source": "frozen_test_job" if split == "test" and frozen_test_hash_verified
                                           else "split_summary" if summary_covers_split else "caller"}
