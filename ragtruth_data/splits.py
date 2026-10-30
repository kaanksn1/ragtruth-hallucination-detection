"""Source-grouped train / validation manifest.

* The official test set is left exactly as published; only its source ids and
  split membership are read.
* Validation is carved out of the official train set by source group: all six
  responses of a source stay on the same side.
* Different source_ids with the same content are grouped (union-find over
  normalised source text, prompt, Yelp business and QA question/passages), so a
  duplicate can never sit on both sides.
* Stratified by (task_type, source_dataset), fixed seed. Labels are never used.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

_WS = re.compile(r"\s+")
MANIFEST_COLUMNS = ("source_id", "group_id", "task_type", "source_dataset", "official_split", "split", "n_responses")


def normalise_text(s: str) -> str:
    return _WS.sub(" ", s.strip().lower())


def _digest(s: str) -> str:
    return hashlib.sha1(normalise_text(s).encode("utf-8")).hexdigest()


def content_keys(src: dict) -> list:
    """Keys under which two sources count as the same content (no labels involved)."""
    si = src["source_info"]
    raw = json.dumps(si, sort_keys=True, ensure_ascii=False) if isinstance(si, dict) else str(si)
    keys = ["info:" + _digest(raw), "prompt:" + _digest(src["prompt"])]
    if isinstance(si, dict) and src["task_type"] == "Data2txt":
        keys.append("business:" + normalise_text(f"{si.get('name')}|{si.get('address')}|{si.get('city')}"))
    if isinstance(si, dict) and src["task_type"] == "QA":
        keys.append("question:" + normalise_text(str(si.get("question", ""))))
        keys.append("passages:" + _digest(str(si.get("passages", ""))))
    return keys


def group_sources(sources: dict) -> dict:
    """source_id -> group_id ('g' + smallest source_id in the group)."""
    parent = {sid: sid for sid in sources}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def order(x):
        return (0, int(x)) if x.isdigit() else (1, x)

    owner = {}
    for sid in sorted(sources, key=order):
        for key in content_keys(sources[sid]):
            if key in owner:
                a, b = find(owner[key]), find(sid)
                if a != b:
                    lo, hi = sorted((a, b), key=order)
                    parent[hi] = lo
            else:
                owner[key] = sid
    return {sid: f"g{find(sid)}" for sid in sources}


def official_split_by_source(responses: list) -> dict:
    seen = defaultdict(set)
    for r in responses:
        seen[str(r["source_id"])].add(r["split"])
    bad = {k: v for k, v in seen.items() if len(v) != 1}
    if bad:
        raise ValueError(f"sources in several official splits: {list(bad)[:5]}")
    return {k: next(iter(v)) for k, v in seen.items()}


def assign_splits(sources: dict, responses: list, val_fraction: float, seed: int) -> tuple[dict, dict]:
    """Return (source_id -> 'train'|'validation'|'test', source_id -> group_id)."""
    official = official_split_by_source(responses)
    group_of = group_sources(sources)
    members = defaultdict(list)
    for sid, g in group_of.items():
        members[g].append(sid)

    split_of, strata = {}, defaultdict(list)
    for g, sids in members.items():
        if any(official.get(s) == "test" for s in sids):
            for s in sids:  # tied to the official test set: never train/validation
                split_of[s] = "test"
            continue
        first = sources[sids[0]]
        strata[(first["task_type"], first["source"])].append(g)

    rng = random.Random(seed)
    for key in sorted(strata):
        groups = sorted(strata[key], key=lambda g: (len(g), g))
        rng.shuffle(groups)
        target = round(val_fraction * sum(len(members[g]) for g in groups))
        taken = 0
        for g in groups:
            side = "validation" if taken < target else "train"
            for s in members[g]:
                split_of[s] = side
            if side == "validation":
                taken += len(members[g])
    return split_of, group_of


def check_no_leakage(split_of: dict, group_of: dict, responses: list) -> dict:
    by_group = defaultdict(set)
    for sid, g in group_of.items():
        by_group[g].add(split_of[sid])
    bad = [g for g, v in by_group.items() if len(v) > 1]
    if bad:
        raise AssertionError(f"source groups split across sides: {bad[:5]}")
    for r in responses:
        if r["split"] == "test" and split_of[str(r["source_id"])] != "test":
            raise AssertionError("an official test response left the test split")
    return {"sources": len(split_of), "groups": len(by_group), "groups_on_two_sides": 0}


def write_manifest(path, sources: dict, responses: list, split_of: dict, group_of: dict) -> None:
    official = official_split_by_source(responses)
    count = defaultdict(int)
    for r in responses:
        count[str(r["source_id"])] += 1
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(MANIFEST_COLUMNS)
        for sid in sorted(sources, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
            s = sources[sid]
            w.writerow([sid, group_of[sid], s["task_type"], s["source"], official[sid], split_of[sid], count[sid]])


def read_manifest(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return {row["source_id"]: row["split"] for row in csv.DictReader(f)}
