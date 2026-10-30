"""Read the verified raw RAGTruth files and join sources, responses and annotations.

Raw files are downloaded and checked by ``scripts/download_data.py`` (1. kişi).
This module only re-checks their SHA-256 against ``configs/source.json`` and
never changes the original text or offsets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator

from .schema import FORMAT_VERSION, LABEL_TYPES

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "configs" / "source.json"


# ---------------------------------------------------------------- files
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw(raw_dir, config_path=SOURCE_CONFIG) -> dict:
    """Check the raw files against the pinned hashes; raise if missing or different."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw_dir = Path(raw_dir)
    hashes = {}
    for name in ("response.jsonl", "source_info.jsonl"):
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `python scripts/download_data.py` first.")
        got = sha256_file(path)
        if got != config["sha256"][name]:
            raise ValueError(f"{name}: SHA-256 differs from configs/source.json")
        hashes[name] = got
    return {"revision": config["revision"], "sha256": hashes}


def read_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{Path(path).name}, line {n}: {exc}") from exc


def write_jsonl(path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_raw(raw_dir) -> tuple[dict, list]:
    raw_dir = Path(raw_dir)
    sources = {}
    for s in read_jsonl(raw_dir / "source_info.jsonl"):
        sid = str(s["source_id"])
        if sid in sources:
            raise ValueError(f"duplicate source_id {sid}")
        sources[sid] = s
    return sources, list(read_jsonl(raw_dir / "response.jsonl"))


# ---------------------------------------------------------------- joining
def build_labels(response_text: str, raw_labels: list, response_id: str) -> list:
    """Normalise one response's annotations without dropping or merging any.

    Every raw annotation is kept (exact duplicates and overlaps included).
    Labels are sorted by (start, end, type) and get label_id 0..n-1; the raw
    position is kept in orig_index. duplicate_of links exact repeats and
    overlaps_with lists labels that share characters.
    """
    labels = []
    for i, lab in enumerate(raw_labels):
        s, e = int(lab["start"]), int(lab["end"])
        if not (0 <= s < e <= len(response_text)):
            raise ValueError(f"response {response_id}: label {s}:{e} out of range")
        if response_text[s:e] != lab["text"]:
            raise ValueError(f"response {response_id}: offsets {s}:{e} do not match the label text")
        if lab["label_type"] not in LABEL_TYPES:
            raise ValueError(f"response {response_id}: unknown label type {lab['label_type']!r}")
        labels.append({
            "start": s,
            "end": e,
            "label_type": lab["label_type"],
            "text": lab["text"],
            "implicit_true": bool(lab.get("implicit_true", False)),
            "due_to_null": bool(lab.get("due_to_null", False)),
            "meta": lab.get("meta", ""),
            "orig_index": i,
        })
    labels.sort(key=lambda x: (x["start"], x["end"], LABEL_TYPES.index(x["label_type"]), x["orig_index"]))
    first = {}
    for k, lab in enumerate(labels):
        key = (lab["start"], lab["end"], lab["label_type"])
        lab["duplicate_of"] = first.get(key)
        first.setdefault(key, k)
    for k, lab in enumerate(labels):
        lab["overlaps_with"] = [
            j for j, o in enumerate(labels)
            if j != k and o["start"] < lab["end"] and lab["start"] < o["end"]
        ]
    return [{"label_id": k, **lab} for k, lab in enumerate(labels)]


def build_record(resp: dict, src: dict, split: str) -> dict:
    labels = build_labels(resp["response"], resp["labels"], str(resp["id"]))
    return {
        "format_version": FORMAT_VERSION,
        "id": str(resp["id"]),
        "source_id": str(resp["source_id"]),
        "split": split,
        "official_split": resp["split"],
        "task_type": src["task_type"],
        "source_dataset": src["source"],
        "generator": resp["model"],
        "temperature": resp["temperature"],
        "quality": resp["quality"],
        # model inputs
        "source_text": src["prompt"],  # verbatim generation prompt: instruction + question + source content
        "response": resp["response"],
        # annotations (targets only, never model input)
        "labels": labels,
        "n_labels": len(labels),
        "has_hallucination": bool(labels),
        # raw structured source, kept for reference
        "source_info": src["source_info"],
    }


def load_split(split: str, processed_dir=ROOT / "data" / "processed", only_good: bool = True) -> list:
    """Load prepared records. only_good=True drops quality != 'good' (see DATA_FORMAT.md)."""
    rows = list(read_jsonl(Path(processed_dir) / f"{split}.jsonl"))
    return [r for r in rows if r["quality"] == "good"] if only_good else rows
