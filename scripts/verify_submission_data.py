"""Verify delivered dataset bytes against pre-existing frozen references.

Standard library only; no network, training, prediction or data-label parsing.
All nine dataset files are streamed as binary bytes for SHA-256/size/newline
counts. Only small configuration/protocol/delivery JSON objects are decoded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "data/delivery_manifest.json"
SOURCE = "configs/source.json"
SUMMARY = "data/splits/split_summary_v1.json"
MEMBERSHIP = "data/splits/split_manifest_v1.csv"
FINAL = "docs/FINAL_PROTOCOL.json"
EVALUATION = "configs/final/final_evaluate.protocol.json"
CRF_TEST = "configs/final/predict_crf_source.protocol.json"
REFERENCES = (SOURCE, SUMMARY, MEMBERSHIP, FINAL, EVALUATION, CRF_TEST)
RAW = ("data/raw/response.jsonl", "data/raw/source_info.jsonl", "data/raw/LICENSE")
PREPARED = tuple(f"data/processed/{prefix}{split}.jsonl"
                 for split in ("train", "validation", "test") for prefix in ("", "crf_"))
FILES = RAW + PREPARED
MAX_FILE_BYTES = 100 * 1024 * 1024


def _path(root, relative):
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"missing or nonlocal submission file: {relative}")
    return candidate


def byte_identity(path):
    """Hash/count bytes and lines without decoding or inspecting any JSON row."""
    digest, size, lines, tail = hashlib.sha256(), 0, 0, b""
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
            lines += chunk.count(b"\n")
            tail = chunk[-1:]
    return {"sha256": digest.hexdigest(), "size_bytes": size,
            "line_count": lines + int(bool(size) and tail != b"\n")}


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"metadata must be a JSON object: {path}")
    return value


def _sha(value, description):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"invalid SHA-256 for {description}")
    return value


def verify(root=ROOT, *, require_tracked=False):
    root = Path(root).resolve()
    delivery = _json(_path(root, MANIFEST))
    if delivery.get("format") != "ragtruth-data-delivery-v1":
        raise ValueError("unsupported delivery manifest format")
    entries, reference_hashes = delivery.get("files"), delivery.get("references_sha256")
    if not isinstance(entries, dict) or set(entries) != set(FILES):
        raise ValueError("delivery manifest must declare exactly the nine dataset files")
    if not isinstance(reference_hashes, dict) or set(reference_hashes) != set(REFERENCES):
        raise ValueError("delivery manifest reference set differs from the frozen contract")

    for relative in REFERENCES:
        expected = _sha(reference_hashes[relative], relative)
        if byte_identity(_path(root, relative))["sha256"] != expected:
            raise ValueError(f"reference SHA-256 mismatch: {relative}")
    source, summary, final, evaluation, crf_test = (
        _json(root / relative) for relative in (SOURCE, SUMMARY, FINAL, EVALUATION, CRF_TEST))
    revision = source.get("revision")
    if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or delivery.get("upstream_revision") != revision or summary.get("revision") != revision):
        raise ValueError("upstream revision disagrees across delivery/config/split summary")
    if final.get("status") != "frozen" or evaluation.get("status") != "frozen" or crf_test.get("status") != "frozen":
        raise ValueError("historical references must remain frozen")

    # Cross-bind the metadata to historical scientific/job records. The new
    # delivery manifest does not replace or rewrite any of these references.
    if final.get("code_sha256", {}).get(SOURCE) != reference_hashes[SOURCE]:
        raise ValueError("source config identity differs from frozen scientific protocol")
    for name in (SUMMARY, MEMBERSHIP):
        if final.get("data_sha256", {}).get(name) != reference_hashes[name]:
            raise ValueError(f"split identity differs from frozen scientific protocol: {name}")
    for job in (evaluation, crf_test):
        if (job.get("scientific_decision") != FINAL
                or job.get("scientific_decision_sha256") != reference_hashes[FINAL]):
            raise ValueError("test job differs from frozen scientific decision")
        for name in (SOURCE, SUMMARY, MEMBERSHIP, FINAL):
            if job.get("inputs_sha256", {}).get(name) != reference_hashes[name]:
                raise ValueError(f"test job input identity disagreement: {name}")

    expected_hashes = {}
    for relative in RAW:
        name = Path(relative).name
        expected_hashes[relative] = _sha(source.get("sha256", {}).get(name), relative)
        if final.get("data_sha256", {}).get(relative) != expected_hashes[relative]:
            raise ValueError(f"raw hash disagrees with frozen scientific protocol: {relative}")
        if name != "LICENSE" and summary.get("raw_sha256", {}).get(name) != expected_hashes[relative]:
            raise ValueError(f"raw hash disagrees with split summary: {relative}")
    for relative in PREPARED:
        if relative.endswith("crf_test.jsonl"):
            expected_hashes[relative] = _sha(crf_test.get("inputs_sha256", {}).get(relative), relative)
        elif relative.endswith("/test.jsonl"):
            expected_hashes[relative] = _sha(evaluation.get("inputs_sha256", {}).get(relative), relative)
        else:
            expected_hashes[relative] = _sha(summary.get("processed_sha256", {}).get(Path(relative).name), relative)
            if final.get("data_sha256", {}).get(relative) != expected_hashes[relative]:
                raise ValueError(f"processed hash disagrees with frozen scientific protocol: {relative}")
    for job in (evaluation, crf_test):
        for relative in FILES:
            declared = job.get("inputs_sha256", {}).get(relative)
            if declared is not None and declared != expected_hashes[relative]:
                raise ValueError(f"historical test job data hash disagreement: {relative}")

    counts = summary.get("responses_per_split", {})
    sources = summary.get("sources_per_split", {})
    if (set(counts) != {"train", "validation", "test"}
            or any(type(counts[s]) is not int or counts[s] < 1 for s in counts)):
        raise ValueError("invalid frozen response counts")
    source_total = 0
    for split in ("train", "validation", "test"):
        total = sources.get(split, {}).get("total")
        if type(total) is not int or total < 1:
            raise ValueError("invalid frozen source counts")
        source_total += total
    expected_counts = {"data/raw/response.jsonl": sum(counts.values()),
                       "data/raw/source_info.jsonl": source_total}
    for split in counts:
        for prefix in ("", "crf_"):
            expected_counts[f"data/processed/{prefix}{split}.jsonl"] = counts[split]

    checked, total_size = {}, 0
    for relative in FILES:
        entry = entries[relative]
        if not isinstance(entry, dict) or set(entry) != {"sha256", "size_bytes", "line_count"}:
            raise ValueError(f"invalid delivery entry: {relative}")
        if _sha(entry["sha256"], relative) != expected_hashes[relative]:
            raise ValueError(f"delivery hash differs from pinned historical hash: {relative}")
        if (type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] < MAX_FILE_BYTES
                or type(entry["line_count"]) is not int or entry["line_count"] < 1):
            raise ValueError(f"invalid size/count or file reaches GitHub 100 MiB limit: {relative}")
        actual = byte_identity(_path(root, relative))
        if actual != entry:
            raise ValueError(f"dataset bytes/size/line count mismatch: {relative}")
        if relative in expected_counts and actual["line_count"] != expected_counts[relative]:
            raise ValueError(f"dataset row count differs from frozen split summary: {relative}")
        checked[relative] = actual
        total_size += actual["size_bytes"]
    if delivery.get("total_bytes") != total_size:
        raise ValueError("delivery total_bytes mismatch")
    if require_tracked:
        command = ["git", "-C", str(root), "ls-files", "--error-unmatch", "--",
                   MANIFEST, *REFERENCES, *FILES]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if completed.returncode:
            raise ValueError("one or more dataset/manifest/reference files are not tracked by Git")
    return {"status": "verified", "dataset_files": len(checked), "total_bytes": total_size,
            "upstream_revision": revision, "tracked_check": require_tracked,
            "test_labels_parsed": False, "files": checked}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-tracked", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = verify(args.root, require_tracked=args.require_tracked)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Submission data verification failed: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
