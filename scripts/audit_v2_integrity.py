"""Read-only V2 integrity checks; never open the original official-test data.

Recompute already frozen V2 aggregates using independent character sets.
This is an audit, not a new model/decoder selection or official-test run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ragtruth_data.alignment import bio_by_type, tags_to_char_spans, word_tokenize
from ragtruth_data.schema import LABEL_TYPES


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prf(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def independent_scores(records, predictions):
    gold = {r["id"]: r["labels"] for r in records}
    predicted = {r["id"]: r["spans"] for r in predictions}
    assert len(gold) == len(records) and len(predicted) == len(predictions)
    assert gold.keys() == predicted.keys()
    character, exact = {}, {}
    for kind in ("any", *LABEL_TYPES):
        c = [0, 0, 0]
        for rid, spans in gold.items():
            sets = [set(i for s in ss if kind == "any" or s["label_type"] == kind
                        for i in range(s["start"], s["end"])) for ss in (spans, predicted[rid])]
            g, p = sets
            for i, value in enumerate((len(g & p), len(p - g), len(g - p))):
                c[i] += value
        character[kind] = prf(*c)
    character["macro_f1_types"] = sum(character[k]["f1"] for k in LABEL_TYPES) / 4
    character["micro_typed"] = prf(*(sum(character[k][field] for k in LABEL_TYPES)
                                     for field in ("tp", "fp", "fn")))
    for kind in ("any", "typed", *LABEL_TYPES):
        c = [0, 0, 0]
        for rid, spans in gold.items():
            def selected(ss):
                return {(s["start"], s["end"]) if kind == "any" else
                        (s["start"], s["end"], s["label_type"]) for s in ss
                        if kind in {"any", "typed"} or s["label_type"] == kind}
            g, p = selected(spans), selected(predicted[rid])
            for i, value in enumerate((len(g & p), len(p - g), len(g - p))):
                c[i] += value
        exact[kind] = prf(*c)
    exact["macro_f1_types"] = sum(exact[k]["f1"] for k in LABEL_TYPES) / 4
    return {"character": character, "span_exact": exact}


def compare_subset(actual, expected):
    for key, value in actual.items():
        if isinstance(value, dict):
            compare_subset(value, expected[key])
        elif isinstance(value, float):
            assert abs(value - expected[key]) < 1e-12, (key, value, expected[key])
        else:
            assert value == expected[key], (key, value, expected[key])


def audit(root, original_root):
    root, original_root = root.resolve(), original_root.resolve()
    assert root != original_root
    # Do not glob/open original data: only these two explicitly allowed files.
    original = [row for split in ("train", "validation")
                for row in read_rows(original_root / "data/processed" / (split + ".jsonl"))]
    original_by_id = {row["id"]: row for row in original}
    assert len(original_by_id) == len(original)
    rows = {split: read_rows(root / "data/processed" / (split + ".jsonl"))
            for split in ("train", "validation", "test")}
    with (root / "data/splits/split_manifest_v2.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    by_source = {r["source_id"]: r for r in manifest}
    assert len(by_source) == len(manifest)
    all_ids = []
    counts, identities = {}, {}
    for split, records in rows.items():
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))
        all_ids.extend(ids)
        counts[split] = {"responses": len(records), "quality": dict(Counter(r["quality"] for r in records))}
        for r in records:
            old = original_by_id[r["id"]]
            assert r["original_project_split"] == old["split"]
            assert r["study_id"] == "ragtruth-v2-fresh-original-train-holdout"
            assert r["split"] == split == by_source[r["source_id"]]["split"]
            assert old["split"] == ("validation" if split == "validation" else "train")
            assert {k: v for k, v in r.items() if k not in {"split", "study_id", "original_project_split"}} == {
                k: v for k, v in old.items() if k != "split"}
            for s in r["labels"]:
                assert 0 <= s["start"] < s["end"] <= len(r["response"])
                assert s["label_type"] in LABEL_TYPES
                assert "text" not in s or s["text"] == r["response"][s["start"]:s["end"]]
        identities[split] = {
            "response_id": set(ids), "source_id": {r["source_id"] for r in records},
            "source_group": {by_source[r["source_id"]]["group_id"] for r in records},
            "source_text_exact": {r["source_text"] for r in records},
            "source_text_normalized": {" ".join(r["source_text"].casefold().split()) for r in records},
        }
        counts[split].update({k: len(v) for k, v in identities[split].items()})
    assert len(all_ids) == len(set(all_ids)) and set(all_ids) == set(original_by_id)
    overlaps = {}
    for a, b in itertools.combinations(rows, 2):
        overlaps[a + "_vs_" + b] = {k: len(identities[a][k] & identities[b][k]) for k in identities[a]}
        assert not any(overlaps[a + "_vs_" + b].values())
    # Common generated boilerplate across distinct contexts is reported separately.
    answer_overlap = {}
    for a, b in itertools.combinations(rows, 2):
        common = {r["response"] for r in rows[a]} & {r["response"] for r in rows[b]}
        affected = [r for split in (a, b) for r in rows[split] if r["response"] in common]
        answer_overlap[a + "_vs_" + b] = {"distinct_answers": len(common), "responses": len(affected),
                                            "responses_with_gold_spans": sum(bool(r["labels"]) for r in affected),
                                            "good_responses": sum(r["quality"] == "good" for r in affected)}
    protocol = json.loads((root / "docs/V2_FINAL_PROTOCOL.json").read_text(encoding="utf-8"))
    frozen = {}
    for section in ("inputs_sha256", "artifacts_sha256"):
        entries = protocol[section]
        for name, expected in entries.items():
            assert sha(root / name) == expected, name
        frozen[section] = len(entries)
    checkpoints = {}
    for family, model in protocol["models"].items():
        directory = (root / model["checkpoint"]).parent
        paths = sorted(directory.glob("epoch-*/checkpoint.json"))
        assert len(paths) == 3, family
        for path in paths:
            saved = json.loads(path.read_text(encoding="utf-8"))
            sizes = 0
            for name, expected in saved["files_sha256"].items():
                target = path.parent / name
                assert target.resolve().is_relative_to(path.parent.resolve())
                assert sha(target) == expected
                sizes += target.stat().st_size
            checkpoints[family + "/" + path.parent.name] = {
                "manifest": path.relative_to(root).as_posix(), "manifest_sha256": sha(path),
                "verified_files": len(saved["files_sha256"]), "bytes": sizes}
    state = json.loads((root / "outputs/v2_study_v2.json").read_text(encoding="utf-8"))
    metric_checks = {}
    for split, key in (("validation", "dev_comparison"), ("test", "holdout_comparison")):
        good = [r for r in rows[split] if r["quality"] == "good"]
        comparison = json.loads((root / state[key]).read_text(encoding="utf-8"))
        assert sha(root / "data/processed" / (split + ".jsonl")) == comparison["data"]["sha256"]["gold"]
        for family, result in comparison["models"].items():
            path = root / result["predictions_path"].replace("\\", "/")
            assert sha(path) == result["predictions_sha256"]
            predictions = read_rows(path)
            lengths = {r["id"]: len(r["response"]) for r in good}
            for prediction in predictions:
                for s in prediction["spans"]:
                    assert 0 <= s["start"] < s["end"] <= lengths[prediction["id"]]
            metrics = independent_scores(good, predictions)
            compare_subset(metrics, result["metrics"])
            grouped_checks = 0
            for field in ("task_type", "generator"):
                for value in sorted({r[field] for r in good}):
                    subset = [r for r in good if r[field] == value]
                    ids = {r["id"] for r in subset}
                    subpred = [p for p in predictions if p["id"] in ids]
                    compare_subset(independent_scores(subset, subpred), result["metrics"]["breakdowns"][field][value])
                    grouped_checks += 1
            metric_checks[split + "/" + family] = {"responses": len(good), "all_aggregate_counts_and_prf_match": True,
                                                     "task_and_generator_breakdowns_verified": grouped_checks}
    # Gold-derived representation diagnostic on DEV only. Not a model score or
    # a strict mathematical maximum; it checks the existing word/BIO conversion.
    good = [r for r in rows["validation"] if r["quality"] == "good"]
    oracle = []
    for r in good:
        _, offsets = word_tokenize(r["response"])
        columns = bio_by_type(offsets, r["labels"])
        spans = [s for kind in LABEL_TYPES for s in tags_to_char_spans(offsets, columns[kind], kind)]
        oracle.append({"id": r["id"], "spans": spans})
    return {"audit_kind": "read_only_frozen_v2_integrity", "original_official_test_opened": False,
            "counts": counts, "cross_split_overlaps": overlaps, "answer_boilerplate_overlap": answer_overlap,
            "all_original_train_dev_records_preserved": True, "all_gold_bounds_and_text_valid": True,
            "frozen_hashes_verified": frozen, "independent_metric_checks": metric_checks,
            "all_nine_epoch_checkpoints_verified": checkpoints,
            "validation_only_gold_word_bio_roundtrip": independent_scores(good, oracle)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(ROOT, args.original_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "checks": "passed"}))


if __name__ == "__main__":
    main()
