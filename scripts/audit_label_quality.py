"""Train/dev label diagnostics with frozen identity checks, never official-test gold.

Note-header differences and repeated responses are not adjudicated errors, IAA
or a model-performance ceiling. This script never changes annotations.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import re
import statistics
import string

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation")
TYPES = ("Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info")
BOUNDARY_KEYS = ("edge_whitespace", "edge_punctuation", "mid_word_start", "mid_word_end", "ends_sentence")
EDGE_PUNCTUATION = set(string.punctuation) - set("$%'\"()")
INPUT_NAMES = ("data/processed/train.jsonl", "data/processed/validation.jsonl", "data/splits/split_manifest_v1.csv",
               "data/splits/split_summary_v1.json", "configs/source.json")
PARSER_VERSION = "explicit-first-line-header-v2"
METHOD_LIMITS = [
    "Format and descriptive patterns are checked; no semantic gold adjudication is performed.",
    "A note header is not an independent annotation. Header/label differences are not proven gold errors or inter-annotator agreement.",
    "Repeated generated responses are not independent annotator samples; pair comparisons share records and are neither IAA nor a model-performance ceiling.",
    "Source ID, exact source text and manifest content group are different grouping rules. A manifest group need not have identical model input; these pair sets overlap and must not be summed.",
    "Missing, ambiguous, misspelled and prose-only note headers stay unparsed; substring mentions of a category are not parsed as declarations.",
    "Raw source hashes are cross-checked as frozen provenance declarations; raw files, including mixed-split response.jsonl, are never opened.",
    "No labels, text, offsets, flags, splits or model settings are changed. Official-test annotation files are never opened.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def input_bytes(root, name):
    path = root / name
    require(path.resolve() == path.absolute() and path.resolve().is_relative_to(root),
            "audit input must be an ordinary path inside the project: " + name)
    return path.read_bytes()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_record(record, split, members):
    require(isinstance(record, dict), "record must be an object")
    for field in ("id", "source_id", "generator", "task_type", "source_dataset"):
        require(isinstance(record.get(field), str) and record[field].strip(), "nonempty string required: " + field)
    require(record.get("format_version") == "ragtruth-data-v1", "unsupported record format")
    require(record.get("split") == split, "record is outside the fixed train/dev split")
    require(record.get("quality") in {"good", "incorrect_refusal", "truncated"}, "unsupported quality")
    sid = record["source_id"]
    require(sid in members and members[sid]["split"] == split, "source differs from manifest split")
    for field in ("official_split", "task_type", "source_dataset"):
        require(record.get(field) == members[sid][field], "record metadata differs from manifest: " + field)
    require(isinstance(record.get("source_text"), str) and isinstance(record.get("response"), str),
            "source_text and response must be strings")
    labels = record.get("labels")
    require(isinstance(labels, list), "labels must be a list")
    require(type(record.get("has_hallucination")) is bool and record["has_hallucination"] == bool(labels),
            "has_hallucination must be a boolean matching labels")
    require(type(record.get("n_labels")) is int and record["n_labels"] == len(labels), "n_labels differs from labels")
    for label in labels:
        require(isinstance(label, dict), "label must be an object")
        start, end = label.get("start"), label.get("end")
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(record["response"]),
                "label bounds must be integer Unicode positions inside response")
        require(label.get("label_type") in TYPES, "unknown label type")
        require(isinstance(label.get("text"), str) and label["text"] == record["response"][start:end],
                "label text differs from response slice")
        for field in ("implicit_true", "due_to_null"):
            require(type(label.get(field)) is bool, "label flag must be an explicit boolean: " + field)
        require(label.get("meta") is None or isinstance(label["meta"], str), "note metadata must be text or null")


def load_audit_inputs(root=ROOT):
    root = Path(root).resolve()
    protocol_bytes = input_bytes(root, "docs/FINAL_PROTOCOL.json")
    protocol = json.loads(protocol_bytes)
    require(isinstance(protocol, dict) and protocol.get("status") == "frozen", "frozen V1 protocol required")
    require(isinstance(protocol.get("data_sha256"), dict) and isinstance(protocol.get("code_sha256"), dict),
            "frozen protocol needs data/code SHA mappings")
    expected = dict(protocol.get("data_sha256", {}))
    expected["configs/source.json"] = protocol.get("code_sha256", {}).get("configs/source.json")
    payloads, hashes = {}, {}
    # Fixed allowlist: never glob data files or read a caller-supplied test split.
    for name in INPUT_NAMES:
        require(valid_hash(expected.get(name)), "missing frozen SHA-256: " + name)
        payloads[name] = input_bytes(root, name)
        hashes[name] = digest(payloads[name])
        require(hashes[name] == expected[name], "input differs from frozen SHA-256: " + name)
    summary = json.loads(payloads["data/splits/split_summary_v1.json"])
    source = json.loads(payloads["configs/source.json"])
    require(isinstance(summary, dict) and isinstance(source, dict), "source and split summary must be objects")
    require(all(isinstance(summary.get(key), dict) for key in
                ("raw_sha256", "processed_sha256", "responses_per_split", "sources_per_split"))
            and isinstance(source.get("sha256"), dict), "source/split identity mappings are missing")
    require(summary.get("format_version") == "ragtruth-data-v1", "unsupported split summary")
    require(isinstance(source.get("revision"), str) and source["revision"] == summary.get("revision"),
            "source revision differs from split summary")
    for name in ("response.jsonl", "source_info.jsonl", "LICENSE"):
        value = source.get("sha256", {}).get(name)
        require(valid_hash(value) and protocol["data_sha256"].get("data/raw/" + name) == value,
                "source SHA declaration differs from frozen protocol: " + name)
        if name != "LICENSE":
            require(summary.get("raw_sha256", {}).get(name) == value, "raw-source SHA differs from split summary")
    for split in SPLITS:
        name = split + ".jsonl"
        require(summary.get("processed_sha256", {}).get(name) == hashes["data/processed/" + name],
                "processed SHA differs from split summary: " + name)
    members, groups = {}, {}
    manifest = csv.DictReader(io.StringIO(payloads["data/splits/split_manifest_v1.csv"].decode("utf-8")))
    required = {"source_id", "group_id", "task_type", "source_dataset", "official_split", "split", "n_responses"}
    require(required <= set(manifest.fieldnames or ()), "split manifest lacks required columns")
    require(len(manifest.fieldnames) == len(set(manifest.fieldnames)), "duplicate manifest columns")
    for member in manifest:
        require(None not in member and all(isinstance(member.get(k), str) and member[k].strip() for k in required),
                "invalid split manifest row")
        sid, group, split = member["source_id"], member["group_id"], member["split"]
        require(split in (*SPLITS, "test") and sid not in members, "invalid or duplicate source in manifest")
        require(member["official_split"] == ("test" if split == "test" else "train"), "invalid official split")
        require(member["task_type"] in {"QA", "Summary", "Data2txt"}, "unknown task in manifest")
        require(member["n_responses"].isdigit(), "invalid manifest response count")
        require(group not in groups or groups[group] == split, "manifest group occurs in multiple splits")
        members[sid], groups[group] = member, split
    records, seen, counts, source_texts, texts_by_split = [], set(), {}, {}, {}
    for split in SPLITS:
        rows = []
        for line_number, line in enumerate(payloads[f"data/processed/{split}.jsonl"].decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_record(record, split, members)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid {split} record at line {line_number}: {exc}") from exc
            require(record["id"] not in seen, "duplicate response ID across train/dev")
            seen.add(record["id"])
            sid = record["source_id"]
            require(sid not in source_texts or source_texts[sid] == record["source_text"], "source ID has inconsistent input text")
            source_texts[sid] = record["source_text"]
            rows.append(record)
        per_source = Counter(r["source_id"] for r in rows)
        expected_sources = {sid for sid, member in members.items() if member["split"] == split}
        require(set(per_source) == expected_sources, "prepared source set differs from manifest")
        require(all(per_source[sid] == int(members[sid]["n_responses"]) for sid in expected_sources),
                "prepared source response count differs from manifest")
        total = summary["responses_per_split"].get(split)
        source_counts = summary["sources_per_split"].get(split)
        require(type(total) is int and total == len(rows), "response count differs from summary")
        require(isinstance(source_counts, dict) and type(source_counts.get("total")) is int
                and source_counts["total"] == len(per_source), "source count differs from summary")
        counts[split] = {"responses": len(rows), "quality": dict(Counter(r["quality"] for r in rows)),
                         "sources": len(per_source), "source_groups": len({members[sid]["group_id"] for sid in per_source})}
        records.extend(r for r in rows if r["quality"] == "good")
        texts_by_split[split] = {r["source_text"] for r in rows}
    overlap = len(texts_by_split["train"] & texts_by_split["validation"])
    normalized = {split: {" ".join(text.casefold().split()) for text in texts}
                  for split, texts in texts_by_split.items()}
    normalized_overlap = len(normalized["train"] & normalized["validation"])
    require(overlap == normalized_overlap == 0, "source text overlaps across train/dev")
    provenance = {"inputs_sha256": hashes, "frozen_protocol_sha256": digest(protocol_bytes),
                  "source_revision": source["revision"], "declared_raw_sha256": source["sha256"],
                  "raw_files_opened": False, "official_test_annotations_opened": False, "counts_before_quality_filter": counts,
                  "cross_split_source_text_overlaps": {"exact": overlap, "casefold_whitespace": normalized_overlap},
                  "code_sha256": digest(Path(__file__).read_bytes())}
    return records, members, provenance


def load_records(root=ROOT):
    return load_audit_inputs(root)[0]


def percent(part, whole):
    return round(100 * part / whole, 2) if whole else None


def boundary_stats(records):
    counts = Counter({key: 0 for key in ("spans", *BOUNDARY_KEYS)})
    for record in records:
        response = record["response"]
        for label in record["labels"]:
            start, end, text = label["start"], label["end"], label["text"]
            stripped = text.strip()
            counts["spans"] += 1
            counts["edge_whitespace"] += text != stripped
            counts["edge_punctuation"] += bool(stripped) and (stripped[0] in EDGE_PUNCTUATION or stripped[-1] in EDGE_PUNCTUATION)
            counts["mid_word_start"] += start > 0 and response[start - 1].isalnum() and response[start].isalnum()
            counts["mid_word_end"] += end < len(response) and response[end - 1].isalnum() and response[end].isalnum()
            counts["ends_sentence"] += text.rstrip()[-1:] in {".", "!", "?"}
    return counts


def type_stats(records):
    words, flags = defaultdict(list), defaultdict(Counter)
    for record in records:
        for label in record["labels"]:
            kind = label["label_type"]
            words[kind].append(len(label["text"].split()))
            flags[kind]["spans"] += 1
            for field in ("implicit_true", "due_to_null"):
                flags[kind][field] += label[field]
    return {kind: {"spans": flags[kind]["spans"],
                   "median_words": statistics.median(words[kind]) if words[kind] else None,
                   "p90_words": sorted(words[kind])[min(len(words[kind]) - 1, int(0.9 * len(words[kind])))] if words[kind] else None,
                   "max_words": max(words[kind]) if words[kind] else None,
                   "implicit_true_pct": percent(flags[kind]["implicit_true"], flags[kind]["spans"]),
                   "due_to_null_pct": percent(flags[kind]["due_to_null"], flags[kind]["spans"])} for kind in TYPES}


def note_type(label):
    """Only explicit first-line category headings are recognized; prose stays unparsed."""
    meta = label.get("meta")
    if not isinstance(meta, str) or not meta.strip():
        return None
    first = meta.strip().splitlines()[0].strip().strip("#*` ")
    first = re.sub(r"^(?:type|label|error type)\s*:\s*", "", first, flags=re.I)
    heading = " ".join(first.partition(":")[0].strip().strip("*` ").rstrip(".").casefold().split())
    canonical = {kind.casefold(): kind for kind in TYPES}
    if heading in canonical:
        return canonical[heading]
    match = re.fullmatch(r"(high|low) intro(?:duction)? of (?:new )?info(?:rmation)?", heading)
    if match:
        return ("Evident" if match[1] == "high" else "Subtle") + " Baseless Info"
    return None


def note_agreement(records):
    confusion, reasons = Counter(), Counter()
    for record in records:
        for label in record["labels"]:
            parsed = note_type(label)
            confusion[label["label_type"], parsed or "unparsed"] += 1
            if parsed is None:
                meta = label.get("meta")
                reason = "missing_or_null" if meta is None else "empty" if not meta.strip() else "unrecognized_or_ambiguous"
                reasons[reason] += 1
    parsed = sum(count for (_, note), count in confusion.items() if note != "unparsed")
    different = [(kind, note, count) for (kind, note), count in confusion.items() if note != "unparsed" and kind != note]
    return {"parser_version": PARSER_VERSION, "parsed": parsed, "unparsed": sum(confusion.values()) - parsed,
            "unparsed_reasons": dict(sorted(reasons.items())), "disagreements": sum(row[2] for row in different),
            "disagreement_pct": percent(sum(row[2] for row in different), parsed),
            "interpretation": "Explicit note-header/label-field differences only; not semantic gold errors or IAA.",
            "pairs": [{"label_type": kind, "note_type": note, "count": count}
                      for kind, note, count in sorted(different, key=lambda row: (-row[2], row[0], row[1]))]}


def covered(record, kind=None):
    return {i for label in record["labels"] if kind is None or label["label_type"] == kind for i in range(label["start"], label["end"])}


def identical_response_agreement(records, grouping="source_id", members=None):
    require(grouping in {"source_id", "source_text", "manifest_group"}, "unknown duplicate grouping")
    groups = defaultdict(list)
    for record in records:
        key = members[record["source_id"]]["group_id"] if grouping == "manifest_group" else record[grouping]
        groups[key, record["response"]].append(record)
    tp = wrong = pairs = binary_same = exact_same = both_positive = cross_source = different_source = 0
    typed = Counter()
    for group in groups.values():
        for left, right in combinations(group, 2):
            pairs += 1
            binary_same += left["has_hallucination"] == right["has_hallucination"]
            signature = lambda row: {(s["start"], s["end"], s["label_type"]) for s in row["labels"]}
            exact_same += signature(left) == signature(right)
            both_positive += left["has_hallucination"] and right["has_hallucination"]
            cross_source += left["source_id"] != right["source_id"]
            different_source += left["source_text"] != right["source_text"]
            a, b = covered(left), covered(right)
            tp, wrong = tp + len(a & b), wrong + len(a ^ b)
            for kind in TYPES:
                a, b = covered(left, kind), covered(right, kind)
                typed[kind, "tp"] += len(a & b)
                typed[kind, "wrong"] += len(a ^ b)
    def f1(overlap, different):
        return round(100 * 2 * overlap / (2 * overlap + different), 2) if overlap + different else None
    return {"grouping": grouping + " + exact response text", "pairs": pairs, "positive_in_both": both_positive,
            "cross_source_id_pairs": cross_source, "different_source_text_pairs": different_source,
            "same_binary_decision_pct": percent(binary_same, pairs), "identical_labels_pct": percent(exact_same, pairs),
            "any_character_agreement_f1": f1(tp, wrong),
            "typed_character_agreement_f1": {kind: f1(typed[kind, "tp"], typed[kind, "wrong"]) for kind in TYPES}}


def build_report(root=ROOT):
    records, members, provenance = load_audit_inputs(root)
    boundaries = boundary_stats(records)
    spans = boundaries["spans"]
    return {"schema_version": 2, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "good-quality train+validation; original official-test annotation files never opened",
            "quality_filter": "good", "label_types": list(TYPES), "provenance": provenance, "method_limits": METHOD_LIMITS,
            "responses": len(records), "spans": spans,
            "boundaries": {key: {"count": boundaries[key], "pct": percent(boundaries[key], spans)} for key in BOUNDARY_KEYS},
            "types": type_stats(records), "note_type_vs_label_type": note_agreement(records),
            "identical_response_pairs": identical_response_agreement(records),
            "identical_source_text_response_pairs": identical_response_agreement(records, "source_text"),
            "same_manifest_group_response_pairs": identical_response_agreement(records, "manifest_group", members)}


def main(argv=None, *, root=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new JSON under results/ or outputs/; never overwritten")
    args = parser.parse_args(argv)
    root = Path(root).resolve()
    output = (root / args.output).resolve()
    try:
        require(output.suffix.lower() == ".json" and any(output.is_relative_to(root / name) for name in ("results", "outputs")),
                "report output must be a JSON file under results/ or outputs/")
        if output.exists():
            raise FileExistsError("refusing to overwrite an existing label audit")
        report = build_report(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps({"output": str(output), "responses": report["responses"], "spans": report["spans"]}))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Label audit failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
