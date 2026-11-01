"""Audit frozen V1 predictions independently; no fitting, inference or selection.

Validation is the default. Test requires --split test --allow-test-report-only,
and only the already-frozen prediction hashes are accepted. The metric math uses
only the standard library, never project scorer imports. --root contains pinned
V1 provenance/raw data; --pack contains saved predictions (defaults to --root).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import string

ROOT = Path(__file__).resolve().parents[1]
TYPES = ("Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info")
QUALITIES = {"good", "incorrect_refusal", "truncated"}
EDGE = set(string.punctuation) - set("$%'\"()")
REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
# Immutable archive identities, not developer-specific prediction paths.
# The expected hash cannot be replaced by editing a supplied comparison JSON.
TRUSTED_REFERENCES = {
    "docs/FINAL_PROTOCOL.json": "19c1c169f6563793fc7dae47856babc9915f98b8850df57cee4917efa39422f3",
    "results/validation_comparison.json": "6f7589c97ae166f74055d42f3b5c541a04eea997f81593964c1e531ce81f3179",
    "results/test_comparison.json": "c2f94261a03653b47365dd371ecd073e843a2e6cdc7fe26b47bb1127e6f1ec0a",
    "configs/final/final_evaluate.protocol.json": "1c602cad045a0ac3642822e7bab110f611f9c6b9e56559aafaac9d4986747dc8",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root, value):
    value = str(value).replace("\\", "/")
    require(".." not in Path(value).parts, "parent traversal is not allowed")
    root = Path(root).resolve()
    path = (root / value).resolve()
    require(path != root and path.is_relative_to(root), "path must remain inside its declared root")
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON object key: " + key)
        result[key] = value
    return result


def decode(text):
    return json.loads(text, object_pairs_hook=_unique_object)


def read_json(path):
    return decode(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line in handle:
            require(bool(line.strip()), "blank JSONL row")
            yield decode(line)


def _trusted(root, name, hashes):
    path = contained(root, name)
    digest = sha256(path)
    require(digest == TRUSTED_REFERENCES[name], "frozen reference hash changed: " + name)
    hashes[name] = digest
    return read_json(path)


def _checked(root, name, expected, hashes):
    require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "missing/invalid expected SHA-256: " + name)
    path = contained(root, name)
    digest = sha256(path)
    require(digest == expected, "input hash changed: " + name)
    hashes[name] = digest
    return path


def provenance(root, split):
    """Verify trusted references, pinned raw bytes and ID-only split manifest."""
    hashes = {}
    decision = _trusted(root, "docs/FINAL_PROTOCOL.json", hashes)
    require(decision.get("status") == "frozen", "scientific decision is not frozen")
    reference = _trusted(root, f"results/{split}_comparison.json", hashes)
    require(reference.get("split") == split and reference.get("metric_version") == "ragtruth-project-eval-v1",
            "reference split/metric version mismatch")
    require(reference.get("data", {}).get("quality_filter") == "good", "reference must use quality=good")
    require(reference.get("metric_policy", {}).get("macro_labels") == list(TYPES), "reference label order differs")
    require(reference["metric_policy"].get("whitespace") == "included", "reference character policy differs")
    source = read_json(_checked(root, "configs/source.json", decision["code_sha256"].get("configs/source.json"), hashes))
    require(source.get("revision") == REVISION, "raw dataset revision is not pinned V1")
    for name in ("response.jsonl", "source_info.jsonl", "LICENSE"):
        expected = source.get("sha256", {}).get(name)
        require(expected == decision["data_sha256"].get("data/raw/" + name), "raw config disagrees with frozen decision")
        _checked(root, "data/raw/" + name, expected, hashes)
    manifest = _checked(root, "data/splits/split_manifest_v1.csv", decision["data_sha256"].get("data/splits/split_manifest_v1.csv"), hashes)
    summary = read_json(_checked(root, "data/splits/split_summary_v1.json", decision["data_sha256"].get("data/splits/split_summary_v1.json"), hashes))
    require(summary.get("revision") == REVISION, "split summary revision differs")
    require(all(summary.get("raw_sha256", {}).get(name) == source["sha256"][name]
                for name in ("response.jsonl", "source_info.jsonl")), "split summary raw identity differs")
    ref_data = reference["data"]["sha256"]
    require(ref_data.get("split_manifest") == hashes["data/splits/split_manifest_v1.csv"]
            and ref_data.get("split_summary") == hashes["data/splits/split_summary_v1.json"], "reference cohort identity differs")
    if split == "validation":
        require(ref_data.get("gold") == summary.get("processed_sha256", {}).get("validation.jsonl"), "reference validation gold identity differs")
    else:
        frozen = _trusted(root, "configs/final/final_evaluate.protocol.json", hashes)
        require(frozen.get("status") == "frozen" and frozen.get("scientific_decision_sha256") == hashes["docs/FINAL_PROTOCOL.json"],
                "frozen test audit protocol differs from scientific decision")
        inputs = frozen.get("inputs_sha256", {})
        require(inputs.get("data/processed/test.jsonl") == ref_data.get("gold"), "reference test gold identity differs from frozen evaluation")
        require(set(reference["models"]) == set(decision["models"]), "test model set differs from frozen decision")
        for model in reference["models"].values():
            name = model["predictions_path"].replace("\\", "/")
            require(inputs.get(name) == model["predictions_sha256"], "test predictions were not frozen for evaluation")
    members, groups = {}, {}
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sid, group, side = row["source_id"], row["group_id"], row["split"]
            require(sid and group and sid not in members and side in {"train", "validation", "test"}, "invalid/duplicate manifest source")
            require(group not in groups or groups[group] == side, "source group crosses manifest splits")
            require(row["official_split"] in {"train", "test"} and int(row["n_responses"]) > 0, "invalid manifest metadata")
            require(side == "test" or row["official_split"] == "train", "official-test source entered development")
            members[sid], groups[group] = row, side
    require(bool(members), "empty source manifest")
    return reference, summary, members, hashes


def _value_end(text, start):
    """Skip a value lexically; skipped annotation values are never JSON-decoded."""
    depth, quoted, escaped = [], False, False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth.append(char)
        elif char in "]}":
            if not depth:
                return index
            require((depth.pop(), char) in {("[", "]"), ("{", "}")}, "unbalanced JSON value")
        elif char == "," and not depth:
            return index
    raise ValueError("unterminated raw JSON object")


def raw_metadata(line):
    """Only decode id/source_id/split; skipped test quality/text/labels stay opaque."""
    text, pos, fields, seen = line.strip(), 1, {}, set()
    require(text.startswith("{"), "raw row must be a JSON object")
    decoder = json.JSONDecoder()
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos < len(text) and text[pos] == "}":
            require(not text[pos+1:].strip(), "trailing raw JSON content")
            break
        key, pos = decoder.raw_decode(text, pos)
        require(isinstance(key, str) and key not in seen, "invalid/duplicate raw object key")
        seen.add(key)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        require(pos < len(text) and text[pos] == ":", "raw JSON key needs colon")
        pos += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        end = _value_end(text, pos)
        if key in {"id", "source_id", "split"}:
            fields[key] = decode(text[pos:end])
        pos = end
        require(pos < len(text), "incomplete raw object")
        if text[pos] == ",":
            pos += 1
        else:
            require(text[pos] == "}", "raw object separator missing")
    require(set(fields) == {"id", "source_id", "split"}, "missing raw identity metadata")
    for name in ("id", "source_id"):
        value = fields[name]
        require(type(value) in (str, int) and str(value), "raw IDs must be nonempty strings or integers")
        fields[name] = str(value)
    require(fields["split"] in {"train", "test"}, "unknown raw official split")
    return fields


def valid_spans(values, text, *, gold=False):
    require(isinstance(values, list), "spans must be a list")
    result = []
    for value in values:
        require(isinstance(value, dict), "span must be an object")
        start, end, label = (value.get(key) for key in ("start", "end", "label_type"))
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(text), "invalid Unicode character span bounds")
        require(label in TYPES, "unknown span label type")
        if gold:
            require(value.get("text") == text[start:end], "raw gold annotation text differs from offsets")
        result.append((start, end, label))
    return result


def raw_gold(root, split, members, reference, summary, *, allow_test=False):
    """Derive the full expected cohort from pinned raw data, never predictions."""
    require(split == "validation" or (split == "test" and allow_test is True),
            "raw test access needs explicit report-only permission")
    gold, seen, counts, excluded = {}, set(), Counter(), Counter()
    file_responses = 0
    with (Path(root) / "data/raw/response.jsonl").open(encoding="utf-8-sig") as handle:
        for line in handle:
            metadata = raw_metadata(line)
            rid, source = metadata["id"], metadata["source_id"]
            require(rid not in seen, "duplicate raw response ID")
            seen.add(rid)
            require(source in members, "raw source missing from manifest")
            member = members[source]
            require(metadata["split"] == member["official_split"], "raw official split differs from manifest")
            counts[source] += 1
            if member["split"] != split:
                continue
            row = decode(line)
            file_responses += 1
            require(isinstance(row.get("quality"), str) and row["quality"] in QUALITIES, "unknown/missing raw quality")
            require(isinstance(row.get("response"), str), "raw response must be text")
            labels = row.get("labels")
            labels = decode(labels) if isinstance(labels, str) else labels
            spans = valid_spans(labels, row["response"], gold=True)
            if row["quality"] != "good":
                excluded[row["quality"]] += 1
                continue
            gold[rid] = {"response": row["response"], "spans": spans}
    require(set(counts) == set(members) and all(counts[sid] == int(member["n_responses"]) for sid, member in members.items()),
            "raw source coverage/count differs from manifest")
    data = reference["data"]
    require(file_responses == summary["responses_per_split"][split] == data["file_responses"], "full raw cohort count differs from frozen reference")
    require(len(gold) == data["included_responses"] and dict(excluded) == data["excluded_by_quality"], "quality-filtered cohort differs from frozen reference")
    require(bool(gold), "empty quality=good cohort")
    return gold


def load_pairs(path, gold):
    predictions = {}
    for row in read_jsonl(path):
        require(isinstance(row, dict) and set(row) == {"id", "spans"}, "prediction needs exactly id and spans")
        rid = row["id"]
        require(isinstance(rid, str) and rid and rid not in predictions, "invalid/duplicate prediction ID")
        require(rid in gold, "prediction ID is outside the quality=good cohort")
        predictions[rid] = valid_spans(row["spans"], gold[rid]["response"])
    require(set(predictions) == set(gold), "prediction IDs must equal the full gold cohort; missing IDs")
    return [(gold[rid]["spans"], predictions[rid], gold[rid]["response"]) for rid in sorted(gold)]


def prf(tp, fp, fn):
    return (tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.,
            2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)


def chars(spans, kind=None):
    return {char for start, end, label in spans if kind is None or label == kind for char in range(start, end)}


def tally(pairs):
    counts = defaultdict(lambda: [0, 0, 0])
    for gold, pred in pairs:
        for kind in (None, *TYPES):
            g, p = chars(gold, kind), chars(pred, kind)
            for index, value in enumerate((len(g & p), len(p - g), len(g - p))):
                counts[kind][index] += value
    return counts


def exact(pairs):
    tp = fp = fn = 0
    for gold, pred in pairs:
        g, p = set(gold), set(pred)
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
    return prf(tp, fp, fn)[2]


def overlap(pairs):
    """Many-to-many coverage diagnostic, NOT one-to-one span matching F1."""
    p_hit = p_all = g_hit = g_all = 0
    for gold, pred in pairs:
        g, p = set(gold), set(pred)
        hit = lambda a, b: a[2] == b[2] and a[0] < b[1] and b[0] < a[1]
        p_all += len(p)
        g_all += len(g)
        p_hit += sum(any(hit(x, y) for y in g) for x in p)
        g_hit += sum(any(hit(x, y) for y in p) for x in g)
    precision, recall = p_hit / p_all if p_all else 0., g_hit / g_all if g_all else 0.
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.


def response_level(pairs):
    tp = sum(bool(g) and bool(p) for g, p in pairs)
    fp = sum(not g and bool(p) for g, p in pairs)
    fn = sum(bool(g) and not p for g, p in pairs)
    tn = len(pairs) - tp - fp - fn
    return dict(zip(("precision", "recall", "f1"), prf(tp, fp, fn))) | {
        "positive_rate": (tp + fn) / len(pairs) if pairs else 0., "accuracy": (tp + tn) / len(pairs) if pairs else 0.}


def strip_edges(span, text):
    start, end, label = span
    while start < end and (text[start].isspace() or text[start] in EDGE):
        start += 1
    while end > start and (text[end-1].isspace() or text[end-1] in EDGE):
        end -= 1
    return (start, end, label) if start < end else None


def _stripped(spans, text):
    return [value for span in spans if (value := strip_edges(span, text)) is not None]


def merge_gaps(spans, text, gap):
    merged = []
    for kind in TYPES:
        current = None
        for start, end, _ in sorted(set(x for x in spans if x[2] == kind)):
            if current and start - current[1] <= gap and not any(c.isalnum() for c in text[current[1]:start]):
                current = (current[0], max(end, current[1]), kind)
            else:
                if current:
                    merged.append(current)
                current = (start, end, kind)
        if current:
            merged.append(current)
    return merged


def summary(pairs):
    counts = tally(pairs)
    typed = [prf(*counts[kind])[2] for kind in TYPES]
    return {"any_p": prf(*counts[None])[0], "any_r": prf(*counts[None])[1], "any_f1": prf(*counts[None])[2],
            "macro_f1": sum(typed) / 4, "per_type_f1": dict(zip(TYPES, typed)),
            "exact_typed_f1": exact(pairs), "overlap_typed_f1": overlap(pairs),
            "predicted_spans": sum(len(set(p)) for _, p in pairs), "gold_spans": sum(len(set(g)) for g, _ in pairs)}


def boundary_diagnostics(triples):
    result = {"pred_stripped_exact": exact([(g, _stripped(p, t)) for g, p, t in triples]),
              "gold_and_pred_stripped_exact": exact([(_stripped(g, t), _stripped(p, t)) for g, p, t in triples])}
    for gap in (1, 3):
        merged = [(g, merge_gaps(p, t, gap)) for g, p, t in triples]
        result[f"merge_gap{gap}_exact"] = exact(merged)
        result[f"merge_gap{gap}_any_f1"] = summary(merged)["any_f1"]
        result[f"merge_gap{gap}_both_stripped_exact"] = exact([
            (_stripped(g, t), _stripped(merge_gaps(p, t, gap), t)) for g, p, t in triples])
    return result


def _check_reference(entry, reference):
    metrics = reference["metrics"]
    checks = [(entry["any_p"], metrics["character"]["any"]["precision"]),
              (entry["any_r"], metrics["character"]["any"]["recall"]),
              (entry["any_f1"], metrics["character"]["any"]["f1"]),
              (entry["macro_f1"], metrics["character"]["macro_f1_types"]),
              (entry["exact_typed_f1"], metrics["span_exact"]["typed"]["f1"])]
    checks += [(entry["per_type_f1"][kind], metrics["character"][kind]["f1"]) for kind in TYPES]
    require(all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in checks), "independent metrics disagree with frozen reference")


def rescore(*, root=ROOT, pack=None, split="validation", allow_test=False, boundary=False, predictions=None):
    require(split in {"validation", "test"}, "only validation or frozen test reporting is allowed")
    require(split != "test" or allow_test is True, "test access needs explicit --allow-test-report-only")
    require(split != "test" or not boundary, "boundary diagnostics are validation-only")
    root, pack = Path(root).resolve(), Path(pack or root).resolve()
    reference, split_summary, members, hashes = provenance(root, split)
    selected = dict(predictions) if predictions is not None else {name: row["predictions_path"] for name, row in reference["models"].items()}
    require(bool(selected) and set(selected) <= set(reference["models"]), "unknown/empty frozen prediction model set")
    if predictions is not None:
        require(len(selected) == len(predictions), "duplicate model argument")
    paths = {}
    for name, value in selected.items():
        path = contained(pack, value)
        require(sha256(path) == reference["models"][name]["predictions_sha256"], "saved prediction hash changed: " + name)
        paths[name] = path
    require(len(set(paths.values())) == len(paths), "models must use distinct prediction files")
    gold = raw_gold(root, split, members, reference, split_summary, allow_test=allow_test)
    models = {}
    for name, path in paths.items():
        triples = load_pairs(path, gold)
        pairs = [(g, p) for g, p, _ in triples]
        entry = summary(pairs) | {"response_level": response_level(pairs)}
        _check_reference(entry, reference["models"][name])
        if boundary:
            entry["boundary"] = boundary_diagnostics(triples)
        models[name] = entry
    # Hashing and decoding are separate passes. Refuse a concurrent input change
    # instead of attributing results to bytes that were checked earlier.
    require(all(sha256(contained(root, name)) == digest for name, digest in hashes.items()),
            "provenance input changed during audit")
    require(all(sha256(path) == reference["models"][name]["predictions_sha256"] for name, path in paths.items()),
            "prediction input changed during audit")
    return {"schema_version": 2, "purpose": "frozen V1 report-only audit; no model/decoder selection",
        "metric_policy": {"whitespace": "included", "character_unit": "Python Unicode code point",
            "macro": "all four types including zero-support types", "exact_duplicates": "deduplicated",
            "implicit_true": "retained", "due_to_null": "retained", "empty_denominator": "zero",
            "overlap_typed_f1": "diagnostic many-to-many typed span coverage; NOT one-to-one matching",
            "boundary": "fixed dev-only diagnostics; empty spans removed; selects nothing"},
        "provenance": {"inputs_sha256": hashes, "predictions": {name: {"path": path.relative_to(pack).as_posix(),
            "sha256": reference["models"][name]["predictions_sha256"]} for name, path in paths.items()},
            "script_sha256": sha256(Path(__file__)), "original_official_test_labels_decoded": split == "test"},
        split: {"responses": len(gold), "data": reference["data"], "reference_metrics_agree": True, "models": models}}


def prediction_argument(value):
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("use FROZEN_MODEL_NAME=PACK_RELATIVE_PATH")
    return name, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="V1 provenance and pinned raw-data root")
    parser.add_argument("--pack", type=Path, help="prediction root; defaults to --root")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test-report-only", action="store_true")
    parser.add_argument("--boundary-diagnostics", action="store_true", help="fixed dev diagnostics, never test tuning")
    parser.add_argument("--predictions", type=prediction_argument, action="append", help="remap/subset frozen models; hashes must match")
    parser.add_argument("--output", required=True, help="new .json file under --root/outputs; never overwrite")
    args = parser.parse_args(argv)
    try:
        require(args.split != "test" or args.allow_test_report_only, "test access needs explicit --allow-test-report-only")
        output = contained(args.root, args.output)
        require(output.is_relative_to(args.root.resolve() / "outputs") and output.suffix == ".json", "output must be .json under --root/outputs")
        if output.exists():
            raise FileExistsError("refusing to overwrite existing independent audit")
        report = rescore(root=args.root, pack=args.pack, split=args.split, allow_test=args.allow_test_report_only,
                         boundary=args.boundary_diagnostics, predictions=args.predictions)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(1, f"Independent audit failed: {exc}\n")
    print(json.dumps({"output": str(output), "split": args.split, "responses": report[args.split]["responses"],
                      "models": list(report[args.split]["models"]), "reference_metrics_agree": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
