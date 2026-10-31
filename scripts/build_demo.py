"""Build an offline viewer of saved validation predictions; no model inference.

Example:
    python scripts/build_demo.py --predictions Transformer=outputs/run/validation_predictions.jsonl --output artifacts/demo.html
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data.data import read_jsonl, sha256_file
from ragtruth_data.schema import LABEL_TYPES


def validate_spans(spans, response, context):
    """Keep original boundaries and cross-type overlap; discard annotation metadata."""
    if not isinstance(spans, list):
        raise ValueError(f"{context}: spans must be a list")
    result = []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError(f"{context}: each span must be an object")
        start, end, kind = span.get("start"), span.get("end"), span.get("label_type")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(response):
            raise ValueError(f"{context}: invalid original-response character offsets")
        if kind not in LABEL_TYPES:
            raise ValueError(f"{context}: unknown annotation type {kind!r}")
        result.append({"start": start, "end": end, "label_type": kind})
    return result


def normalize_records(rows):
    """Reject mixed/test files before quality filtering or demo subsampling."""
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each validation record must be an object")
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("record IDs must be unique nonempty strings")
        seen.add(rid)
        if row.get("split") != "validation":
            raise ValueError("this demo accepts validation records only; official test is not a demo input")
        if row.get("quality") not in {"good", "truncated", "incorrect_refusal"}:
            raise ValueError(f"record {rid}: unknown or missing quality")
        if row.get("quality") != "good":
            continue
        if not isinstance(row.get("source_text"), str) or not isinstance(row.get("response"), str):
            raise ValueError(f"record {rid}: source_text and response must be strings")
        labels = validate_spans(row.get("labels"), row["response"], f"record {rid}")
        result.append({"id": rid, "split": "validation", "quality": "good",
                       "source_id": str(row.get("source_id", "")),
                       "task_type": str(row.get("task_type", "Unknown")),
                       "generator": str(row.get("generator", "Unknown")),
                       "source_text": row["source_text"], "response": row["response"], "labels": labels})
    if not result:
        raise ValueError("no quality=good validation records to display")
    return result


def normalize_predictions(rows, records, model_name):
    responses = {row["id"]: row["response"] for row in records}
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "spans"}:
            raise ValueError(f"{model_name}: predictions must contain exactly id and spans")
        rid = row["id"]
        if not isinstance(rid, str) or rid in result:
            raise ValueError(f"{model_name}: prediction IDs must be unique strings")
        if rid not in responses:
            raise ValueError(f"{model_name}: extra prediction ID {rid!r}")
        result[rid] = validate_spans(row["spans"], responses[rid], f"{model_name} / {rid}")
    if set(result) != set(responses):
        raise ValueError(f"{model_name}: missing predictions for {len(set(responses) - set(result))} validation IDs")
    return result


def span_segments(response, spans):
    """Lossless Unicode-codepoint segmentation, including multiple active types.

    Segment in Python, where offsets have exactly the dataset's meaning, rather
    than indexing JavaScript UTF-16 strings (which would shift emoji offsets).
    """
    boundaries = sorted({0, len(response), *(s["start"] for s in spans), *(s["end"] for s in spans)})
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        if start == end:
            continue
        kinds = sorted({LABEL_TYPES.index(span["label_type"]) for span in spans
                        if span["start"] < end and start < span["end"]})
        segments.append({"start": start, "end": end, "text": response[start:end], "types": kinds})
    return segments


def panel(response, spans):
    return {"spans": spans, "segments": span_segments(response, spans)}


def exact_signature(spans):
    return {(span["start"], span["end"], span["label_type"]) for span in spans}


def build_payload(records, predictions_by_model, max_examples=None, metadata=None):
    records = normalize_records(records)
    if not predictions_by_model:
        raise ValueError("at least one saved prediction file is required")
    if max_examples is not None and (type(max_examples) is not int or max_examples <= 0):
        raise ValueError("max_examples must be positive")
    normalized = {}
    for name, rows in predictions_by_model.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("model names must be nonempty strings")
        # Validate the COMPLETE prediction set before a display-only limit.
        normalized[name] = normalize_predictions(rows, records, name)
    examples = []
    for row in records[:max_examples]:
        response, gold = row["response"], row["labels"]
        examples.append({key: row[key] for key in ("id", "source_id", "task_type", "generator", "source_text", "response")}
                        | {"response_characters": len(response), "gold": panel(response, gold),
                           "predictions": {name: panel(response, by_id[row["id"]]) for name, by_id in normalized.items()},
                           "exact_match": {name: exact_signature(by_id[row["id"]]) == exact_signature(gold)
                                           for name, by_id in normalized.items()}})
    return {"format_version": 1, "split": "validation", "quality": "good", "label_types": list(LABEL_TYPES),
            "models": list(normalized), "validated_records": len(records), "displayed_records": len(examples),
            "examples": examples, "metadata": metadata or {}}


def safe_json(value):
    """JSON remains parseable while it cannot close an HTML script element."""
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def build_document(payload, title="RAGTruth · Source-grounded hallucination detection"):
    css = (ROOT / "demo/viewer.css").read_text(encoding="utf-8")
    javascript = (ROOT / "demo/viewer.js").read_text(encoding="utf-8")
    template = (ROOT / "demo/viewer.html").read_text(encoding="utf-8")
    # Replace each placeholder once in the template; never interpolate data as JS/HTML.
    replacements = {"__TITLE__": html.escape(title, quote=True), "__STYLE__": css,
                    "__DATA__": safe_json(payload), "__JAVASCRIPT__": javascript}
    import re
    return re.sub("|".join(re.escape(key) for key in replacements),
                  lambda match: replacements[match.group()], template)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT / "data/processed/validation.jsonl")
    parser.add_argument("--predictions", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, help="Display-only limit; complete ID sets are still checked")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite an existing demo")
        models, fingerprints = {}, {}
        for specification in args.predictions:
            name, separator, path = specification.partition("=")
            name = name.strip()
            if not separator or not name or not path or name in models:
                raise ValueError("use distinct model names: --predictions NAME=PATH")
            models[name] = list(read_jsonl(Path(path)))
            fingerprints[name] = sha256_file(Path(path))
        payload = build_payload(list(read_jsonl(args.records)), models, args.max_examples,
                                {"records_sha256": sha256_file(args.records), "predictions_sha256": fingerprints})
        document = build_document(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
        print(json.dumps({"output": str(args.output), "validated_records": payload["validated_records"],
                          "displayed_records": payload["displayed_records"], "models": payload["models"]}))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Demo build failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
