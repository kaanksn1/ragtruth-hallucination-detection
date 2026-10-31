"""Compare fixed predictions with common project metrics, without fitting models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

from ragtruth_data.data import sha256_file
from ragtruth_data.schema import LABEL_TYPES
from ragtruth_eval import METRIC_VERSION, evaluate_records, load_gold, read_predictions
from scripts.run_job import require_frozen_test_job

ROOT = Path(__file__).resolve().parent
METRIC_FILES = ("evaluate.py", "ragtruth_eval/__init__.py", "ragtruth_eval/metrics.py", "ragtruth_eval/io.py",
                "ragtruth_data/data.py", "ragtruth_data/schema.py", "scripts/run_job.py")


def _prediction_argument(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("use MODEL_NAME=predictions.jsonl")
    name, path = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name) or not path:
        raise argparse.ArgumentTypeError("model names need 1-64 letters/numbers/underscore/dot/hyphen")
    return name, Path(path)


def _test_gate(args):
    # The central helper revalidates exact running invocation, committed protocol,
    # clean code and every declared hash. No gold/prediction parsing precedes it.
    inputs = require_frozen_test_job(ROOT, args.invocation, stage="evaluate")
    required = [ROOT / name for name in METRIC_FILES]
    required += [args.gold, args.manifest, args.split_summary, *(path for _, path in args.predictions)]
    for path in required:
        resolved = path.resolve()
        if not resolved.is_relative_to(ROOT) or resolved.relative_to(ROOT).as_posix() not in inputs:
            raise ValueError("frozen evaluation job must hash evaluator code, gold, manifest, summary and predictions")
    for path in required:
        relative = path.resolve().relative_to(ROOT).as_posix()
        if inputs[relative] != sha256_file(path):
            raise ValueError("frozen evaluation input changed: " + relative)


def report_markdown(comparison):
    lines = [f"# RAGTruth project comparison — {comparison['split']}", "",
        f"Metric: `{METRIC_VERSION}`. Quality filter: `good`. Original response character offsets; "
        "whitespace included. Project metrics; not a claim of paper-scorer equivalence.", "",
        "| Model | Any character P | R | F1 | Typed character macro F1 | Typed exact-span F1 | Responses |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, entry in comparison["models"].items():
        metrics = entry["metrics"]
        any_ = metrics["character"]["any"]
        lines.append(f"| {name} | {any_['precision']:.4f} | {any_['recall']:.4f} | {any_['f1']:.4f} | "
                     f"{metrics['character']['macro_f1_types']:.4f} | {metrics['span_exact']['typed']['f1']:.4f} | {metrics['responses']} |")
    lines += ["", "## Type support and performance", "",
              "| Model | Type | Character F1 | Gold characters | Gold responses | Predicted characters | Exact span F1 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for name, entry in comparison["models"].items():
        for kind in LABEL_TYPES:
            row = entry["metrics"]["character"][kind]
            exact = entry["metrics"]["span_exact"][kind]
            lines.append(f"| {name} | {kind} | {row['f1']:.4f} | {row['gold_chars']} | {row['gold_responses']} | "
                         f"{row['predicted_chars']} | {exact['f1']:.4f} |")
    for field in ("task_type", "generator"):
        lines += ["", f"## Breakdown: {field}", "", "| Model | Group | Responses | Any character F1 | Typed macro F1 |",
                  "|---|---|---:|---:|---:|"]
        for name, entry in comparison["models"].items():
            for value, metrics in entry["metrics"]["breakdowns"][field].items():
                safe_value = value.replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {name} | {safe_value} | {metrics['responses']} | {metrics['character']['any']['f1']:.4f} | "
                             f"{metrics['character']['macro_f1_types']:.4f} |")
    lines += ["", "Empty denominators have precision/recall/F1=0. Macro always includes all four fixed types. "
              "Exact duplicate spans are counted once. Exact-span matching preserves original overlapping boundaries, "
              "so a predicted union is not an exact match for two separate raw annotations.", ""]
    return "\n".join(lines)


def evaluate(args):
    if args.split == "test":
        _test_gate(args)
    destinations = [args.output, *(p for p in (args.report, args.csv) if p is not None)]
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("output/report/CSV paths must be distinct")
    for path in destinations:
        if path.exists():
            raise FileExistsError("refusing to overwrite evaluation output: " + str(path))
    if len({name for name, _ in args.predictions}) != len(args.predictions):
        raise ValueError("duplicate model name")
    rows, data = load_gold(args.gold, args.split, manifest=args.manifest, summary=args.split_summary,
                           frozen_test_hash_verified=args.split == "test")
    comparison = {"schema_version": 1, "metric_version": METRIC_VERSION, "split": args.split,
        "created_utc": datetime.now(timezone.utc).isoformat(), "data": data,
        "metric_policy": {"character_unit": "Python Unicode code point", "whitespace": "included",
            "empty_denominator": "zero", "macro_labels": list(LABEL_TYPES),
            "exact_duplicates": "deduplicated within each response", "exact_raw_boundaries": "not merged",
            "implicit_true": "retained", "due_to_null": "retained"},
        "code_sha256": {name: sha256_file(ROOT / name) for name in METRIC_FILES}, "models": {}}
    for name, path in args.predictions:
        comparison["models"][name] = {"predictions_path": str(path), "predictions_sha256": sha256_file(path),
            "metrics": evaluate_records(rows, read_predictions(path))}
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.report:
        args.report.write_text(report_markdown(comparison), encoding="utf-8")
    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["model", "split", "responses", "char_any_precision", "char_any_recall", "char_any_f1",
                             "char_macro_f1_types", "span_exact_typed_f1", *["char_f1_" + k for k in LABEL_TYPES]])
            for name, entry in comparison["models"].items():
                m = entry["metrics"]
                c = m["character"]
                writer.writerow([name, args.split, m["responses"], c["any"]["precision"], c["any"]["recall"],
                                 c["any"]["f1"], c["macro_f1_types"], m["span_exact"]["typed"]["f1"],
                                 *[c[k]["f1"] for k in LABEL_TYPES]])
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/split_manifest_v1.csv"))
    parser.add_argument("--split-summary", type=Path, default=Path("data/splits/split_summary_v1.json"))
    parser.add_argument("--predictions", action="append", type=_prediction_argument, required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    args.gold = args.gold or Path(f"data/processed/{args.split}.jsonl")
    args.invocation = [str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    try:
        result = evaluate(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(json.dumps({"output": str(args.output), "split": args.split, "models": list(result["models"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
