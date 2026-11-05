"""Supplementary response-only supervised HMM baseline, added after grading.

Four independent three-state BIO chains retain overlapping annotation types.
Vocabulary/emission/transition estimates use train only. No pretrained model,
annotation metadata, source text, validation vocabulary or test labels are input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data.alignment import tags_to_char_spans, word_level_example
from ragtruth_data.data import sha256_file
from ragtruth_data.schema import BIO, LABEL_TYPES
from ragtruth_eval import evaluate_records, load_gold

FORMAT = "ragtruth-supervised-hmm-v1"
UNKNOWN = ("unk:number", "unk:upper", "unk:title", "unk:punct", "unk:other")


def _settings(alpha, min_count):
    if type(alpha) not in (float, int) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if type(min_count) is not int or min_count < 1:
        raise ValueError("min_count must be a positive integer")


def observation(token, vocabulary):
    """Known lower-case lexical item, otherwise a fixed unknown-word category."""
    normalized = token.casefold()
    if normalized in vocabulary:
        return "w:" + normalized
    if any(c.isdigit() for c in token):
        return UNKNOWN[0]
    if token.isupper() and any(c.isalpha() for c in token):
        return UNKNOWN[1]
    if token.istitle():
        return UNKNOWN[2]
    if not any(c.isalnum() for c in token):
        return UNKNOWN[3]
    return UNKNOWN[4]


def fit(examples, *, alpha=0.1, min_count=2):
    """Count maximum-likelihood supervised states with additive smoothing.

    Illegal BIO starts (I) and O->I transitions have zero probability. End
    probabilities are estimated separately for each state and participate in
    Viterbi decoding. A fixed vocabulary is learned solely from these examples.
    """
    _settings(alpha, min_count)
    if not examples:
        raise ValueError("cannot fit an empty training set")
    ids = [ex["id"] for ex in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate training IDs")
    frequencies = Counter(w.casefold() for ex in examples for w in ex["tokens"])
    vocabulary = sorted(w for w, count in frequencies.items() if count >= min_count)
    known = frozenset(vocabulary)
    symbols = ["w:" + w for w in vocabulary] + list(UNKNOWN)
    models = {}
    nonempty = sum(bool(ex["tokens"]) for ex in examples)
    if not nonempty:
        raise ValueError("training set has no tokens")
    for kind in LABEL_TYPES:
        start, end, state = [0] * 3, [0] * 3, [0] * 3
        transition = [[0] * 3 for _ in BIO]
        emissions = {symbol: [0] * 3 for symbol in symbols}
        for ex in examples:
            tokens = ex["tokens"]
            tags = ex["bio_by_type"][kind]
            if len(tokens) != len(tags) or any(t not in BIO for t in tags):
                raise ValueError("invalid BIO target shape")
            if not tokens:
                continue
            states = [BIO.index(t) for t in tags]
            if states[0] == 2 or any(a == 0 and b == 2 for a, b in zip(states, states[1:])):
                raise ValueError("illegal BIO training sequence")
            start[states[0]] += 1
            end[states[-1]] += 1
            for token, index in zip(tokens, states):
                state[index] += 1
                emissions[observation(token, known)][index] += 1
            for previous, current in zip(states, states[1:]):
                transition[previous][current] += 1
        models[kind] = {"start": start, "end": end, "state": state,
                        "transition": transition, "emission": emissions}
    return {"format": FORMAT, "input": "response_only", "label_types": list(LABEL_TYPES),
            "states": list(BIO), "alpha": float(alpha), "min_count": min_count,
            "vocabulary": vocabulary, "training_responses": len(examples),
            "nonempty_training_responses": nonempty, "models": models}


def validate_model(model):
    if (model.get("format") != FORMAT or model.get("input") != "response_only"
            or model.get("label_types") != list(LABEL_TYPES) or model.get("states") != list(BIO)):
        raise ValueError("unsupported HMM model contract")
    _settings(model["alpha"], model["min_count"])
    vocabulary = model["vocabulary"]
    if (not isinstance(vocabulary, list) or any(not isinstance(w, str) for w in vocabulary)
            or vocabulary != sorted(set(vocabulary))):
        raise ValueError("invalid vocabulary")
    expected_symbols = {"w:" + w for w in vocabulary} | set(UNKNOWN)
    if set(model["models"]) != set(LABEL_TYPES):
        raise ValueError("missing HMM label chain")
    for chain in model["models"].values():
        if set(chain["emission"]) != expected_symbols:
            raise ValueError("invalid emission vocabulary")
        counts = [chain[key] for key in ("start", "end", "state")]
        counts += chain["transition"] + list(chain["emission"].values())
        if (len(chain["transition"]) != 3 or any(not isinstance(row, list) or len(row) != 3
                or any(type(n) is not int or n < 0 for n in row) for row in counts)):
            raise ValueError("invalid HMM count matrix")
        if chain["start"][2] or chain["transition"][0][2]:
            raise ValueError("illegal BIO counts")
        for index in range(3):
            if sum(row[index] for row in chain["emission"].values()) != chain["state"][index]:
                raise ValueError("state/emission count disagreement")
            if sum(chain["transition"][index]) + chain["end"][index] != chain["state"][index]:
                raise ValueError("transition/end count disagreement")
    return model


def _compiled(model):
    """Compile normalized log probabilities once per prediction batch."""
    validate_model(model)
    alpha = model["alpha"]
    count_symbols = len(model["vocabulary"]) + len(UNKNOWN)
    compiled = {}
    for kind, chain in model["models"].items():
        starts = sum(chain["start"])
        initial = [math.log((chain["start"][i] + alpha) / (starts + 2 * alpha))
                   if i != 2 else -math.inf for i in range(3)]
        transition, end = [], []
        for i in range(3):
            # Outgoing transitions plus an explicit END outcome form one distribution.
            allowed = (0, 1) if i == 0 else (0, 1, 2)
            total = chain["state"][i] + (len(allowed) + 1) * alpha
            transition.append([math.log((chain["transition"][i][j] + alpha) / total)
                               if j in allowed else -math.inf for j in range(3)])
            end.append(math.log((chain["end"][i] + alpha) / total))
        emission = {symbol: [math.log((counts[i] + alpha) /
                                    (chain["state"][i] + count_symbols * alpha)) for i in range(3)]
                    for symbol, counts in chain["emission"].items()}
        compiled[kind] = {"initial": initial, "transition": transition,
                          "end": end, "emission": emission}
    return frozenset(model["vocabulary"]), compiled


def viterbi(symbols, chain):
    if not symbols:
        return []
    initial, transition, emission = chain["initial"], chain["transition"], chain["emission"]
    score = [initial[i] + emission[symbols[0]][i] for i in range(3)]
    back = []
    for symbol in symbols[1:]:
        current, pointers = [], []
        for j in range(3):
            previous = max(range(3), key=lambda i: score[i] + transition[i][j])
            current.append(score[previous] + transition[previous][j] + emission[symbol][j])
            pointers.append(previous)
        score = current
        back.append(pointers)
    best = max(range(3), key=lambda i: score[i] + chain["end"][i])
    path = [best]
    for pointers in reversed(back):
        path.append(pointers[path[-1]])
    return [BIO[index] for index in reversed(path)]


def predict(model, examples):
    known, compiled = _compiled(model)
    predictions = []
    seen = set()
    for example in examples:
        rid = example["id"]
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("invalid or duplicate prediction ID")
        seen.add(rid)
        if len(example["tokens"]) != len(example["offsets"]):
            raise ValueError("tokens/offsets disagree")
        symbols = [observation(token, known) for token in example["tokens"]]
        spans = []
        for kind in LABEL_TYPES:
            spans.extend(tags_to_char_spans(example["offsets"], viterbi(symbols, compiled[kind]), kind))
        predictions.append({"id": rid, "spans": sorted(spans, key=lambda s:
                            (s["start"], s["end"], LABEL_TYPES.index(s["label_type"])))})
    return predictions


def load_records(processed_dir, split, manifest, summary):
    if split not in ("train", "validation"):
        raise ValueError("supplementary HMM accepts only train/validation, not official test")
    return load_gold(Path(processed_dir) / f"{split}.jsonl", split,
                     manifest=manifest, summary=summary)


def _write_json(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _write_predictions(path, predictions):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "predict"))
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/split_manifest_v1.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/splits/split_summary_v1.json")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--model", type=Path, help="model.json to load for predict")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args(argv)
    _settings(args.alpha, args.min_count)
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    if args.action == "predict" and args.model is None:
        parser.error("predict requires --model")
    if args.action == "train" and args.split != "validation":
        parser.error("train always reports held-out validation")
    started = time.perf_counter()
    train_identity = None
    if args.action == "train":
        records, train_identity = load_records(args.processed_dir, "train", args.manifest, args.summary)
        model = fit([word_level_example(row) for row in records], alpha=args.alpha, min_count=args.min_count)
        del records
    else:
        model = validate_model(json.loads(args.model.read_text(encoding="utf-8")))
    records, validation_identity = load_records(args.processed_dir, args.split, args.manifest, args.summary)
    predictions = predict(model, [word_level_example(row) for row in records])
    metrics = evaluate_records(records, predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    _write_json(args.output / "model.json", model)
    prediction_file = args.output / f"{args.split}_predictions.jsonl"
    _write_predictions(prediction_file, predictions)
    report = {"format": FORMAT, "study": "post-feedback supplementary validation benchmark",
              "model": "supervised_hmm_response_only", "input": "response_only",
              "settings": {"alpha": model["alpha"], "min_count": model["min_count"],
                           "decoding": "BIO-constrained Viterbi including END probabilities"},
              "selection": "fixed a priori settings; no validation hyperparameter search",
              "train": train_identity, "evaluation": validation_identity, "split": args.split,
              "official_test_used": False, "validation": metrics,
              "training_responses": model["training_responses"], "vocabulary_size": len(model["vocabulary"]),
              "elapsed_seconds": time.perf_counter() - started, "python": platform.python_version(),
              "sha256": {"script": sha256_file(__file__),
                         "model": sha256_file(args.output / "model.json"),
                         "predictions": sha256_file(prediction_file)}}
    _write_json(args.output / "summary.json", report)
    print(json.dumps({"output": str(args.output), "responses": len(predictions),
                      "any_f1": metrics["character"]["any"]["f1"],
                      "macro_f1_types": metrics["character"]["macro_f1_types"],
                      "elapsed_seconds": report["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
