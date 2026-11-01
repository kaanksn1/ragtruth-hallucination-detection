"""Rescore saved predictions without project scorer code; report-only, selects nothing."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import string
import sys

ROOT = Path(__file__).resolve().parents[1]
TYPES = ("Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info")
EDGE = set(string.punctuation) - set("$%'\"()")
FALLBACK_RUNS = "outputs/runs"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def raw_gold(split_ids):
    # Gold, projenin işlenmiş dosyasından değil doğrudan ham indirmeden okunur.
    gold = {}
    for row in read_jsonl(ROOT / "data/raw/response.jsonl"):
        if row["id"] not in split_ids:
            continue
        labels = row["labels"]
        labels = json.loads(labels) if isinstance(labels, str) else labels
        gold[row["id"]] = {"response": row["response"], "quality": row["quality"],
                                "spans": [(x["start"], x["end"], x["label_type"]) for x in labels]}
    return gold


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def chars(spans, kind=None):
    found = set()
    for start, end, label in spans:
        if kind is None or label == kind:
            found.update(range(start, end))
    return found


def tally(pairs):
    counts = defaultdict(lambda: [0, 0, 0])
    for gold, pred in pairs:
        for kind in (None, *TYPES):
            g, p = chars(gold, kind), chars(pred, kind)
            bucket = counts[kind]
            bucket[0] += len(g & p)
            bucket[1] += len(p - g)
            bucket[2] += len(g - p)
    return counts


def exact(pairs):
    tp = fp = fn = 0
    for gold, pred in pairs:
        g, p = set(gold), set(pred)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    return prf(tp, fp, fn)[2]


def overlap(pairs):
    # Aynı türde en az bir karakter paylaşım: sınır katılığını kaldıran gevşek ölçü.
    p_hit = p_all = g_hit = g_all = 0
    for gold, pred in pairs:
        g, p = set(gold), set(pred)
        hit = lambda a, b: a[2] == b[2] and a[0] < b[1] and b[0] < a[1]
        p_all += len(p)
        g_all += len(g)
        p_hit += sum(any(hit(x, y) for y in g) for x in p)
        g_hit += sum(any(hit(x, y) for y in p) for x in g)
    precision = p_hit / p_all if p_all else 0.0
    recall = g_hit / g_all if g_all else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def response_level(pairs):
    tp = sum(bool(g) and bool(p) for g, p in pairs)
    fp = sum(not g and bool(p) for g, p in pairs)
    fn = sum(bool(g) and not p for g, p in pairs)
    tn = len(pairs) - tp - fp - fn
    return dict(zip(("precision", "recall", "f1"), prf(tp, fp, fn))) | {"positive_rate": (tp + fn) / len(pairs),
                                                                       "accuracy": (tp + tn) / len(pairs)}


def strip_edges(span, text):
    start, end, label = span
    while start < end and (text[start].isspace() or text[start] in EDGE):
        start += 1
    while end > start and (text[end - 1].isspace() or text[end - 1] in EDGE):
        end -= 1
    return start, end, label


def merge_gaps(spans, text, gap):
    # Aynı türde yalnız kısa boşluk/noktalama ile ayrılmış parçaları birleştirir.
    merged = []
    for kind in TYPES:
        current = None
        for start, end, label in sorted(x for x in spans if x[2] == kind):
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


def load_pairs(path, gold):
    pairs = []
    for row in read_jsonl(path):
        entry = gold[row["id"]]
        pred = [(x["start"], x["end"], x["label_type"]) for x in row["spans"]]
        pairs.append((entry["spans"], pred, entry["response"]))
    return pairs


def main():
    pack = Path(sys.argv[1])
    runs = pack / FALLBACK_RUNS
    files = {"test": {"crf_source": runs / "20261031T202752426074Z-final-predict-crf-source/predictions.jsonl",
                      "crf_response_only": runs / "20261031T202805690074Z-final-predict-crf-response-only/predictions.jsonl",
                      "transformer": runs / "20261031T202820352826Z-final-predict-transformer/predictions.jsonl"},
             "validation": {"crf_source": runs / "20261031T133357203274Z-crf-validation/validation_predictions.jsonl",
                            "crf_response_only": runs / "20261031T133357203274Z-crf-validation/validation_predictions_response_only.jsonl",
                            "transformer_default": runs / "20261031T110122793234Z-transformer-train/validation_predictions.jsonl",
                            "transformer": pack / "outputs/calibrated-validation.jsonl"}}
    output = {}
    for split, models in files.items():
        ids = {row["id"] for path in models.values() for row in read_jsonl(path)}
        gold = raw_gold(ids)
        output[split] = {"responses": len(ids), "models": {}}
        for name, path in models.items():
            triples = load_pairs(path, gold)
            pairs = [(g, p) for g, p, _ in triples]
            entry = summary(pairs) | {"response_level": response_level(pairs)}
            if split == "validation":
                # Sınır denemeleri yalnız validation; test'te hiç çalıştırılmaz.
                stripped = [(g, [strip_edges(x, t) for x in p]) for g, p, t in triples]
                both = [([strip_edges(x, t) for x in g], [strip_edges(x, t) for x in p]) for g, p, t in triples]
                entry["boundary"] = {"pred_stripped_exact": exact(stripped),
                                     "gold_and_pred_stripped_exact": exact(both)}
                for gap in (1, 3):
                    merged = [(g, merge_gaps(p, t, gap)) for g, p, t in triples]
                    entry["boundary"][f"merge_gap{gap}_exact"] = exact(merged)
                    entry["boundary"][f"merge_gap{gap}_any_f1"] = summary(merged)["any_f1"]
                    merged_both = [([strip_edges(x, t) for x in g], [strip_edges(x, t) for x in merge_gaps(p, t, gap)])
                                   for g, p, t in triples]
                    entry["boundary"][f"merge_gap{gap}_both_stripped_exact"] = exact(merged_both)
            output[split]["models"][name] = entry
    (ROOT / "results/independent_rescore.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=1))


if __name__ == "__main__":
    main()
