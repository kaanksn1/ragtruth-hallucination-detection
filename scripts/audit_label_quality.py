"""Audit annotation quality on train/validation only; official test labels are never read."""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import re
import statistics
import string

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation")
TYPES = ("Evident Conflict", "Subtle Conflict", "Evident Baseless Info", "Subtle Baseless Info")
EDGE_PUNCTUATION = set(string.punctuation) - set("$%'\"()")


def load_records():
    records = []
    for split in SPLITS:
        with (ROOT / "data/processed" / f"{split}.jsonl").open(encoding="utf-8") as handle:
            records += [json.loads(line) for line in handle]
    return [record for record in records if record["quality"] == "good"]


def percent(part, whole):
    return round(100 * part / whole, 2) if whole else 0.0


def boundary_stats(records):
    counts = Counter()
    for record in records:
        response = record["response"]
        for label in record["labels"]:
            start, end, text = label["start"], label["end"], label["text"]
            counts["spans"] += 1
            counts["edge_whitespace"] += text != text.strip()
            counts["edge_punctuation"] += bool(text.strip()) and (
                text.strip()[0] in EDGE_PUNCTUATION or text.strip()[-1] in EDGE_PUNCTUATION)
            counts["mid_word_start"] += start > 0 and response[start - 1].isalnum() and response[start].isalnum()
            counts["mid_word_end"] += end < len(response) and response[end - 1].isalnum() and response[end].isalnum()
            counts["ends_sentence"] += text.rstrip()[-1:] in {".", "!", "?"}
    return counts


def type_stats(records):
    words = defaultdict(list)
    flags = defaultdict(Counter)
    for record in records:
        for label in record["labels"]:
            kind = label["label_type"]
            words[kind].append(len(label["text"].split()))
            flags[kind]["spans"] += 1
            flags[kind]["implicit_true"] += bool(label["implicit_true"])
            flags[kind]["due_to_null"] += bool(label["due_to_null"])
    return {kind: {"spans": flags[kind]["spans"],
                   "median_words": statistics.median(words[kind]),
                   "p90_words": sorted(words[kind])[int(0.9 * len(words[kind]))],
                   "max_words": max(words[kind]),
                   "implicit_true_pct": percent(flags[kind]["implicit_true"], flags[kind]["spans"]),
                   "due_to_null_pct": percent(flags[kind]["due_to_null"], flags[kind]["spans"])}
            for kind in TYPES}


def note_type(label):
    # Anotasyon notunun ilk satırı, etiketleyicinin yazdığı türü taşıyabilir.
    first_line = (label.get("meta") or "").strip().split("\n")[0].lower()
    first_line = re.sub(r"[^a-z ]", " ", first_line)
    if first_line.startswith(("high intro", "low intro")):
        return ("Evident" if first_line.startswith("high") else "Subtle") + " Baseless Info"
    subtle = "subtle" in first_line
    evident = "evident" in first_line
    conflict = "conflict" in first_line
    baseless = "baseless" in first_line
    if subtle == evident or conflict == baseless:
        return None
    return ("Subtle " if subtle else "Evident ") + ("Conflict" if conflict else "Baseless Info")


def note_agreement(records):
    confusion = Counter()
    for record in records:
        for label in record["labels"]:
            confusion[(label["label_type"], note_type(label) or "unparsed")] += 1
    parsed = sum(count for (_, note), count in confusion.items() if note != "unparsed")
    different = [(kind, note, count) for (kind, note), count in confusion.items()
                 if note != "unparsed" and kind != note]
    return {"parsed": parsed, "unparsed": sum(confusion.values()) - parsed,
            "disagreements": sum(item[2] for item in different),
            "disagreement_pct": percent(sum(item[2] for item in different), parsed),
            "pairs": [{"label_type": kind, "note_type": note, "count": count}
                      for kind, note, count in sorted(different, key=lambda item: -item[2])]}


def covered(record, kind=None):
    chars = set()
    for label in record["labels"]:
        if kind is None or label["label_type"] == kind:
            chars.update(range(label["start"], label["end"]))
    return chars


def identical_response_agreement(records):
    # Aynı kaynak + aynı metin: etiket farkı modelden değil, anotasyondan gelir.
    groups = defaultdict(list)
    for record in records:
        groups[(record["source_id"], record["response"])].append(record)
    any_tp = any_fp = any_fn = 0
    typed = Counter()
    pairs = binary_same = exact_same = both_positive = 0
    for group in groups.values():
        for left, right in combinations(group, 2):
            pairs += 1
            binary_same += left["has_hallucination"] == right["has_hallucination"]
            exact_same += ({(x["start"], x["end"], x["label_type"]) for x in left["labels"]}
                           == {(x["start"], x["end"], x["label_type"]) for x in right["labels"]})
            both_positive += left["has_hallucination"] and right["has_hallucination"]
            a, b = covered(left), covered(right)
            any_tp += len(a & b)
            any_fp += len(a - b)
            any_fn += len(b - a)
            for kind in TYPES:
                a, b = covered(left, kind), covered(right, kind)
                typed[kind, "tp"] += len(a & b)
                typed[kind, "wrong"] += len(a ^ b)
    def f1(tp, wrong):
        return round(100 * 2 * tp / (2 * tp + wrong), 2) if tp + wrong else None
    return {"pairs": pairs, "positive_in_both": both_positive,
            "same_binary_decision_pct": percent(binary_same, pairs),
            "identical_labels_pct": percent(exact_same, pairs),
            "any_character_agreement_f1": f1(any_tp, any_fp + any_fn),
            "typed_character_agreement_f1": {kind: f1(typed[kind, "tp"], typed[kind, "wrong"]) for kind in TYPES}}


def main():
    records = load_records()
    boundaries = boundary_stats(records)
    spans = boundaries["spans"]
    report = {"scope": "good-quality train+validation; official test labels not read",
              "responses": len(records),
              "spans": spans,
              "boundaries": {key: {"count": boundaries[key], "pct": percent(boundaries[key], spans)}
                             for key in boundaries if key != "spans"},
              "types": type_stats(records),
              "note_type_vs_label_type": note_agreement(records),
              "identical_response_pairs": identical_response_agreement(records)}
    path = ROOT / "results/label_quality.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
