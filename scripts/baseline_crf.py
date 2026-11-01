"""CRF baseline for RAGTruth hallucination spans (3. kişi).

    python scripts/baseline_crf.py train --config configs/crf.json --output outputs/crf-local
    python scripts/baseline_crf.py predict --model-dir RUN_DIR/models/source --split validation --output preds.jsonl

One linear-chain CRF per annotation type (four O/B/I taggers), so labels of
different types never compete for a token and cross-type overlaps survive
(docs/DATA_FORMAT.md). Features are hand-written: the response word, its
neighbours and word shape, plus — unless switched off — whether the word, its
number or its bigram also occurs in ``source_text``. Only
``schema.MODEL_INPUT_FIELDS`` reach the features; ids, task type, generator and
annotation fields never do.

``train`` fits every grid setting on the train split, scores validation with a
character-level F1 that is used only to pick the setting, refits the chosen
setting without source features (response-only control) and writes
predictions in the shared format ``{"id", "spans": [{"start", "end", "label_type"}]}``.
The project scorer belongs to 5. kişi; numbers here are for model selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data.alignment import merge_intervals, tags_to_char_spans, word_tokenize  # noqa: E402
from ragtruth_data.data import read_jsonl, sha256_file, write_jsonl  # noqa: E402
from ragtruth_data.schema import LABEL_TYPES  # noqa: E402
from scripts.run_job import require_frozen_test_job  # noqa: E402

STOPWORDS = frozenset("""
a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those he she they them his her their we our you your i me my not no so than then there here which who
whom what when where why how can could would should will may might must do does did has have had into over
under about after before also just very more most such only other some any each all both
""".split())
SENTENCE_END = frozenset(".!?")
VARIANTS = {"source": True, "response_only": False}


# ---------------------------------------------------------------- features
def word_shape(word: str) -> str:
    """'Ankara' -> 'Xxx', '2018' -> 'dd', 'COVID-19' -> 'XX-dd' (runs capped at two)."""
    shape = re.sub(r"[A-Z]", "X", word)
    shape = re.sub(r"[a-z]", "x", shape)
    shape = re.sub(r"\d", "d", shape)
    return re.sub(r"(.)\1+", r"\1\1", shape)


def is_content(word: str) -> bool:
    """Numbers and non-stopword words carry the facts a response can get wrong."""
    return any(c.isdigit() for c in word) or (any(c.isalpha() for c in word) and word.lower() not in STOPWORDS)


def source_index(source_text: str) -> tuple[frozenset, frozenset]:
    """Lower-cased word and bigram sets of the source, with the same tokenizer as the response."""
    low = [t.lower() for t in word_tokenize(source_text)[0]]
    return frozenset(low), frozenset(zip(low, low[1:]))


def sentence_ids(tokens: list) -> list:
    """Sentence index per token; a sentence ends at . ! ? followed by a capital or the end ('3.5' does not)."""
    ids, k = [], 0
    for i, word in enumerate(tokens):
        ids.append(k)
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if word in SENTENCE_END and (nxt is None or nxt[:1].isupper()):
            k += 1
    return ids


def _bucket(ratio: float) -> str:
    return "0" if ratio == 0 else "low" if ratio <= 0.25 else "mid" if ratio <= 0.5 else "high"


def token_features(tokens: list, source: tuple | None = None, novelty_window: int = 3) -> list:
    """Binary feature strings for every response token; ``source=None`` gives the response-only view."""
    low = [t.lower() for t in tokens]
    n = len(tokens)
    sents = sentence_ids(tokens)
    if source is not None:
        vocab, bigrams = source
        new = [is_content(t) and w not in vocab for t, w in zip(tokens, low)]
        content, novel, novel_num = {}, {}, set()
        for i, s in enumerate(sents):
            if is_content(tokens[i]):
                content[s] = content.get(s, 0) + 1
                novel[s] = novel.get(s, 0) + new[i]
                if new[i] and any(c.isdigit() for c in tokens[i]):
                    novel_num.add(s)
    feats = []
    for i, word in enumerate(tokens):
        sent_start = i == 0 or sents[i] != sents[i - 1]
        f = ["bias", f"w={low[i]}", f"suf3={low[i][-3:]}", f"shape={word_shape(word)}",
             f"sent_index={min(sents[i], 3)}"]
        if word.istitle():
            f.append("title")
        if word.isupper() and len(word) > 1:
            f.append("upper")
        if any(c.isdigit() for c in word):
            f.append("has_digit")
        if not any(c.isalnum() for c in word):
            f.append("punct")
        if sent_start:
            f.append("sent_start")
        for d in (-1, 1):
            j = i + d
            if 0 <= j < n:
                f += [f"{d}:w={low[j]}", f"{d}:shape={word_shape(tokens[j])}"]
            else:
                f.append(f"{d}:pad")
        if source is not None:
            s = sents[i]
            f.append(f"src:sentence_new={_bucket(novel.get(s, 0) / content[s]) if content.get(s) else 'none'}")
            if s in novel_num:
                f.append("src:sentence_new_number")
            f.append("src:in" if low[i] in vocab else "src:out")
            if is_content(word):
                f.append("src:content_new" if new[i] else "src:content_seen")
            if any(c.isdigit() for c in word):
                f.append("src:num_new" if low[i] not in vocab else "src:num_seen")
            if word.istitle() and not sent_start and low[i] not in vocab:
                f.append("src:cap_new")
            if i > 0:
                f.append("src:bigram_prev_in" if (low[i - 1], low[i]) in bigrams else "src:bigram_prev_out")
            if i < n - 1:
                f.append("src:bigram_next_in" if (low[i], low[i + 1]) in bigrams else "src:bigram_next_out")
            for d in (-1, 1):
                if 0 <= i + d < n and new[i + d]:
                    f.append(f"{d}:src:content_new")
            k = sum(new[max(0, i - novelty_window):i + novelty_window + 1])
            f.append(f"src:new_in_window={min(k, 3)}")
        feats.append(f)
    return feats


# ---------------------------------------------------------------- data
def load_examples(processed_dir: Path, split: str, limit: int | None = None, *, with_labels: bool = True) -> list:
    """CRF rows joined with their records; quality == 'good' only, like ``data.load_split``."""
    records = {}
    for record in read_jsonl(processed_dir / f"{split}.jsonl"):
        if record["id"] in records:
            raise ValueError(f"duplicate response id in {split}: {record['id']}")
        records[record["id"]] = record
    examples, seen = [], set()
    for row in read_jsonl(processed_dir / f"crf_{split}.jsonl"):
        record = records.get(row["id"])
        if record is None:
            raise ValueError(f"CRF row has no matching response: {row['id']}")
        if row["id"] in seen:
            raise ValueError(f"duplicate CRF response id: {row['id']}")
        seen.add(row["id"])
        if record["quality"] != "good":
            continue
        expected_tokens, expected_offsets = word_tokenize(record["response"])
        offsets = [tuple(o) for o in row["offsets"]]
        if row["tokens"] != expected_tokens or offsets != expected_offsets:
            raise ValueError(f"CRF tokens/offsets differ from the original response: {row['id']}")
        example = {
            "id": row["id"], "tokens": row["tokens"], "offsets": [tuple(o) for o in row["offsets"]],
            "source_text": record["source_text"],
        }
        if with_labels:
            targets = row.get("bio_by_type")
            if (not isinstance(targets, dict) or set(targets) != set(LABEL_TYPES)
                    or any(len(tags) != len(expected_tokens) or any(t not in {"O", "B", "I"} for t in tags)
                           for tags in targets.values())):
                raise ValueError(f"invalid per-type BIO targets: {row['id']}")
            example.update(bio_by_type=targets, labels=record["labels"])
        examples.append(example)
    missing = {rid for rid, r in records.items() if r["quality"] == "good"} - seen
    if missing:
        raise ValueError(f"missing CRF rows for {len(missing)} good responses")
    return examples[:limit] if limit is not None else examples


def example_features(examples: list, use_source: bool, novelty_window: int):
    cache = {}
    for ex in examples:
        source = None
        if use_source:
            source = cache.get(ex["source_text"])
            if source is None:
                source = cache[ex["source_text"]] = source_index(ex["source_text"])
        yield ex, token_features(ex["tokens"], source, novelty_window)


# ---------------------------------------------------------------- spans and selection metric
def tags_to_spans(offsets: list, tags_by_type: dict) -> list:
    spans = [s for t in LABEL_TYPES for s in tags_to_char_spans(offsets, tags_by_type[t], t)]
    return sorted(spans, key=lambda s: (s["start"], s["end"], LABEL_TYPES.index(s["label_type"])))


def invalid_starts(tags: list) -> int:
    """'I' with no B/I before it; tags_to_char_spans reads it as a span start."""
    return sum(t == "I" and (i == 0 or tags[i - 1] == "O") for i, t in enumerate(tags))


def _chars(spans, label_type=None) -> set:
    return {c for s, e in merge_intervals((x["start"], x["end"]) for x in spans
                                          if label_type is None or x["label_type"] == label_type)
            for c in range(s, e)}


def char_scores(gold_by_id: dict, pred_by_id: dict) -> dict:
    """Micro character precision/recall/F1 per type and for the union of all types ('any')."""
    if gold_by_id.keys() != pred_by_id.keys():
        raise ValueError("prediction IDs must exactly match gold IDs for CRF selection")
    out = {}
    for label_type in (None, *LABEL_TYPES):
        tp = fp = fn = 0
        for rid, gold in gold_by_id.items():
            g, p = _chars(gold, label_type), _chars(pred_by_id.get(rid, []), label_type)
            tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label_type or "any"] = {"precision": precision, "recall": recall, "f1": f1, "gold_chars": tp + fn}
    out["macro_f1_types"] = sum(out[t]["f1"] for t in LABEL_TYPES) / len(LABEL_TYPES)
    return out


def decodings(config: dict) -> list:
    """Viterbi first, then posterior thresholds from high to low (tie-break order)."""
    out = [{"name": "viterbi"}]
    for tau in sorted(config["decoding"]["posterior_thresholds"], reverse=True):
        out.append({"name": f"posterior_{tau:g}", "threshold": tau})
    return out


def threshold_tags(probs: list, threshold: float) -> list:
    """Flag tokens with P(B or I) >= threshold; a flagged run starts with B."""
    return [("I" if i and probs[i - 1] >= threshold else "B") if p >= threshold else "O" for i, p in enumerate(probs)]


def decode(examples: list, run: dict, decoding: dict) -> tuple[dict, dict]:
    """Spans per response for one decoding, with the count of I tags that start a span."""
    validate_decoding(decoding)
    if decoding["name"] == "posterior_by_type":
        return decode_per_type(examples, run, decoding["thresholds_by_type"])
    spans, invalid = {}, {t: 0 for t in LABEL_TYPES}
    for ex in examples:
        if decoding["name"] == "viterbi":
            tags = run["viterbi"][ex["id"]]
        else:
            tags = {t: threshold_tags(p, decoding["threshold"]) for t, p in run["probs"][ex["id"]].items()}
        for t in LABEL_TYPES:
            invalid[t] += invalid_starts(tags[t])
        spans[ex["id"]] = tags_to_spans(ex["offsets"], tags)
    return spans, invalid


def select_candidate(candidates: list) -> dict:
    """Highest validation 'any' character F1; ties keep the earliest (setting, decoding)."""
    if not candidates:
        raise ValueError("no grid results to select from")
    return max(candidates, key=lambda c: c["validation"]["any"]["f1"])


# ------------------------------------------------- per-type threshold calibration
MIN_PRECISION_FLOOR = 0.01
"""Below this validation precision, a threshold is excluded from per-type selection.

The shared grid/decoding search above picks ONE posterior threshold for all
four types from their combined ('any') character F1. A rare type's few gold
characters barely move that aggregate, so a threshold that floods the rare
type with false positives can still report a nonzero (if tiny) F1 for that
type and would otherwise be preferred over predicting nothing at all -- the
exact failure this floor exists to block. A threshold predicting zero
characters is always eligible regardless of precision (it cannot be a false
positive flood). If no grid threshold clears the floor, selection fails closed;
the highest grid point is not automatically safe. This constraint applies to
per-type/joint candidates, not the historical global/default diagnostics.
"""


def _type_prf(tp: int, fp: int, fn: int) -> dict:
    predicted, gold = tp + fp, tp + fn
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * tp / (predicted + gold) if predicted + gold else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive_chars": tp, "predicted_chars": predicted, "gold_chars": gold}


def _predicted_char_sets(examples: list, run: dict, label_type: str, grid: list) -> dict:
    """{threshold: {response_id: predicted character set}} for one type, every grid point."""
    return {threshold: {ex["id"]: _chars(tags_to_char_spans(
                ex["offsets"], threshold_tags(run["probs"][ex["id"]][label_type], threshold), label_type), label_type)
            for ex in examples} for threshold in sorted(grid)}


def _eligible_thresholds(rows: list, grid: list, min_precision: float) -> list:
    """Only grid points clearing the floor; refuse an unsafe fallback."""
    eligible = [row["threshold"] for row in rows if row["predicted_chars"] == 0 or row["precision"] >= min_precision]
    if not eligible:
        raise ValueError("no posterior threshold satisfies the precision floor; declare a more conservative grid")
    return eligible


def validate_type_thresholds(thresholds):
    if (not isinstance(thresholds, dict) or set(thresholds) != set(LABEL_TYPES)
            or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 < t <= 1
                   for t in thresholds.values())):
        raise ValueError("posterior thresholds must name all four types with finite values in (0, 1]")


def validate_decoding(decoding):
    if not isinstance(decoding, dict) or not isinstance(decoding.get("name"), str):
        raise ValueError("saved CRF decoding must identify its strategy")
    if decoding["name"] == "viterbi":
        return
    if decoding["name"] == "posterior_by_type":
        validate_type_thresholds(decoding.get("thresholds_by_type"))
        return
    threshold = decoding.get("threshold")
    if (not decoding["name"].startswith("posterior_") or type(threshold) not in (int, float)
            or not math.isfinite(threshold) or not 0 < threshold <= 1):
        raise ValueError("saved CRF posterior decoding needs a finite threshold in (0, 1]")


def validate_threshold_search(examples, run, gold, grid, min_precision):
    if (not isinstance(grid, (list, tuple)) or not grid
            or any(type(t) not in (int, float) or not math.isfinite(t) or not 0 < t <= 1 for t in grid)
            or len(set(grid)) != len(grid)):
        raise ValueError("posterior search grid must contain distinct finite thresholds in (0, 1]")
    if type(min_precision) not in (int, float) or not math.isfinite(min_precision) or not 0 <= min_precision <= 1:
        raise ValueError("precision floor must be finite and between zero and one")
    ids = [ex["id"] for ex in examples]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(gold) or set(ids) != set(run.get("probs", {})):
        raise ValueError("threshold search requires identical nonempty response ID sets")
    for ex in examples:
        columns = run["probs"][ex["id"]]
        if (set(columns) != set(LABEL_TYPES) or any(len(p) != len(ex["offsets"])
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                for p in columns.values())):
            raise ValueError("posterior probabilities must align with tokens and be finite values in [0, 1]")


def tune_type_thresholds(examples: list, run: dict, gold: dict, grid: list,
                         min_precision: float = MIN_PRECISION_FLOOR) -> tuple[dict, dict]:
    """Independent per-type posterior threshold, chosen only from that type's own characters.

    Unlike ``decodings``/``select_candidate`` above (one shared threshold,
    picked from the combined 'any' F1), every type here searches the same
    posterior grid on its own predicted/gold characters, with false positives
    monitored and bounded by ``min_precision``.
    """
    validate_threshold_search(examples, run, gold, grid, min_precision)
    gold_chars = {rid: {t: _chars(spans, t) for t in LABEL_TYPES} for rid, spans in gold.items()}
    selected, table = {}, {}
    for label_type in LABEL_TYPES:
        char_sets = _predicted_char_sets(examples, run, label_type, grid)
        rows = []
        for threshold in sorted(grid):
            tp = fp = fn = 0
            for ex in examples:
                predicted = char_sets[threshold][ex["id"]]
                g = gold_chars[ex["id"]][label_type]
                tp, fp, fn = tp + len(g & predicted), fp + len(predicted - g), fn + len(g - predicted)
            rows.append({"threshold": threshold, **_type_prf(tp, fp, fn)})
        table[label_type] = rows
        eligible_values = set(_eligible_thresholds(rows, grid, min_precision))
        eligible_rows = [row for row in rows if row["threshold"] in eligible_values]
        best = max(eligible_rows, key=lambda row: (row["f1"], row["threshold"]))
        selected[label_type] = best["threshold"]
    return selected, table


def tune_joint_thresholds(examples: list, run: dict, gold: dict, grid: list,
                          min_precision: float = MIN_PRECISION_FLOOR) -> tuple[dict, dict]:
    """Per-type thresholds chosen jointly to maximize the combined ('any') detection F1.

    ``tune_type_thresholds`` lets each type maximize its *own* F1 independently, which can
    leave the primary detection metric on the table (two types trading precision for recall
    in ways that do not help, or even hurt, their union). This instead searches combinations
    of thresholds -- restricted to each type's own false-positive-floor-eligible values, so a
    flooding threshold is never admitted just because some combination nominally helps the
    union -- and keeps the combination with the highest combined 'any' F1. Ties prefer the
    combination with the higher macro F1, then the higher threshold tuple.
    """
    validate_threshold_search(examples, run, gold, grid, min_precision)
    gold_any = {rid: _chars(spans) for rid, spans in gold.items()}
    gold_by_type = {rid: {t: _chars(spans, t) for t in LABEL_TYPES} for rid, spans in gold.items()}
    char_sets, eligible = {}, {}
    independent_table = {}
    for label_type in LABEL_TYPES:
        char_sets[label_type] = _predicted_char_sets(examples, run, label_type, grid)
        rows = []
        for threshold in sorted(grid):
            tp = fp = fn = 0
            for ex in examples:
                predicted = char_sets[label_type][threshold][ex["id"]]
                g = gold_by_type[ex["id"]][label_type]
                tp, fp, fn = tp + len(g & predicted), fp + len(predicted - g), fn + len(g - predicted)
            rows.append({"threshold": threshold, **_type_prf(tp, fp, fn)})
        independent_table[label_type] = rows
        eligible[label_type] = sorted(_eligible_thresholds(rows, grid, min_precision))

    best_combo, best_key = None, None
    evaluated = []
    for combo in itertools.product(*(eligible[t] for t in LABEL_TYPES)):
        tp = fp = fn = 0
        type_tp = {t: 0 for t in LABEL_TYPES}
        type_predicted = {t: 0 for t in LABEL_TYPES}
        type_gold = {t: 0 for t in LABEL_TYPES}
        for ex in examples:
            rid = ex["id"]
            predicted_any = set()
            for label_type, threshold in zip(LABEL_TYPES, combo):
                predicted = char_sets[label_type][threshold][rid]
                predicted_any |= predicted
                g = gold_by_type[rid][label_type]
                type_tp[label_type] += len(g & predicted)
                type_predicted[label_type] += len(predicted)
                type_gold[label_type] += len(g)
            g_any = gold_any[rid]
            tp, fp, fn = tp + len(g_any & predicted_any), fp + len(predicted_any - g_any), fn + len(g_any - predicted_any)
        any_scores = _type_prf(tp, fp, fn)
        per_type = {t: _type_prf(type_tp[t], type_predicted[t] - type_tp[t], type_gold[t] - type_tp[t]) for t in LABEL_TYPES}
        macro_f1 = sum(per_type[t]["f1"] for t in LABEL_TYPES) / len(LABEL_TYPES)
        key = (any_scores["f1"], macro_f1, combo)  # tie-break: higher macro F1, then higher thresholds (more conservative)
        evaluated.append({"thresholds": dict(zip(LABEL_TYPES, combo)), "any": any_scores,
                          "macro_f1_types": macro_f1, "per_type": per_type})
        if best_key is None or key > best_key:
            best_key, best_combo = key, evaluated[-1]
    return best_combo["thresholds"], {"evaluated_combinations": len(evaluated), "eligible_per_type": eligible,
                                      "selected": best_combo, "independent_table": independent_table}


def select_best_configuration(candidates: dict) -> str:
    """Highest validation detection ('any') F1 among named configurations; macro F1 only ties.

    ``candidates`` maps a configuration name (e.g. 'default', 'global', 'per_type', 'joint')
    to its validation ``char_scores``-shaped metrics. This never lets a configuration with a
    macro-F1 or per-class gain but a *lower* detection F1 be preferred over one that is
    strictly better (or equal) on the primary metric -- the exact failure this guards against
    is a per-type search independently improving several classes while the union regresses.
    """
    if not candidates:
        raise ValueError("no configurations to select from")
    for metrics in candidates.values():
        values = (metrics.get("any", {}).get("f1"), metrics.get("macro_f1_types"))
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("configuration selection requires finite F1 values in [0, 1]")
    order = list(candidates)  # earliest-declared wins ties
    return max(order, key=lambda name: (candidates[name]["any"]["f1"], candidates[name]["macro_f1_types"],
                                        -order.index(name)))


def decode_per_type(examples: list, run: dict, thresholds_by_type: dict) -> tuple[dict, dict]:
    """Spans per response using one independently-tuned threshold for each type."""
    validate_type_thresholds(thresholds_by_type)
    spans, invalid = {}, {t: 0 for t in LABEL_TYPES}
    for ex in examples:
        tags = {t: threshold_tags(run["probs"][ex["id"]][t], thresholds_by_type[t]) for t in LABEL_TYPES}
        for t in LABEL_TYPES:
            invalid[t] += invalid_starts(tags[t])
        spans[ex["id"]] = tags_to_spans(ex["offsets"], tags)
    return spans, invalid


# ---------------------------------------------------------------- CRFsuite
def setting_name(setting: dict) -> str:
    return f"c1_{setting['c1']:g}_c2_{setting['c2']:g}"


def type_slug(label_type: str) -> str:
    return label_type.lower().replace(" ", "_")


def train_to_file(trainer, path: Path) -> None:
    """Stage native Windows output through an ASCII filename; Python copies it safely.

    CRFsuite's native filename API can silently fail on paths such as Masaüstü.
    Temporary files are scoped to this call, and model existence is checked.
    """
    if sys.platform == "win32" and not str(path).isascii():
        with tempfile.TemporaryDirectory(prefix="ragtruth-crf-") as directory:
            temporary = Path(directory) / "model.crfsuite"
            if not str(temporary).isascii():
                raise ValueError("CRFsuite on Windows needs an ASCII TMP/TEMP directory")
            trainer.train(str(temporary))
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError("CRFsuite did not write a model file")
            shutil.copyfile(temporary, path)
    else:
        trainer.train(str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("CRFsuite did not write a model file")


def train_variant(pycrfsuite, examples, grid, training, use_source, novelty_window, model_root: Path) -> dict:
    """Train every grid setting for each type; data is appended once per type and reused."""
    stats = {setting_name(s): {} for s in grid}
    for label_type in LABEL_TYPES:
        trainer = pycrfsuite.Trainer(algorithm=training.get("algorithm", "lbfgs"), verbose=False)
        start = time.perf_counter()
        for ex, feats in example_features(examples, use_source, novelty_window):
            if feats:
                trainer.append(feats, ex["bio_by_type"][label_type])
        append_seconds = time.perf_counter() - start
        for setting in grid:
            trainer.set_params({
                "c1": setting["c1"], "c2": setting["c2"], "max_iterations": training["max_iterations"],
                "feature.possible_transitions": training["possible_transitions"],
                "feature.minfreq": training["min_feature_frequency"],
            })
            path = model_root / setting_name(setting) / f"{type_slug(label_type)}.crfsuite"
            path.parent.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            train_to_file(trainer, path)
            last = trainer.logparser.last_iteration or {}
            stats[setting_name(setting)][label_type] = {
                "train_seconds": time.perf_counter() - start, "feature_append_seconds": append_seconds,
                "iterations": len(trainer.logparser.iterations), "final_loss": last.get("loss"),
                "active_features": last.get("active_features"),
                "stopped_at_max_iterations": len(trainer.logparser.iterations) >= training["max_iterations"],
            }
        del trainer
    return stats


class _MemoryTagger:
    """Keep the native model's borrowed bytes alive until its tagger is closed."""

    def __init__(self, pycrfsuite, value):
        self._model_bytes = value
        self._native = pycrfsuite.Tagger()
        self._native.open_inmemory(self._model_bytes)

    def __getattr__(self, name):
        return getattr(self._native, name)

    def close(self):
        native = self.__dict__.get("_native")
        if native is not None:
            native.close()
            self._native = None
        self._model_bytes = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def open_taggers(pycrfsuite, model_dir: Path) -> dict:
    taggers = {}
    for label_type in LABEL_TYPES:
        path = model_dir / f"{type_slug(label_type)}.crfsuite"
        if sys.platform == "win32" and not str(path).isascii():
            tagger = _MemoryTagger(pycrfsuite, path.read_bytes())
        else:
            tagger = pycrfsuite.Tagger()
            tagger.open(str(path))
        taggers[label_type] = tagger
    return taggers


def predict(pycrfsuite, examples, taggers_by_name: dict, use_source, novelty_window) -> dict:
    """Viterbi tags and per-token marginals P(B or I) from every tagger set."""
    runs = {name: {"viterbi": {}, "probs": {}} for name in taggers_by_name}
    for ex, feats in example_features(examples, use_source, novelty_window):
        xseq = pycrfsuite.ItemSequence(feats)
        for name, taggers in taggers_by_name.items():
            tags, probs = {}, {}
            for t in LABEL_TYPES:
                tagger = taggers[t]
                tags[t] = tagger.tag(xseq) if feats else []  # also sets xseq for marginal()
                probs[t] = ([1.0 - tagger.marginal("O", i) for i in range(len(feats))]
                            if "O" in tagger.labels() else [1.0] * len(feats))
            runs[name]["viterbi"][ex["id"]], runs[name]["probs"][ex["id"]] = tags, probs
    return runs


def top_weights(tagger, n: int = 15) -> dict:
    info = tagger.info()
    state = sorted(info.state_features.items(), key=lambda kv: -kv[1])
    return {
        "transitions": {f"{a}->{b}": w for (a, b), w in sorted(info.transitions.items())},
        "B": [{"feature": f, "weight": w} for (f, label), w in state if label == "B"][:n],
        "I": [{"feature": f, "weight": w} for (f, label), w in state if label == "I"][:n],
    }


def write_predictions(path: Path, examples: list, spans_by_id: dict) -> int:
    return write_jsonl(path, ({"id": ex["id"], "spans": spans_by_id[ex["id"]]} for ex in examples))


# ---------------------------------------------------------------- commands
def data_identity(processed_dir: Path, splits: tuple, summary_path: Path | None = None) -> dict:
    names = [f"{prefix}{split}.jsonl" for split in splits for prefix in ("", "crf_")]
    hashes = {name: sha256_file(processed_dir / name) for name in names}
    summary_path = summary_path or ROOT / "data/splits/split_summary_v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = summary.get("processed_sha256", {})
    mismatches = [name for name, value in hashes.items() if expected.get(name) != value]
    if mismatches:
        raise ValueError(f"processed data differs from split summary: {', '.join(mismatches)}")
    return {"files_sha256": hashes, "matches_split_summary_v1": True,
            "summary_sha256": sha256_file(summary_path)}


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("quality_filter") != "good":
        raise ValueError("CRF config requires schema_version 1 and quality_filter=good")
    if config["training"].get("algorithm") != "lbfgs":
        raise ValueError("this CRF grid supports the lbfgs algorithm")
    if type(config["features"].get("source_features")) is not bool:
        raise ValueError("source_features must be boolean")
    for value, name, minimum in ((config["features"]["novelty_window"], "novelty_window", 0),
                                 (config["training"]["max_iterations"], "max_iterations", 1),
                                 (config["training"]["min_feature_frequency"], "min_feature_frequency", 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not config.get("grid") or len({setting_name(s) for s in config["grid"]}) != len(config["grid"]):
        raise ValueError("grid must contain distinct nonempty settings")
    import math
    if any(not math.isfinite(float(s[k])) or s[k] < 0 for s in config["grid"] for k in ("c1", "c2")):
        raise ValueError("c1/c2 must be finite nonnegative values")
    if any(not math.isfinite(float(t)) or not 0 <= t <= 1 for t in config["decoding"]["posterior_thresholds"]):
        raise ValueError("posterior thresholds must be within [0, 1]")


def environment(pycrfsuite) -> dict:
    from importlib.metadata import version
    return {"python": platform.python_version(), "python_crfsuite": version("python-crfsuite"),
            "platform": platform.platform()}


def cmd_train(args, pycrfsuite) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    processed = ROOT / config["processed_dir"]
    window = config["features"]["novelty_window"]
    grid, training = config["grid"], config["training"]
    use_source = config["features"]["source_features"]
    identity = data_identity(processed, ("train", "validation"), ROOT / config.get(
        "split_summary", "data/splits/split_summary_v1.json"))
    out = args.output
    train = load_examples(processed, "train", args.limit_train)
    validation = load_examples(processed, "validation", args.limit_validation)
    if not train or not validation or not any(ex["tokens"] for ex in train):
        raise ValueError("nonempty train and validation data, with train tokens, are required")
    if {ex["id"] for ex in train} & {ex["id"] for ex in validation}:
        raise ValueError("train and validation response IDs overlap")
    reserved = ("grid", "response_only", "models", "crf_summary.json", "crf_report.md",
                "validation_predictions.jsonl", "validation_predictions_response_only.jsonl",
                "validation_predictions_per_type.jsonl", "validation_predictions_joint.jsonl")
    if any((out / name).exists() for name in reserved):
        raise FileExistsError("refusing to overwrite an existing CRF training run")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gold = {ex["id"]: ex["labels"] for ex in validation}
    summary = {
        "model": config["model"], "config": config, "seed": args.seed,
        "seed_note": "CRFsuite L-BFGS is deterministic; the seed is recorded, not used.",
        "data": identity,
        "use_source_features": use_source,
        "run_kind": "smoke" if args.limit_train is not None or args.limit_validation is not None else "full_validation",
        "train_responses": len(train), "validation_responses": len(validation),
        "train_tokens": sum(len(ex["tokens"]) for ex in train),
        "limits": {"train": args.limit_train, "validation": args.limit_validation},
        "selection_metric": "validation character F1 of the union of all predicted types ('any')",
        "official_test_evaluated": False, "environment": environment(pycrfsuite),
    }

    # 1) grid with source features; setting and decoding chosen on validation
    stats = train_variant(pycrfsuite, train, grid, training, use_source, window, out / "grid")
    taggers = {setting_name(s): open_taggers(pycrfsuite, out / "grid" / setting_name(s)) for s in grid}
    runs = predict(pycrfsuite, validation, taggers, use_source, window)
    results, candidates, spans_of = [], [], {}
    for setting in grid:
        name = setting_name(setting)
        rows = []
        for decoding in decodings(config):
            spans, invalid = decode(validation, runs[name], decoding)
            row = {"decoding": decoding, "validation": char_scores(gold, spans), "invalid_i_starts": invalid}
            rows.append(row)
            candidates.append({"setting": setting, "name": name, **row})
            spans_of[(name, decoding["name"])] = spans
        results.append({"setting": setting, "name": name, "training": stats[name], "decodings": rows})
    best = select_candidate(candidates)
    summary["grid"] = results
    summary["selected"] = {"setting": best["setting"], "name": best["name"], "decoding": best["decoding"],
                           "validation": best["validation"]}
    save_models(out / "grid" / best["name"], out / "models" / "source", use_source, window, best, identity)
    write_predictions(out / "validation_predictions.jsonl", validation,
                      spans_of[(best["name"], best["decoding"]["name"])])
    summary["top_weights"] = {t: top_weights(taggers[best["name"]][t]) for t in LABEL_TYPES}

    # 1b) per-type posterior thresholds for the chosen setting: the shared decoding above
    # picks one threshold from the combined 'any' F1, which can quietly flood a rare type
    # with false positives while barely moving that aggregate. This independent, bounded
    # search (see MIN_PRECISION_FLOOR) is additive and does not replace `selected` above.
    per_type_thresholds, per_type_table = tune_type_thresholds(
        validation, runs[best["name"]], gold, config["decoding"]["posterior_thresholds"])
    per_type_spans, per_type_invalid = decode_per_type(validation, runs[best["name"]], per_type_thresholds)
    summary["selected_per_type"] = {
        "setting": best["setting"], "name": best["name"], "thresholds_by_type": per_type_thresholds,
        "threshold_search": per_type_table, "min_precision_floor": MIN_PRECISION_FLOOR,
        "validation": char_scores(gold, per_type_spans), "invalid_i_starts": per_type_invalid,
    }
    write_predictions(out / "validation_predictions_per_type.jsonl", validation, per_type_spans)

    # 1c) joint per-type thresholds: unlike 1b, chosen together to directly maximize the
    # combined 'any' detection F1 (measured to beat both `selected` and `selected_per_type`
    # on validation -- see docs/BASELINE_CRF.md, 2026-11-01 note).
    joint_thresholds, joint_search = tune_joint_thresholds(
        validation, runs[best["name"]], gold, config["decoding"]["posterior_thresholds"])
    joint_spans, joint_invalid = decode_per_type(validation, runs[best["name"]], joint_thresholds)
    summary["selected_joint"] = {
        "setting": best["setting"], "name": best["name"], "thresholds_by_type": joint_thresholds,
        "evaluated_combinations": joint_search["evaluated_combinations"], "min_precision_floor": MIN_PRECISION_FLOOR,
        "validation": char_scores(gold, joint_spans), "invalid_i_starts": joint_invalid,
    }
    write_predictions(out / "validation_predictions_joint.jsonl", validation, joint_spans)

    # 1d) final configuration choice across all three threshold strategies plus the
    # un-calibrated default (0.5 for every type), ranked by detection F1 first.
    default_spans, _ = decode_per_type(validation, runs[best["name"]], {t: 0.5 for t in LABEL_TYPES})
    configuration_metrics = {
        "default": char_scores(gold, default_spans), "global": best["validation"],
        "per_type": summary["selected_per_type"]["validation"], "joint": summary["selected_joint"]["validation"],
    }
    summary["best_configuration"] = export_selected_configuration(
        out, validation, runs[best["name"]], best, identity, use_source, window,
        configuration_metrics, {
            "default": {"name": "posterior_0.5", "threshold": 0.5},
            "global": best["decoding"],
            "per_type": {"name": "posterior_by_type", "thresholds_by_type": per_type_thresholds},
            "joint": {"name": "posterior_by_type", "thresholds_by_type": joint_thresholds},
        })

    # 2) response-only control: same setting, decoding, split and budget; no source features
    if use_source and config.get("response_only_control", True):
        stats = train_variant(pycrfsuite, train, [best["setting"]], training, False, window, out / "response_only")
        control_dir = out / "response_only" / best["name"]
        run = predict(pycrfsuite, validation, {"m": open_taggers(pycrfsuite, control_dir)}, False, window)["m"]
        ro_spans, ro_invalid = decode(validation, run, best["decoding"])
        viterbi_spans, _ = decode(validation, run, {"name": "viterbi"})
        save_models(control_dir, out / "models" / "response_only", False, window, best, identity)
        write_predictions(out / "validation_predictions_response_only.jsonl", validation, ro_spans)
        summary["response_only_control"] = {
            "setting": best["setting"], "decoding": best["decoding"], "training": stats[best["name"]],
            "validation": char_scores(gold, ro_spans), "invalid_i_starts": ro_invalid,
            "validation_viterbi": char_scores(gold, viterbi_spans),
        }

    summary["total_seconds"] = time.perf_counter() - started
    (out / "crf_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "crf_report.md").write_text(render_report(summary), encoding="utf-8")
    return summary


def save_models(src_dir: Path, dst_dir: Path, use_source: bool, window: int, chosen: dict, identity: dict) -> None:
    """Copy the chosen four models with the settings ``predict`` needs to reproduce them."""
    validate_decoding(chosen["decoding"])
    dst_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for label_type in LABEL_TYPES:
        name = f"{type_slug(label_type)}.crfsuite"
        (dst_dir / name).write_bytes((src_dir / name).read_bytes())
        hashes[name] = sha256_file(dst_dir / name)
    (dst_dir / "model_config.json").write_text(json.dumps({
        "schema_version": 1, "label_types": LABEL_TYPES,
        "use_source_features": use_source, "novelty_window": window,
        "setting": chosen["setting"], "decoding": chosen["decoding"],
        "models_sha256": hashes, "data": identity,
    }, indent=2) + "\n", encoding="utf-8")


def export_selected_configuration(out, examples, run, best, identity, use_source, window, metrics, decoders):
    """Export the new winner separately from the unchanged historical global output."""
    winner = select_best_configuration(metrics)
    decoding = decoders[winner]
    spans, _ = decode(examples, run, decoding)
    model_dir = Path("models/source_selected")
    predictions = Path("validation_predictions_selected.jsonl")
    save_models(out / "grid" / best["name"], out / model_dir, use_source, window,
                dict(best, decoding=decoding), identity)
    write_predictions(out / predictions, examples, spans)
    return {"winner": winner,
            "rule": "highest validation detection ('any') F1, then macro F1, then earliest-declared on an exact tie",
            "candidates": {name: value["any"]["f1"] for name, value in metrics.items()},
            "decoding": decoding, "model_dir": model_dir.as_posix(), "predictions": predictions.as_posix(),
            "historical_global_model_dir": "models/source",
            "historical_global_predictions": "validation_predictions.jsonl"}


def cmd_predict(args, pycrfsuite) -> int:
    processed = ROOT / args.processed_dir
    if args.split == "test":
        if args.limit is not None:
            raise ValueError("official test predictions cannot be limited")
        inputs = require_frozen_test_job(ROOT, args.invocation)
        required = [args.model_dir / "model_config.json", Path(__file__),
                    ROOT / "scripts/run_job.py", ROOT / "ragtruth_data/alignment.py",
                    ROOT / "ragtruth_data/data.py", ROOT / "ragtruth_data/schema.py", ROOT / "requirements-crf.txt",
                    processed / "test.jsonl", processed / "crf_test.jsonl"]
        required += [args.model_dir / f"{type_slug(t)}.crfsuite" for t in LABEL_TYPES]
        if any(not path.resolve().is_relative_to(ROOT) or path.resolve().relative_to(ROOT).as_posix() not in inputs
               for path in required):
            raise ValueError("frozen test job must hash all CRF models, code and test input files")
    if args.output.exists():
        raise FileExistsError("refusing to overwrite existing predictions")
    model_config = json.loads((args.model_dir / "model_config.json").read_text(encoding="utf-8"))
    if model_config.get("schema_version") != 1 or model_config.get("label_types") != list(LABEL_TYPES):
        raise ValueError("saved model metadata has an unsupported label schema")
    validate_decoding(model_config.get("decoding"))
    expected = {f"{type_slug(t)}.crfsuite" for t in LABEL_TYPES}
    if set(model_config.get("models_sha256", {})) != expected or any(
            sha256_file(args.model_dir / name) != value for name, value in model_config["models_sha256"].items()):
        raise ValueError("saved CRF model checksum mismatch")
    if args.split == "validation":
        for name in ("validation.jsonl", "crf_validation.jsonl"):
            if sha256_file(processed / name) != model_config["data"]["files_sha256"].get(name):
                raise ValueError("validation data differs from the saved model's data identity")
    examples = load_examples(processed, args.split, args.limit, with_labels=False)
    run = predict(pycrfsuite, examples, {"m": open_taggers(pycrfsuite, args.model_dir)},
                  model_config["use_source_features"], model_config["novelty_window"])["m"]
    spans, _ = decode(examples, run, model_config["decoding"])
    return write_predictions(args.output, examples, spans)


# ---------------------------------------------------------------- report
def _row(name, scores):
    cells = [f"{scores[k]['f1']:.3f}" for k in ("any", *LABEL_TYPES)]
    return f"| {name} | {scores['any']['precision']:.3f} | {scores['any']['recall']:.3f} | " + " | ".join(cells) + \
        f" | {scores['macro_f1_types']:.3f} |"


def render_report(summary: dict) -> str:
    head = "| Ayar | P (any) | R (any) | F1 (any) | " + " | ".join(f"F1 {t}" for t in LABEL_TYPES) + " | Macro F1 (4 tür) |"
    lines = [
        "# CRF baseline — validation (seçim metriği)", "",
        "Kapsam: " + ("küçük doğrulama koşumu; tam veri sonucu değildir." if summary["run_kind"] == "smoke"
                       else "tam train/validation koşumu."),
        f"Train: {summary['train_responses']:,} cevap / {summary['train_tokens']:,} kelime token; "
        f"validation: {summary['validation_responses']:,} cevap (yalnız quality = good).",
        f"Veri split_summary_v1 ile aynı: {'evet' if summary['data']['matches_split_summary_v1'] else 'HAYIR'}.",
        "Metrik: karakter düzeyinde micro precision/recall/F1; yalnız ayar seçimi için. "
        "Projenin ortak scorer'ı 5. kişidedir. Resmî test kullanılmadı.", "",
        "Viterbi en olası etiket dizisini seçer; `posterior_τ`, forward-backward marjinali "
        "P(B veya I) ≥ τ olan tokenları işaretler.", "",
        "## " + ("Kaynak özellikli CRF" if summary["use_source_features"] else "Yalnız cevap CRF")
        + " — ayar ve çözümleme ızgarası", "", head, "|" + " --- |" * (len(LABEL_TYPES) + 5),
    ]
    for r in summary["grid"]:
        for d in r["decodings"]:
            lines.append(_row(f"{r['name']}, {d['decoding']['name']}", d["validation"]))
    selected = summary["selected"]
    lines += ["", f"Seçilen: **{selected['name']}, {selected['decoding']['name']}** "
                  "(en yüksek validation F1 any; eşitlikte tablodaki ilk satır).", ""]
    if "selected_per_type" in summary:
        per_type = summary["selected_per_type"]
        lines += ["## Tür başına bağımsız eşik (ek, `selected`i değiştirmez)", "",
                  f"Tek paylaşılan eşik, nadir türü neredeyse hiç etkilemeyen birleşik 'any' F1 ile seçilir; "
                  f"bu arama her türün kendi F1'ini kullanır ve precision < {per_type['min_precision_floor']:.0%} "
                  "olan eşikleri (sıfır tahmin hariç) eler.", "",
                  "| Tür | Eşik | P | R | F1 | Tahmin edilen karakter | Gold karakter |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for label_type in LABEL_TYPES:
            threshold = per_type["thresholds_by_type"][label_type]
            row = next(r for r in per_type["threshold_search"][label_type] if r["threshold"] == threshold)
            lines.append(f"| {label_type} | {threshold:g} | {row['precision']:.3f} | {row['recall']:.3f} | "
                         f"{row['f1']:.3f} | {row['predicted_chars']:,} | {row['gold_chars']:,} |")
        lines += ["", f"Birleşik (any, tür başına eşiklerle): F1 {per_type['validation']['any']['f1']:.3f}, "
                      f"macro F1 {per_type['validation']['macro_f1_types']:.3f}.", ""]
    if "selected_joint" in summary:
        joint = summary["selected_joint"]
        lines += ["## Birlikte (joint) eşik araması (ek, `selected`i değiştirmez)", "",
                  f"Tür başına bağımsız arama her türün kendi F1'ini en iyiler; bu arama "
                  f"{joint['evaluated_combinations']:,} eşik kombinasyonunu doğrudan birleşik "
                  "'any' F1'e göre dener (yine precision < "
                  f"{joint['min_precision_floor']:.0%} eşikleri eler).", "",
                  "| Tür | Eşik |", "| --- | ---: |"]
        for label_type in LABEL_TYPES:
            lines.append(f"| {label_type} | {joint['thresholds_by_type'][label_type]:g} |")
        lines += ["", f"Birleşik (any, joint eşiklerle): F1 {joint['validation']['any']['f1']:.3f}, "
                      f"macro F1 {joint['validation']['macro_f1_types']:.3f}.", ""]
    if "best_configuration" in summary:
        best_cfg = summary["best_configuration"]
        lines += ["## Dört yapılandırmanın karşılaştırması ve nihai seçim", "",
                  "| Yapılandırma | Detection F1 (any) |", "| --- | ---: |"]
        for name, f1 in best_cfg["candidates"].items():
            marker = " **(seçilen)**" if name == best_cfg["winner"] else ""
            lines.append(f"| {name} | {f1:.3f}{marker} |")
        lines += ["", f"Kural: {best_cfg['rule']}.", ""]
        if "model_dir" in best_cfg:
            lines += [f"Yeniden yüklenebilir seçili model: `{best_cfg['model_dir']}`; "
                      f"eşleşen dev tahmini: `{best_cfg['predictions']}`. "
                      "Tarihsel global çıktı ayrı korunur.", ""]
    if "response_only_control" in summary:
        control = summary["response_only_control"]
        lines += ["## Kaynak özelliklerinin katkısı (aynı ayar ve çözümleme, aynı veri)", "", head,
                  "|" + " --- |" * (len(LABEL_TYPES) + 5),
                  _row("kaynak + cevap", selected["validation"]),
                  _row("yalnız cevap", control["validation"]),
                  _row("yalnız cevap, viterbi", control["validation_viterbi"]), ""]
    lines += ["## Eğitim", "", "| Ayar | Tür | İterasyon | Max'ta durdu | Süre (s) |", "| --- | --- | ---: | :---: | ---: |"]
    for r in summary["grid"]:
        for t, s in r["training"].items():
            lines.append(f"| {r['name']} | {t} | {s['iterations']} | {'evet' if s['stopped_at_max_iterations'] else 'hayır'} "
                         f"| {s['train_seconds']:.1f} |")
    lines += ["", f"Toplam süre: {summary['total_seconds'] / 60:.1f} dk.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="grid on train, selection on validation, response-only control")
    tr.add_argument("--config", type=Path, default=ROOT / "configs/crf.json")
    tr.add_argument("--output", type=Path, required=True)
    tr.add_argument("--seed", type=int, default=42)
    def positive_int(value):
        number = int(value)
        if number <= 0:
            raise argparse.ArgumentTypeError("limit must be a positive integer")
        return number
    tr.add_argument("--limit-train", type=positive_int, help="first N good train responses (smoke runs only)")
    tr.add_argument("--limit-validation", type=positive_int, help="first N good validation responses (smoke runs only)")
    pr = sub.add_parser("predict", help="tag a split with saved models")
    pr.add_argument("--model-dir", type=Path, required=True)
    pr.add_argument("--split", choices=("validation", "test"), required=True)
    pr.add_argument("--processed-dir", default="data/processed")
    pr.add_argument("--output", type=Path, required=True)
    pr.add_argument("--limit", type=positive_int, help="first N good validation responses (smoke reproduction only)")
    args = ap.parse_args(argv)
    args.invocation = [str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]
    try:
        import pycrfsuite
    except ImportError:
        ap.error("python-crfsuite is missing: pip install -r requirements-crf.txt")
    if args.command == "train":
        summary = cmd_train(args, pycrfsuite)
        selected = summary["selected"]
        print(json.dumps({"selected": selected["name"], "decoding": selected["decoding"]["name"],
                          "validation_f1_any": round(selected["validation"]["any"]["f1"], 4),
                          "output": str(args.output), "total_seconds": round(summary["total_seconds"], 1)}, indent=2))
    else:
        print(json.dumps({"predictions": cmd_predict(args, pycrfsuite), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
