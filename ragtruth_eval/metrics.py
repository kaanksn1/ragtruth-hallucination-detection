"""Strict, overlap-preserving project metrics on original character offsets.

Character metrics include whitespace and count Python Unicode code points.
Exact metrics deduplicate identical spans, retain overlapping/touching bounds,
and do not silently merge raw annotations. Every empty-denominator PRF is zero.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from ragtruth_data.schema import LABEL_TYPES

METRIC_VERSION = "ragtruth-project-eval-v1"


def _id(value):
    if not isinstance(value, str) or not value:
        raise ValueError("response IDs must be nonempty strings")
    return value


def validate_spans(spans, response=None):
    if not isinstance(spans, (list, tuple)):
        raise ValueError("spans/labels must be a list")
    result = []
    for span in spans:
        if not isinstance(span, Mapping) or not {"start", "end", "label_type"} <= set(span):
            raise ValueError("each span needs start, end and label_type")
        start, end, kind = span["start"], span["end"], span["label_type"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("span offsets must be integers with 0 <= start < end")
        if kind not in LABEL_TYPES:
            raise ValueError(f"unknown label_type: {kind!r}")
        if response is not None:
            if end > len(response):
                raise ValueError("span offsets exceed original response length")
            if "text" in span and span["text"] != response[start:end]:
                raise ValueError("gold span text differs from its character offsets")
        result.append({"start": start, "end": end, "label_type": kind})
    return result


def _validate_maps(gold_by_id, pred_by_id):
    if not isinstance(gold_by_id, Mapping) or not isinstance(pred_by_id, Mapping):
        raise ValueError("gold and predictions must be mappings from response IDs to span lists")
    if set(gold_by_id) != set(pred_by_id):
        missing = len(set(gold_by_id) - set(pred_by_id))
        extra = len(set(pred_by_id) - set(gold_by_id))
        raise ValueError(f"prediction IDs must exactly match gold IDs (missing={missing}, extra={extra})")
    if not gold_by_id:
        raise ValueError("cannot evaluate an empty response collection")
    gold, predicted = {}, {}
    for rid in gold_by_id:
        _id(rid)
        gold[rid] = validate_spans(gold_by_id[rid])
        predicted[rid] = validate_spans(pred_by_id[rid])
    return gold, predicted


def _union(spans, kind=None):
    result = []
    for start, end in sorted((s["start"], s["end"]) for s in spans if kind is None or s["label_type"] == kind):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _intersection_length(first, second):
    i = j = total = 0
    while i < len(first) and j < len(second):
        total += max(0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return total


def _prf(tp, fp, fn, unit):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "gold_" + unit: tp + fn, "predicted_" + unit: tp + fp}


def character_scores(gold_by_id, pred_by_id):
    """Micro character PRF; flat keys match the existing CRF selection helper.

    This low-level API validates IDs/types/offset ordering. Use evaluate_records
    or the CLI for response-bound checks and metadata breakdowns.
    """
    gold, predicted = _validate_maps(gold_by_id, pred_by_id)
    result = {}
    for kind in (None, *LABEL_TYPES):
        tp = fp = fn = gold_responses = predicted_responses = 0
        for rid in gold:
            g, p = _union(gold[rid], kind), _union(predicted[rid], kind)
            overlap = _intersection_length(g, p)
            tp += overlap
            fp += sum(end - start for start, end in p) - overlap
            fn += sum(end - start for start, end in g) - overlap
            gold_responses += bool(g)
            predicted_responses += bool(p)
        result[kind or "any"] = dict(_prf(tp, fp, fn, "chars"),
                                     gold_responses=gold_responses, predicted_responses=predicted_responses)
    result["macro_f1_types"] = sum(result[k]["f1"] for k in LABEL_TYPES) / len(LABEL_TYPES)
    result["micro_typed"] = _prf(*(sum(result[k][name] for k in LABEL_TYPES) for name in ("tp", "fp", "fn")), "chars")
    return result


def exact_span_scores(gold_by_id, pred_by_id):
    """Exact boundaries; any ignores types, typed requires correct type as well."""
    gold, predicted = _validate_maps(gold_by_id, pred_by_id)
    output = {}
    for mode in ("any", "typed", *LABEL_TYPES):
        tp = fp = fn = 0
        for rid in gold:
            def unique(spans):
                chosen = spans if mode in {"any", "typed"} else [s for s in spans if s["label_type"] == mode]
                return {(s["start"], s["end"]) if mode == "any" else (s["start"], s["end"], s["label_type"])
                        for s in chosen}
            g, p = unique(gold[rid]), unique(predicted[rid])
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
        output[mode] = _prf(tp, fp, fn, "spans")
    output["macro_f1_types"] = sum(output[k]["f1"] for k in LABEL_TYPES) / len(LABEL_TYPES)
    return output


def evaluate_records(records, predictions, *, breakdowns=True):
    """Score already-filtered original records and {id, spans} predictions.

    predictions may be JSONL-like rows or an ID->spans mapping. Records are not
    silently filtered here: the CLI's load_gold applies the quality policy.
    """
    gold, metadata = {}, {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("gold records must be objects")
        rid = _id(record.get("id"))
        if rid in gold:
            raise ValueError("duplicate gold response ID")
        if not isinstance(record.get("response"), str):
            raise ValueError("gold response must be original text")
        gold[rid] = validate_spans(record.get("labels"), record["response"])
        metadata[rid] = record
    predicted = {}
    rows = ({"id": rid, "spans": spans} for rid, spans in predictions.items()) if isinstance(predictions, Mapping) else predictions
    for prediction in rows:
        if not isinstance(prediction, Mapping) or set(prediction) != {"id", "spans"}:
            raise ValueError("each prediction must contain exactly id and spans")
        rid = _id(prediction["id"])
        if rid in predicted:
            raise ValueError("duplicate prediction response ID")
        if rid not in metadata:
            raise ValueError("prediction IDs must exactly match gold IDs")
        if not isinstance(prediction["spans"], (list, tuple)) or any(
                not isinstance(span, Mapping) or set(span) != {"start", "end", "label_type"}
                for span in prediction["spans"]):
            raise ValueError("prediction spans must contain exactly start, end and label_type")
        predicted[rid] = validate_spans(prediction["spans"], metadata[rid]["response"])
    character = character_scores(gold, predicted)
    exact = exact_span_scores(gold, predicted)
    result = {"metric_version": METRIC_VERSION, "responses": len(gold), "character": character, "span_exact": exact,
              "raw_gold_spans": sum(len(spans) for spans in gold.values()),
              "raw_predicted_spans": sum(len(spans) for spans in predicted.values()),
              "unique_gold_typed_spans": exact["typed"]["gold_spans"],
              "unique_predicted_typed_spans": exact["typed"]["predicted_spans"]}
    if breakdowns:
        result["breakdowns"] = {}
        for field in ("task_type", "generator"):
            groups = defaultdict(list)
            for rid, record in metadata.items():
                value = record.get(field, "__missing__")
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{field} metadata must be a nonempty string")
                groups[value].append(rid)
            result["breakdowns"][field] = {value: evaluate_records([metadata[rid] for rid in ids],
                    {rid: predicted[rid] for rid in ids}, breakdowns=False) for value, ids in sorted(groups.items())}
    return result
