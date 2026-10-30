"""Character spans <-> tokens, and source windows <-> response.

All positions are character offsets into the original ``response``; the text
is never lower-cased, stripped or rewritten.

Token rule: a token belongs to a label if it shares at least one character
with it.

Overlap policy (docs/DATA_FORMAT.md):
* binary target ("any annotation"): overlapping labels are merged;
* per-type targets: one O/B/I sequence per annotation type, so labels of
  different types never compete for a token and nothing is lost;
* same-type overlapping labels are merged into their union; going back from
  tags gives that union. The raw labels in the data file stay untouched and
  are what scoring uses.

CRF (word level): ``word_level_example``.
Transformer (subword level, any HF *fast* tokenizer): ``encode_with_windows``,
then ``aggregate_window_scores`` maps window predictions back to the response.
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
from typing import Iterable, Sequence

from .schema import BIO, LABEL_TYPES

WORD_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
BIO_ID = {t: i for i, t in enumerate(BIO)}


# ---------------------------------------------------------------- basics
def word_tokenize(text: str) -> tuple[list, list]:
    """Regex word tokens with (start, end) offsets; punctuation is its own token."""
    toks, offs = [], []
    for m in WORD_RE.finditer(text):
        toks.append(m.group())
        offs.append((m.start(), m.end()))
    return toks, offs


def merge_intervals(intervals: Iterable[tuple]) -> list:
    """Union of [start, end) intervals that share characters; touching ones stay apart."""
    out = []
    for s, e in sorted(intervals):
        if out and s < out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def _interval_index(offsets: Sequence, intervals: Sequence) -> list:
    """Index of the first merged interval each token overlaps, or None. Offsets ascending."""
    res, j = [], 0
    for s, e in offsets:
        if e <= s:
            res.append(None)
            continue
        while j < len(intervals) and intervals[j][1] <= s:
            j += 1
        res.append(j if j < len(intervals) and intervals[j][0] < e else None)
    return res


def bio_tags(offsets: Sequence, labels: Sequence[dict], label_type: str | None = None, seq_start: int = 0) -> list:
    """O/B/I tags for tokens.

    label_type=None: any annotation (binary target); otherwise only that type.
    seq_start: where this token sequence begins in the response; a label that
    started earlier continues with I (used when a long response is split).
    """
    chosen = [l for l in labels if label_type is None or l["label_type"] == label_type]
    iv = merge_intervals((l["start"], l["end"]) for l in chosen)
    tags, seen = [], set()
    for i in _interval_index(offsets, iv):
        if i is None:
            tags.append("O")
        else:
            tags.append("I" if (i in seen or iv[i][0] < seq_start) else "B")
            seen.add(i)
    return tags


def bio_by_type(offsets: Sequence, labels: Sequence[dict], seq_start: int = 0) -> dict:
    return {t: bio_tags(offsets, labels, t, seq_start) for t in LABEL_TYPES}


def multi_hot(offsets: Sequence, labels: Sequence[dict]) -> list:
    """Per token, 0/1 for each of the four types (order = LABEL_TYPES)."""
    return [
        [int(e > s and any(l["label_type"] == t and l["start"] < e and s < l["end"] for l in labels))
         for t in LABEL_TYPES]
        for s, e in offsets
    ]


def tags_to_char_spans(offsets: Sequence, tags: Sequence[str], label_type: str | None = None) -> list:
    """O/B/I tags back to [{'start', 'end', 'label_type'}] character spans."""
    spans, cur = [], None
    for (s, e), t in zip(offsets, tags):
        if t == "O" or e <= s:
            if cur:
                spans.append(cur)
                cur = None
        elif t == "I" and cur:
            cur["end"] = e
        else:
            if cur:
                spans.append(cur)
            cur = {"start": s, "end": e, "label_type": label_type}
    if cur:
        spans.append(cur)
    return spans


# ---------------------------------------------------------------- CRF view
def word_level_example(record: dict) -> dict:
    """Word tokens of the response with the binary and per-type O/B/I tags."""
    toks, offs = word_tokenize(record["response"])
    return {
        "id": record["id"],
        "tokens": toks,
        "offsets": offs,
        "bio_any": bio_tags(offs, record["labels"]),
        "bio_by_type": bio_by_type(offs, record["labels"]),
    }


# ---------------------------------------------------------------- Transformer view
def _content_words(text: str) -> set:
    return {w.lower() for w in re.findall(r"\w+", text) if len(w) > 3}


def _chunks(n: int, size: int, overlap: int) -> list:
    if n <= size:
        return [(0, n)]
    step, out, a = max(1, size - overlap), [], 0
    while True:
        b = min(a + size, n)
        out.append((a, b))
        if b == n:
            return out
        a += step


def encode_with_windows(
    tokenizer,
    response: str,
    source_text: str,
    labels: Sequence[dict] | None = None,
    max_length: int = 512,
    context_stride: int = 128,
    min_context_tokens: int = 128,
    response_overlap: int = 64,
) -> list:
    """Encode one record as (response, source window) pairs.

    The response comes first and is only split if it alone is too long; the
    source is split into windows that overlap by ``context_stride`` tokens, so
    every source character is seen by at least one window. Window choice uses
    model inputs only (no labels).

    Each feature: input_ids, attention_mask, (token_type_ids), response_mask,
    response_offsets (absolute response offsets, None elsewhere),
    response_char_span, source_char_span, response_chunk, window_index,
    source_overlap, and with labels: labels_any (0/1), labels_bio (O/B/I ids),
    labels_multi (4 x 0/1), labels_bio_by_type (4 lists of O/B/I ids); every
    label is -100 (or None for labels_multi) outside the response.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer (offset mapping) is required")
    r_offs = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    c_offs = tokenizer(source_text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    c_ends = [o[1] for o in c_offs]
    n_special = tokenizer.num_special_tokens_to_add(pair=True)
    resp_budget = max_length - n_special - min_context_tokens
    if resp_budget <= 0:
        raise ValueError("max_length is too small for min_context_tokens")

    features = []
    for ci, (ra, rb) in enumerate(_chunks(len(r_offs), resp_budget, response_overlap)):
        rs = 0 if ra == 0 else r_offs[ra][0]
        re_ = len(response) if rb == len(r_offs) else r_offs[rb - 1][1]
        r_text = response[rs:re_]
        r_words = _content_words(r_text)
        ctx_budget = max(1, max_length - n_special - (rb - ra))
        i, wi = 0, 0
        while True:
            j = min(i + ctx_budget, len(c_offs))
            cs = 0 if i == 0 else c_offs[i][0]
            ce = len(source_text) if j == len(c_offs) else c_offs[j - 1][1]
            enc = tokenizer(r_text, source_text[cs:ce], truncation="only_second",
                            max_length=max_length, return_offsets_mapping=True)
            seq_ids = enc.sequence_ids()
            resp_abs, ctx_end = [], cs
            for sid, (s, e) in zip(seq_ids, enc["offset_mapping"]):
                resp_abs.append((s + rs, e + rs) if sid == 0 and e > s else None)
                if sid == 1 and e > s:
                    ctx_end = max(ctx_end, e + cs)
            feat = {k: list(v) for k, v in enc.items() if k != "offset_mapping"}
            feat.update(
                response_mask=[int(o is not None) for o in resp_abs],
                response_offsets=resp_abs,
                response_char_span=(rs, re_),
                source_char_span=(cs, ctx_end),
                response_chunk=ci,
                window_index=wi,
                source_overlap=(len(r_words & _content_words(source_text[cs:ctx_end])) / len(r_words)
                                if r_words else 0.0),
            )
            if labels is not None:
                pos = [k for k, o in enumerate(resp_abs) if o is not None]
                offs = [resp_abs[k] for k in pos]
                any_tags = bio_tags(offs, labels, None, rs)
                per_type = bio_by_type(offs, labels, rs)
                mh = multi_hot(offs, labels)
                n = len(resp_abs)
                l_any, l_bio, l_multi = [-100] * n, [-100] * n, [None] * n
                l_types = [[-100] * n for _ in LABEL_TYPES]
                for q, k in enumerate(pos):
                    l_any[k] = int(any_tags[q] != "O")
                    l_bio[k] = BIO_ID[any_tags[q]]
                    l_multi[k] = mh[q]
                    for ti, t in enumerate(LABEL_TYPES):
                        l_types[ti][k] = BIO_ID[per_type[t][q]]
                feat.update(labels_any=l_any, labels_bio=l_bio, labels_multi=l_multi, labels_bio_by_type=l_types)
            features.append(feat)

            if not c_offs or ctx_end >= c_ends[-1]:
                break
            last = bisect.bisect_right(c_ends, ctx_end) - 1
            i, wi = max(i + 1, last + 1 - context_stride), wi + 1
            if i >= len(c_offs):
                break
    return features


def select_windows(features: list, policy: str = "all") -> list:
    """'all' = every window; 'best' = per response chunk, the window whose source
    part shares the most content words with the response (model inputs only)."""
    if policy == "all":
        return features
    if policy == "best":
        best = {}
        for f in features:
            k = f["response_chunk"]
            if k not in best or f["source_overlap"] > best[k]["source_overlap"]:
                best[k] = f
        return [best[k] for k in sorted(best)]
    raise ValueError(f"unknown window policy {policy!r}")


def aggregate_window_scores(response_len: int, features: Sequence[dict], token_scores: Sequence[Sequence[float]],
                            across_windows: str = "min") -> list:
    """Per-character score for the response from per-token scores of each window.

    across_windows='min': a token stays flagged only if every source window
    flags it (support found in any window clears it). Overlapping response
    chunks are averaged. Call once per type for per-type scores.
    """
    reduce = {"min": min, "max": max, "mean": lambda x: sum(x) / len(x)}[across_windows]
    per_chunk = defaultdict(dict)
    for f, sc in zip(features, token_scores):
        for o, v in zip(f["response_offsets"], sc):
            if o is not None:
                per_chunk[f["response_chunk"]].setdefault(tuple(o), []).append(float(v))
    total, count = [0.0] * response_len, [0] * response_len
    for chunk in per_chunk.values():
        for (s, e), vals in chunk.items():
            v = reduce(vals)
            for c in range(s, e):
                total[c] += v
                count[c] += 1
    return [t / n if n else 0.0 for t, n in zip(total, count)]


def char_scores_to_spans(text: str, char_scores: Sequence[float], threshold: float = 0.5,
                         label_type: str | None = None) -> list:
    """Threshold character scores into [{'start', 'end', 'label_type'}] spans.

    Span edges never include whitespace; whitespace between two flagged
    stretches does not break a span.
    """
    runs, i, n = [], 0, len(text)
    while i < n:
        if char_scores[i] >= threshold and not text[i].isspace():
            j = i
            while j < n and char_scores[j] >= threshold and not text[j].isspace():
                j += 1
            if runs and text[runs[-1][1]:i].isspace():
                runs[-1][1] = j
            else:
                runs.append([i, j])
            i = j
        else:
            i += 1
    return [{"start": s, "end": e, "label_type": label_type} for s, e in runs]
