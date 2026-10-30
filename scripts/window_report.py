"""Tokenizer lengths, window counts and coverage on train + validation (2. kişi).

Needs `transformers` and the chosen tokenizer (run where Hugging Face is reachable):

    python scripts/window_report.py --tokenizer answerdotai/ModernBERT-base --max-length 512 1024 2048
    python scripts/window_report.py --tokenizer microsoft/deberta-v3-base --max-length 512

Per max_length it reports how many records need more than one source window,
features per record, how much source text a single window would cut, and a
round-trip check that every gold label survives labels -> windows -> spans.
The official test set is not read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data.alignment import aggregate_window_scores, char_scores_to_spans, encode_with_windows  # noqa: E402
from ragtruth_data.data import load_split  # noqa: E402


def _q(v, p):
    v = sorted(v)
    return v[int((len(v) - 1) * p)] if v else None


def measure(tok, records, max_length, stride):
    nfeat, multi, split_resp, lost = Counter(), 0, 0, 0
    cut_share = []
    t0 = time.time()
    for r in records:
        f = encode_with_windows(tok, r["response"], r["source_text"], r["labels"], max_length=max_length,
                                context_stride=stride)
        nfeat[r["split"]] += len(f)
        multi += len(f) > 1
        split_resp += len({x["response_chunk"] for x in f}) > 1
        first = f[0]["source_char_span"][1]
        cut_share.append(1 - first / max(1, len(r["source_text"])))
        if r["labels"]:
            sc = [[float(v == 1) if v != -100 else 0.0 for v in x["labels_any"]] for x in f]
            pred = char_scores_to_spans(r["response"], aggregate_window_scores(len(r["response"]), f, sc))
            for l in r["labels"]:
                chars = [c for c in range(l["start"], l["end"]) if not r["response"][c].isspace()]
                lost += not all(any(p["start"] <= c < p["end"] for p in pred) for c in chars)
    n = len(records)
    return {
        "features": dict(nfeat),
        "features_per_record": round(sum(nfeat.values()) / n, 3),
        "share_records_needing_several_windows": round(multi / n, 4),
        "records_with_split_response": split_resp,
        "single_window_source_cut": {"share_records_cut": round(sum(c > 0 for c in cut_share) / n, 4),
                                     "median_share_cut": round(_q(cut_share, .5), 4),
                                     "p95_share_cut": round(_q(cut_share, .95), 4)},
        "gold_labels_lost_in_round_trip": lost,
        "seconds": round(time.time() - t0, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer name or local path")
    ap.add_argument("--max-length", type=int, nargs="+", default=[512])
    ap.add_argument("--context-stride", type=int, default=128)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    records = load_split("train") + load_split("validation")
    report = {"tokenizer": args.tokenizer, "records": len(records), "splits": ["train", "validation"],
              "context_stride": args.context_stride, "by_max_length": {}}
    for L in args.max_length:
        report["by_max_length"][str(L)] = measure(tok, records, L, args.context_stride)
        print(L, json.dumps(report["by_max_length"][str(L)]), flush=True)
    out = args.output or ROOT / "docs" / f"window_report_{args.tokenizer.replace('/', '__')}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
