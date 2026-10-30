"""Model-ready features for the contextual Transformer (shared with 4. kişi).

    from transformers import AutoTokenizer
    from ragtruth_data.data import load_split
    from ragtruth_data.features import build_features, pad_batch

    tok = AutoTokenizer.from_pretrained("<model chosen with 4. kişi>")
    train = build_features(load_split("train"), tok, max_length=2048, window_policy="best")
    batch = pad_batch(train[:8], tok.pad_token_id, label_key="labels_any", return_tensors="pt")

Inference keeps every window; map window scores back with
alignment.aggregate_window_scores (one call per type for per-type scores).
"""

from __future__ import annotations

from typing import Iterable

from .alignment import encode_with_windows, select_windows


def build_features(records: Iterable[dict], tokenizer, max_length: int = 512, context_stride: int = 128,
                   min_context_tokens: int = 128, response_overlap: int = 64,
                   window_policy: str = "all", with_labels: bool = True) -> list:
    out = []
    for r in records:
        feats = encode_with_windows(tokenizer, r["response"], r["source_text"],
                                    r["labels"] if with_labels else None, max_length=max_length,
                                    context_stride=context_stride, min_context_tokens=min_context_tokens,
                                    response_overlap=response_overlap)
        for k, f in enumerate(select_windows(feats, window_policy)):
            f["id"] = r["id"]
            f["feature_index"] = k
            out.append(f)
    return out


def pad_batch(features: list, pad_token_id: int, label_key: str = "labels_any", return_tensors: str | None = None) -> dict:
    """Right-pad to the longest feature. Labels pad with -100.

    label_key may be labels_any, labels_bio, or labels_bio_by_type (-> [type][batch][seq]).
    """
    n = max(len(f["input_ids"]) for f in features)

    def pad(seq, value):
        return list(seq) + [value] * (n - len(seq))

    batch = {"input_ids": [pad(f["input_ids"], pad_token_id) for f in features],
             "attention_mask": [pad(f["attention_mask"], 0) for f in features],
             "response_mask": [pad(f["response_mask"], 0) for f in features]}
    if all("token_type_ids" in f for f in features):
        batch["token_type_ids"] = [pad(f["token_type_ids"], 0) for f in features]
    if all(label_key in f for f in features):
        if label_key == "labels_bio_by_type":
            batch["labels"] = [[pad(f[label_key][t], -100) for f in features] for t in range(len(features[0][label_key]))]
        else:
            batch["labels"] = [pad(f[label_key], -100) for f in features]
    if return_tensors == "pt":
        import torch

        batch = {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
    return batch
