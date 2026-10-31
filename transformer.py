"""Person 4: contextual, four-label token classifier and shared-data adapter.

No pretrained weights are downloaded on import. Training/prediction CLI:
    python scripts/transformer.py --help
"""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial
import inspect
import json
import math
from pathlib import Path
import random
import re
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
from safetensors.torch import load_file, save_file

from ragtruth_data.alignment import aggregate_window_scores, char_scores_to_spans
from ragtruth_data.features import build_features, pad_batch
from ragtruth_data.data import sha256_file
from ragtruth_data.schema import LABEL_TYPES


def validate_config(config):
    for key in ("max_length", "min_context_tokens", "batch_size", "epochs",
                "gradient_accumulation_steps"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("context_stride", "response_overlap", "seed"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 <= config["dropout"] < 1 or not 0 < config["threshold"] < 1:
        raise ValueError("dropout must be in [0,1), threshold in (0,1)")
    if "thresholds_by_type" in config:
        thresholds = config["thresholds_by_type"]
        if (not isinstance(thresholds, dict) or set(thresholds) != set(LABEL_TYPES)
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1
                       for value in thresholds.values())):
            raise ValueError("thresholds_by_type must contain four finite thresholds in (0,1)")
    weighting = config.get("positive_weighting", "none")
    if weighting not in {"none", "sqrt_capped"}:
        raise ValueError("positive_weighting must be none or sqrt_capped")
    cap = config.get("positive_weighting_cap", 50.0)
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap < 1:
        raise ValueError("positive_weighting_cap must be finite and at least one")
    if "positive_weights" in config:
        validate_positive_weights(config["positive_weights"])
        if any(weight > cap for weight in config["positive_weights"]):
            raise ValueError("positive_weights exceed the declared cap")
        if weighting == "none" and any(weight != 1 for weight in config["positive_weights"]):
            raise ValueError("non-unit positive_weights require sqrt_capped weighting")
    for key, allowed in (("window_policy", {"best", "all"}),
                         ("aggregation", {"min", "mean", "max"}),
                         ("precision", {"fp32", "bf16"}),
                         ("attention_implementation", {"sdpa", "eager"}),
                         ("device", {"auto", "cpu", "cuda"})):
        if config[key] not in allowed:
            raise ValueError(f"unsupported {key}: {config[key]}")
    if type(config["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be boolean")
    if not config["model_name"] or not config["revision"]:
        raise ValueError("model_name and revision must be recorded")
    if not isinstance(config["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("revision must be a pinned 40-character Hub commit SHA")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA/ROCm device requested but torch.cuda.is_available() is false")
    return torch.device(name)


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 requires a supporting CUDA/ROCm device; use fp32 on CPU")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def validate_positive_weights(weights):
    if (not isinstance(weights, (list, tuple)) or len(weights) != len(LABEL_TYPES)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 1 for value in weights)):
        raise ValueError("positive_weights must contain four finite values >= 1")


def weights_from_counts(positive, total, cap=50.0):
    """Per-type min(cap, max(1, sqrt(negative/positive))).

    A zero-positive class gets the finite cap; this cannot invent positives.
    Counts are response-token targets, not raw span or character frequencies.
    """
    if type(total) is not int or total <= 0 or len(positive) != len(LABEL_TYPES):
        raise ValueError("positive counts require four types and a nonempty token total")
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap < 1:
        raise ValueError("positive weight cap must be finite and >= 1")
    if any(type(count) is not int or not 0 <= count <= total for count in positive):
        raise ValueError("positive token counts must be integers between zero and total")
    return [float(cap if count == 0 else min(cap, max(1.0, math.sqrt((total - count) / count))))
            for count in positive]


def count_training_positive_weights(records, tokenizer, config, *, split):
    """Stream the selected TRAIN windows; never accept validation/test counts.

    WindowDataset uses a local random.Random, so this pass does not consume the
    torch RNG state or alter the seed-42 model/head initialization.
    """
    if split != "train":
        raise ValueError("positive weights may only be counted from train")
    started = time.perf_counter()
    positive, total, windows = [0] * len(LABEL_TYPES), 0, 0
    for feature in WindowDataset(records, tokenizer, config, training=True, epoch=0):
        windows += 1
        require_length = len(feature["input_ids"])
        if any(len(feature[key]) != require_length for key in ("labels_multi", "response_mask", "attention_mask")):
            raise ValueError("training weight targets/masks are misaligned")
        for target, response, attended in zip(feature["labels_multi"], feature["response_mask"], feature["attention_mask"]):
            if not response or not attended:
                continue
            if not isinstance(target, (list, tuple)) or len(target) != len(LABEL_TYPES) or any(value not in (0, 1) for value in target):
                raise ValueError("training positive-weight targets must be four binary values")
            total += 1
            for index, value in enumerate(target):
                positive[index] += int(value)
        if windows % 1000 == 0:
            print(json.dumps({"phase": "train_positive_weight_count", "windows": windows,
                              "response_tokens": total}), flush=True)
    cap = config.get("positive_weighting_cap", 50.0)
    weights = weights_from_counts(positive, total, cap)
    return {"method": "sqrt_capped", "split": "train", "label_types": list(LABEL_TYPES),
            "cap": cap, "records": len(records), "windows": windows, "response_tokens": total,
            "positive_response_tokens": dict(zip(LABEL_TYPES, positive)),
            "negative_response_tokens": {kind: total - count for kind, count in zip(LABEL_TYPES, positive)},
            "weights": dict(zip(LABEL_TYPES, weights)), "weights_vector": weights,
            "window_policy": config["window_policy"], "mask_policy": "response_mask & attention_mask",
            "zero_positive_policy": "cap; no positive targets are created",
            "elapsed_seconds": time.perf_counter() - started}


def masked_bce_loss(logits, labels, response_mask, attention_mask, pos_weight=None):
    """Mean BCE over valid answer tokens AND four independent label columns.

    Index before BCE: ignored targets/logits cannot contaminate the loss.
    An empty selection has a differentiable zero, including a zero-size batch.
    """
    if logits.shape != labels.shape or logits.shape[-1] != len(LABEL_TYPES):
        raise ValueError("logits and labels must have matching [batch, sequence, 4] shapes")
    if response_mask.shape != logits.shape[:2] or attention_mask.shape != response_mask.shape:
        raise ValueError("masks must match [batch, sequence]")
    active = response_mask.bool() & attention_mask.bool()
    selected = logits[active].float()
    if not selected.numel():
        return selected.sum()
    if pos_weight is not None:
        if pos_weight.shape != (len(LABEL_TYPES),) or not torch.isfinite(pos_weight).all() or (pos_weight < 1).any():
            raise ValueError("pos_weight must be a finite four-vector >= 1")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)(selected, labels[active].float())


class HallucinationTransformer(nn.Module):
    def __init__(self, encoder, dropout=0.1, positive_weights=None):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, len(LABEL_TYPES))
        if positive_weights is not None:
            validate_positive_weights(positive_weights)
        # The vector is serialized in checkpoint config/metadata, not in the
        # old four-logit head file. Inference logits receive no prior correction.
        self.register_buffer("positive_weights", None if positive_weights is None else
                             torch.tensor(positive_weights, dtype=torch.float32), persistent=False)
        self.accepts_token_types = "token_type_ids" in inspect.signature(encoder.forward).parameters

    @classmethod
    def from_pretrained(cls, config):
        if config.get("positive_weighting", "none") == "sqrt_capped" and "positive_weights" not in config:
            raise ValueError("derive train-only positive_weights before constructing the weighted model")
        enc_config = AutoConfig.from_pretrained(config["model_name"], revision=config["revision"])
        # ModernBERT's optional torch.compile path is unsuitable for portable CPU/Windows smoke runs.
        if enc_config.model_type == "modernbert":
            enc_config.reference_compile = False
        encoder = AutoModel.from_pretrained(
            config["model_name"], revision=config["revision"], config=enc_config,
            attn_implementation=config["attention_implementation"],
        )
        if config["gradient_checkpointing"]:
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return cls(encoder, config["dropout"], config.get("positive_weights"))

    def forward(self, input_ids, attention_mask, response_mask, labels=None, token_type_ids=None):
        if input_ids.numel() == 0:
            logits = self.classifier.weight.sum() * 0 + self.classifier.weight.new_zeros(
                (*input_ids.shape, len(LABEL_TYPES)))
        else:
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if self.accepts_token_types and token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids
            hidden = self.encoder(**inputs).last_hidden_state
            logits = self.classifier(self.dropout(hidden))
        loss = None if labels is None else masked_bce_loss(logits, labels, response_mask, attention_mask, self.positive_weights)
        return {"logits": logits, "loss": loss}


def validate_tokenizer(tokenizer, config, encoder_config=None):
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError("a fast tokenizer with a padding token is required")
    budget = config["max_length"] - tokenizer.num_special_tokens_to_add(pair=True) - config["min_context_tokens"]
    if budget <= 0 or config["response_overlap"] >= budget:
        raise ValueError("max_length must leave an answer budget larger than response_overlap")
    if encoder_config is not None and config["max_length"] > encoder_config.max_position_embeddings:
        raise ValueError("max_length exceeds the encoder position limit")


def record_features(record, tokenizer, config, training=False, with_labels=True):
    return build_features(
        [record], tokenizer, max_length=config["max_length"],
        context_stride=config["context_stride"], min_context_tokens=config["min_context_tokens"],
        response_overlap=config["response_overlap"],
        window_policy=config["window_policy"] if training else "all", with_labels=with_labels,
    )


class WindowDataset(IterableDataset):
    """Stream one response's windows at a time; do not cache the entire corpus's token targets."""
    def __init__(self, records, tokenizer, config, training=False, epoch=0):
        self.records, self.tokenizer, self.config = records, tokenizer, config
        self.training, self.epoch = training, epoch

    def __iter__(self):
        indices = list(range(len(self.records)))
        if self.training:
            random.Random(self.config["seed"] + self.epoch).shuffle(indices)
        for index in indices:
            yield from record_features(self.records[index], self.tokenizer, self.config, self.training)


def collate_features(features, pad_token_id):
    if not features:
        return {"input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "response_mask": torch.empty((0, 0), dtype=torch.long),
                "labels": torch.empty((0, 0, len(LABEL_TYPES)), dtype=torch.float32)}
    return pad_batch(features, pad_token_id, label_key="labels_multi", return_tensors="pt")


def make_loader(records, tokenizer, config, training=False, epoch=0):
    return DataLoader(WindowDataset(records, tokenizer, config, training, epoch),
                      batch_size=config["batch_size"], num_workers=0,
                      collate_fn=partial(collate_features, pad_token_id=tokenizer.pad_token_id))


def run_epoch(model, loader, device, config, optimizer=None):
    """Token-weighted accumulation, including the short last group of batches.

    Skip optimizer steps on empty answers: even zero gradients would otherwise
    change AdamW parameters through weight decay/momentum.
    """
    training = optimizer is not None
    model.train(training)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    total_loss = total_tokens = windows = steps = skipped = 0
    pending_tokens = pending_batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    def step(token_count):
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(token_count)
        nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(training):
        for batch in loader:
            windows += len(batch["input_ids"])
            count = int((batch["response_mask"].bool() & batch["attention_mask"].bool()).sum())
            if count == 0:
                skipped += 1
                continue
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, config["precision"]):
                loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite Transformer loss")
            total_loss += loss.detach().item() * count
            total_tokens += count
            if training:
                (loss * count).backward()
                pending_tokens += count
                pending_batches += 1
                if pending_batches == config["gradient_accumulation_steps"]:
                    step(pending_tokens)
                    steps += 1
                    pending_tokens = pending_batches = 0
            if windows % 500 == 0:
                print(json.dumps({"phase": "train" if training else "validation", "windows": windows,
                    "optimizer_steps": steps, "elapsed_seconds": time.perf_counter() - started,
                    "loss_so_far": total_loss / total_tokens}), flush=True)
        if training and pending_tokens:
            step(pending_tokens)
            steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {"loss": total_loss / total_tokens if total_tokens else None, "answer_tokens": total_tokens,
            "windows": windows, "optimizer_steps": steps, "empty_batches_skipped": skipped,
            "elapsed_seconds": elapsed, "answer_tokens_per_second": total_tokens / elapsed if elapsed else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}


def aggregate_record_scores(record, features, scores, config):
    """Four character-score arrays, after the unchanged window policy."""
    if len(features) != len(scores):
        raise ValueError("each window needs its own scores")
    for feature, values in zip(features, scores):
        if len(values) != len(feature["input_ids"]) or any(len(v) != len(LABEL_TYPES) for v in values):
            raise ValueError("window scores must align with input tokens and four types")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for v in values for x in v):
            raise ValueError("scores must be finite sigmoid probabilities")
    return [aggregate_window_scores(
            len(record["response"]), features, [[v[index] for v in row] for row in scores],
            config["aggregation"],
        ) for index in range(len(LABEL_TYPES))]


def decode_character_scores(record, scores_by_type, config):
    if len(scores_by_type) != len(LABEL_TYPES) or any(len(values) != len(record["response"]) for values in scores_by_type):
        raise ValueError("four character-score arrays must match the original response length")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for values in scores_by_type for value in values):
        raise ValueError("character scores must be finite probabilities in [0,1]")
    spans = []
    for label_type, char_scores in zip(LABEL_TYPES, scores_by_type):
        threshold = config.get("thresholds_by_type", {}).get(label_type, config["threshold"])
        spans.extend(char_scores_to_spans(record["response"], char_scores, threshold, label_type))
    spans.sort(key=lambda span: (span["start"], span["end"], LABEL_TYPES.index(span["label_type"])))
    return {"id": record["id"], "spans": spans}


def decode_record(record, features, scores, config):
    """Scores: sigmoid probabilities [window][sequence][four LABEL_TYPES]."""
    return decode_character_scores(record, aggregate_record_scores(record, features, scores, config), config)


@torch.inference_mode()
def predict_character_scores(model, records, tokenizer, config, device):
    model.eval()
    for record in records:
        features = record_features(record, tokenizer, config, with_labels=False)
        scores = []
        for start in range(0, len(features), config["batch_size"]):
            group = features[start:start + config["batch_size"]]
            batch = {key: value.to(device) for key, value in collate_features(group, tokenizer.pad_token_id).items()}
            with autocast_context(device, config["precision"]):
                probabilities = model(**batch)["logits"].float().sigmoid().cpu()
            scores.extend(probabilities[i, :len(feature["input_ids"])].tolist() for i, feature in enumerate(group))
        yield record, aggregate_record_scores(record, features, scores, config)


def predict_records(model, records, tokenizer, config, device):
    for record, scores_by_type in predict_character_scores(model, records, tokenizer, config, device):
        yield decode_character_scores(record, scores_by_type, config)


def save_checkpoint(path, model, tokenizer, config, metadata):
    """Self-contained inference checkpoint; intentionally no optimizer/resume state."""
    path = Path(path)
    if config.get("positive_weighting", "none") == "sqrt_capped" and "positive_weights" not in config:
        raise ValueError("weighted checkpoint requires recorded train-derived positive_weights")
    actual_weights = model.positive_weights
    expected_weights = config.get("positive_weights")
    if actual_weights is not None and (expected_weights is None or not torch.equal(
            actual_weights.detach().cpu(), torch.tensor(expected_weights, dtype=torch.float32))):
        raise ValueError("checkpoint config does not match the model's positive weights")
    if actual_weights is None and expected_weights is not None and any(value != 1 for value in expected_weights):
        raise ValueError("checkpoint config declares weights missing from the model")
    path.mkdir(parents=True, exist_ok=False)
    model.encoder.save_pretrained(path / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(path / "tokenizer")
    save_file({key: value.detach().cpu().contiguous() for key, value in model.classifier.state_dict().items()},
              str(path / "head.safetensors"))
    (path / "checkpoint.json").write_text(json.dumps({
        "format_version": 2, "label_types": list(LABEL_TYPES), "config": config, "metadata": metadata,
        "files_sha256": {p.relative_to(path).as_posix(): sha256_file(p)
                         for p in sorted(path.rglob("*")) if p.is_file()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    saved = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if saved["format_version"] != 2 or saved["label_types"] != list(LABEL_TYPES):
        raise ValueError("checkpoint version/label order does not match the shared data contract")
    expected = saved.get("files_sha256", {})
    actual_files = {p.relative_to(path).as_posix() for p in path.rglob("*")
                    if p.is_file() and p != path / "checkpoint.json"}
    if not expected or set(expected) != actual_files:
        raise ValueError("checkpoint checksum manifest does not match its files")
    for name, checksum in expected.items():
        if sha256_file(path / name) != checksum:
            raise ValueError(f"checkpoint checksum mismatch: {name}")
    config = saved["config"]
    validate_config(config)
    if config.get("positive_weighting", "none") == "sqrt_capped" and "positive_weights" not in config:
        raise ValueError("weighted checkpoint is missing its train-derived positive_weights")
    enc_config = AutoConfig.from_pretrained(path / "encoder", local_files_only=True)
    # ModernBertConfig.to_dict intentionally omits reference_compile. Restore
    # the same portable policy used for initial training before model loading.
    if enc_config.model_type == "modernbert":
        enc_config.reference_compile = False
    encoder = AutoModel.from_pretrained(path / "encoder", config=enc_config, local_files_only=True,
                                        attn_implementation=config["attention_implementation"])
    model = HallucinationTransformer(encoder, config["dropout"], config.get("positive_weights"))
    model.classifier.load_state_dict(load_file(str(path / "head.safetensors")))
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True, use_fast=True)
    validate_tokenizer(tokenizer, config, encoder.config)
    return model.to(device).eval(), tokenizer, saved
