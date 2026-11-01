"""Fresh encoder, original-word mean pooling, and four independent BIO CRFs.

The four types are a tensor batch dimension, so overlapping types never compete.
No model assets are loaded on import. Only answer subwords enter the word pools.
"""
from __future__ import annotations

from bisect import bisect_right
from functools import partial
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

import transformer as core
from ragtruth_data.alignment import bio_by_type, tags_to_char_spans, word_tokenize
from ragtruth_data.data import sha256_file
from ragtruth_data.features import pad_batch
from ragtruth_data.schema import BIO, LABEL_TYPES

ARCHITECTURE = "original-word-four-bio-crf-v1"
BIAS_GRID = [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def validate_config(config):
    core.validate_config(config)
    if config.get("positive_weighting", "none") != "none" or config.get("loss", "bce") != "bce":
        raise ValueError("structured model uses unweighted CRF NLL, without BCE/focal weighting")
    if config["aggregation"] != "min":
        raise ValueError("structured context aggregation requires min whole-vector selection")
    if config["window_policy"] != "best":
        raise ValueError("structured training requires best source window per answer chunk")


def bias_vector(value):
    if isinstance(value, dict):
        if set(value) != set(LABEL_TYPES):
            raise ValueError("O biases must name all four types")
        value = [value[kind] for kind in LABEL_TYPES]
    elif type(value) in (float, int):
        value = [float(value)] * len(LABEL_TYPES)
    if (not isinstance(value, (list, tuple)) or len(value) != len(LABEL_TYPES)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError("O bias must be a finite scalar or four-vector")
    return list(value)


class BatchedBIOCRF(nn.Module):
    """Four independent linear-chain CRFs with hard start-I/O-to-I constraints.

    Emissions are [batch, words, type, BIO]. Dynamic programming loops only
    over words, never over annotation types. Padding masks must be prefixes.
    """
    def __init__(self):
        super().__init__()
        self.start_transitions = nn.Parameter(torch.zeros(4, 3))
        self.end_transitions = nn.Parameter(torch.zeros(4, 3))
        self.transitions = nn.Parameter(torch.zeros(4, 3, 3))

    def _prepare(self, emissions, mask):
        if emissions.ndim != 4 or emissions.shape[-2:] != (4, 3) or mask.shape != emissions.shape[:2]:
            raise ValueError("CRF requires [batch, words, 4, 3] emissions and [batch, words] mask")
        mask = mask.bool()
        if mask.shape[1] > 1 and (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("CRF masks must be contiguous prefixes")
        if not torch.isfinite(emissions[mask]).all():
            raise ValueError("CRF active emissions must be finite")
        batch, words = mask.shape
        values = emissions.float().permute(0, 2, 1, 3).reshape(batch * 4, words, 3)
        active = mask[:, None, :].expand(batch, 4, words).reshape(batch * 4, words)
        # Ignored NaNs must never enter even an unselected DP branch: backward
        # through logsumexp can propagate them despite a later torch.where.
        values = values.masked_fill(~active[..., None], 0.0)
        start = self.start_transitions.float().clone()
        trans = self.transitions.float().clone()
        start[:, 2] = -torch.inf
        trans[:, 0, 2] = -torch.inf
        return (values, active, start.repeat(batch, 1), trans.repeat(batch, 1, 1),
                self.end_transitions.float().repeat(batch, 1))

    def log_partition(self, emissions, mask):
        values, active, start, trans, end = self._prepare(emissions, mask)
        if values.shape[1] == 0:
            return values.sum(dim=(1, 2)).reshape(emissions.shape[0], 4)
        alpha = start + values[:, 0]
        for index in range(1, values.shape[1]):
            candidate = torch.logsumexp(alpha[:, :, None] + trans, dim=1) + values[:, index]
            alpha = torch.where(active[:, index, None], candidate, alpha)
        result = torch.logsumexp(alpha + end, dim=1)
        return torch.where(active[:, 0], result, torch.zeros_like(result)).reshape(emissions.shape[0], 4)

    def neg_log_likelihood(self, emissions, tags, mask):
        values, active, start, trans, end = self._prepare(emissions, mask)
        if tags.shape != emissions.shape[:3]:
            raise ValueError("BIO tags must have [batch, words, 4] shape")
        if not values.shape[1] or not active.any():
            return emissions[mask.bool()].sum()
        labels = tags.permute(0, 2, 1).reshape(-1, tags.shape[1]).long()
        if ((labels[active] < 0) | (labels[active] > 2)).any():
            raise ValueError("active BIO tags must be O/B/I integers")
        labels = labels.masked_fill(~active, 0)
        if ((labels[:, 0] == 2) & active[:, 0]).any():
            raise ValueError("local BIO sequences cannot start with I; normalize chunk-start I to B")
        if ((labels[:, :-1] == 0) & (labels[:, 1:] == 2) & active[:, 1:]).any():
            raise ValueError("O-to-I is illegal in BIO gold tags")
        rows = torch.arange(values.shape[0], device=values.device)
        score = start[rows, labels[:, 0]]
        emit = values.gather(2, labels[..., None]).squeeze(-1)
        score = score + torch.where(active, emit, torch.zeros_like(emit)).sum(1)
        if labels.shape[1] > 1:
            edges = trans[rows[:, None], labels[:, :-1], labels[:, 1:]]
            score = score + torch.where(active[:, 1:], edges, torch.zeros_like(edges)).sum(1)
        last = active.sum(1).clamp_min(1) - 1
        score = score + end[rows, labels[rows, last]]
        score = torch.where(active[:, 0], score, torch.zeros_like(score))
        partition = self.log_partition(emissions, mask).reshape(-1)
        return (partition - score).sum() / active.sum()

    @torch.no_grad()
    def decode(self, emissions, mask, o_bias=0.0):
        values, active, start, trans, end = self._prepare(emissions, mask)
        bias = torch.tensor(bias_vector(o_bias), device=values.device, dtype=values.dtype)
        values = values.clone()
        values[:, :, 0] += bias.repeat(emissions.shape[0])[:, None]
        batch, words = mask.shape
        result = torch.full((batch * 4, words), -1, dtype=torch.long, device=values.device)
        if not words:
            return result.reshape(batch, 4, words).permute(0, 2, 1)
        score = start + values[:, 0]
        history = []
        for index in range(1, words):
            candidate, pointer = (score[:, :, None] + trans).max(1)
            candidate = candidate + values[:, index]
            score = torch.where(active[:, index, None], candidate, score)
            history.append(pointer)
        current = (score + end).argmax(1)
        rows = torch.arange(len(current), device=values.device)
        lengths = active.sum(1)
        for index in range(words - 1, -1, -1):
            valid = lengths > index
            result[:, index] = torch.where(valid, current, -1)
            if index:
                previous = history[index - 1][rows, current]
                current = torch.where(valid, previous, current)
        return result.reshape(batch, 4, words).permute(0, 2, 1)


def record_features(record, tokenizer, config, training=False, with_labels=True):
    """Attach original regex words, including words split across BPE chunks.

    One subword can overlap multiple regex words; UTF-8 byte-piece offsets can
    overlap too. Every overlapping answer piece contributes once to its pool.
    """
    _, all_offsets = word_tokenize(record["response"])
    ends = [end for _, end in all_offsets]
    features = core.record_features(record, tokenizer, config, training=training, with_labels=False)
    for feature in features:
        pools = {}
        for token, (offset, answer, attended) in enumerate(zip(
                feature["response_offsets"], feature["response_mask"], feature["attention_mask"])):
            if not answer or not attended or offset is None:
                continue
            start, end = offset
            index = bisect_right(ends, start)
            while index < len(all_offsets) and all_offsets[index][0] < end:
                pools.setdefault(index, []).append(token)
                index += 1
        indices = sorted(pools)
        if indices and indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("tokenizer omitted an internal original response word")
        feature["word_indices"] = indices
        feature["word_offsets"] = [all_offsets[index] for index in indices]
        feature["word_token_indices"] = [pools[index] for index in indices]
        if with_labels:
            offsets = feature["word_offsets"]
            columns = bio_by_type(offsets, record["labels"], seq_start=offsets[0][0] if offsets else 0)
            for values in columns.values():
                if values and values[0] == "I":
                    values[0] = "B"
            feature["labels_bio"] = [[BIO.index(columns[kind][index]) for kind in LABEL_TYPES]
                                     for index in range(len(offsets))]
    return features


def collate_features(features, pad_token_id):
    if not features:
        return {"input_ids": torch.empty((0, 0), dtype=torch.long),
                "attention_mask": torch.empty((0, 0), dtype=torch.long),
                "pool_indices": torch.empty(0, dtype=torch.long),
                "pool_offsets": torch.zeros(1, dtype=torch.long),
                "word_mask": torch.empty((0, 0), dtype=torch.bool)}
    batch = pad_batch(features, pad_token_id, return_tensors="pt")
    batch.pop("response_mask", None)
    length = batch["input_ids"].shape[1]
    words = max(len(feature["word_indices"]) for feature in features)
    mask = torch.zeros((len(features), words), dtype=torch.bool)
    labels = torch.full((len(features), words, 4), -100, dtype=torch.long)
    indices, offsets = [], [0]
    labeled = ["labels_bio" in feature for feature in features]
    if any(labeled) and not all(labeled):
        raise ValueError("a batch cannot mix labeled and unlabeled features")
    for row, feature in enumerate(features):
        count = len(feature["word_indices"])
        mask[row, :count] = True
        if labeled[row] and count:
            labels[row, :count] = torch.tensor(feature["labels_bio"], dtype=torch.long)
        for index in range(words):
            pieces = feature["word_token_indices"][index] if index < count else []
            if index < count and not pieces:
                raise ValueError("every active original word requires answer subwords")
            for token in pieces:
                if not (0 <= token < len(feature["input_ids"]) and feature["response_mask"][token]
                        and feature["attention_mask"][token]):
                    raise ValueError("source/special/padding positions cannot enter answer word pools")
                indices.append(row * length + token)
            offsets.append(len(indices))
    batch.update(pool_indices=torch.tensor(indices, dtype=torch.long),
                 pool_offsets=torch.tensor(offsets, dtype=torch.long), word_mask=mask)
    if all(labeled):
        batch["labels"] = labels
    return batch


class StructuredTransformer(nn.Module):
    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, 12)
        self.crf = BatchedBIOCRF()
        self.accepts_token_types = "token_type_ids" in inspect.signature(encoder.forward).parameters

    @classmethod
    def from_pretrained(cls, config):
        validate_config(config)
        enc_config = AutoConfig.from_pretrained(config["model_name"], revision=config["revision"])
        if enc_config.model_type == "modernbert":
            enc_config.reference_compile = False
        encoder = AutoModel.from_pretrained(config["model_name"], revision=config["revision"], config=enc_config,
                                             attn_implementation=config["attention_implementation"])
        if config["gradient_checkpointing"]:
            encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return cls(encoder, config["dropout"])

    def forward(self, input_ids, attention_mask, pool_indices, pool_offsets, word_mask,
                labels=None, token_type_ids=None):
        batch, words = word_mask.shape
        if not batch or not words:
            emissions = self.classifier.weight.sum() * 0 + torch.zeros(
                (batch, words, 4, 3), device=input_ids.device)
        else:
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if self.accepts_token_types and token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids
            hidden = self.encoder(**inputs).last_hidden_state
            pooled = F.embedding_bag(pool_indices, hidden.reshape(-1, hidden.shape[-1]), pool_offsets,
                                     mode="mean", include_last_offset=True).reshape(batch, words, -1)
            emissions = self.classifier(self.dropout(pooled)).reshape(batch, words, 4, 3)
        result = {"emissions": emissions}
        if labels is not None:
            result["loss"] = self.crf.neg_log_likelihood(emissions.float(), labels, word_mask)
        return result


class WindowDataset(IterableDataset):
    def __init__(self, records, tokenizer, config, training=False, epoch=0):
        super().__init__()
        self.records, self.tokenizer, self.config = records, tokenizer, config
        self.training, self.epoch = training, epoch

    def __iter__(self):
        order = list(range(len(self.records)))
        if self.training:
            random.Random(self.config["seed"] + self.epoch).shuffle(order)
        for index in order:
            yield from record_features(self.records[index], self.tokenizer, self.config, self.training)


def make_loader(records, tokenizer, config, training=False, epoch=0):
    return DataLoader(WindowDataset(records, tokenizer, config, training, epoch),
                      batch_size=config["batch_size"], num_workers=0,
                      collate_fn=partial(collate_features, pad_token_id=tokenizer.pad_token_id))


def run_epoch(model, loader, device, config, optimizer=None):
    """Word-weighted NLL/accumulation; empty answers never update AdamW."""
    training = optimizer is not None
    model.train(training)
    started = time.perf_counter()
    total_loss = total_words = windows = steps = pending_words = pending_batches = skipped = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if training:
        optimizer.zero_grad(set_to_none=True)

    def step():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(pending_words)
        nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(training):
        for batch in loader:
            windows += len(batch["input_ids"])
            count = int(batch["word_mask"].sum())
            if not count:
                skipped += 1
                continue
            batch = {key: value.to(device) for key, value in batch.items()}
            with core.autocast_context(device, config["precision"]):
                loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite structured CRF loss")
            total_loss += loss.detach().item() * count
            total_words += count
            if training:
                (loss * count).backward()
                pending_words += count
                pending_batches += 1
                if pending_batches == config["gradient_accumulation_steps"]:
                    step()
                    steps += 1
                    pending_words = pending_batches = 0
            if windows % 500 == 0:
                print(json.dumps({"phase": "train" if training else "validation", "windows": windows,
                                  "optimizer_steps": steps, "loss_so_far": total_loss / total_words,
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
        if training and pending_words:
            step()
            steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {"loss": total_loss / total_words if total_words else None, "answer_words": total_words,
            "windows": windows, "optimizer_steps": steps, "empty_batches_skipped": skipped,
            "elapsed_seconds": elapsed, "answer_words_per_second": total_words / elapsed if elapsed else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}


def aggregate_word_scores(record, features, scores):
    """Whole-vector conservative source-window choice, then answer-chunk mean."""
    _, offsets = word_tokenize(record["response"])
    if len(features) != len(scores):
        raise ValueError("each feature needs its word scores")
    chunks = {}
    for feature, values in zip(features, scores):
        values = torch.as_tensor(values, dtype=torch.float32)
        if values.numel() == 0:
            values = values.reshape(0, 4, 3)
        if (values.shape != (len(feature["word_indices"]), 4, 3) or not torch.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()
                or not torch.allclose(values.sum(-1), torch.ones_like(values[..., 0]), atol=1e-5)):
            raise ValueError("word probabilities must be normalized finite [words, 4, 3] vectors")
        for index, word in enumerate(feature["word_indices"]):
            key = (feature["response_chunk"], word)
            current = values[index]
            if key in chunks:
                old = chunks[key]
                choose = current[:, 1:].sum(-1) < old[:, 1:].sum(-1)
                chunks[key] = torch.where(choose[:, None], current, old)
            else:
                chunks[key] = current.clone()
    total = torch.zeros((len(offsets), 4, 3))
    counts = torch.zeros(len(offsets))
    for (_, word), values in chunks.items():
        total[word] += values
        counts[word] += 1
    if len(offsets) and (counts == 0).any():
        raise ValueError("inference windows omitted original response words")
    probabilities = total / counts.clamp_min(1)[:, None, None]
    return {"id": record["id"], "word_offsets": [list(value) for value in offsets],
            "response_sha256": hashlib.sha256(record["response"].encode("utf-8")).hexdigest(),
            "probabilities": probabilities.tolist()}


@torch.inference_mode()
def predict_word_scores(model, records, tokenizer, config, device):
    model.eval()
    for record in records:
        features = record_features(record, tokenizer, config, with_labels=False)
        scores = []
        for start in range(0, len(features), config["batch_size"]):
            group = features[start:start + config["batch_size"]]
            batch = {key: value.to(device) for key, value in collate_features(group, tokenizer.pad_token_id).items()}
            with core.autocast_context(device, config["precision"]):
                probabilities = model(**batch)["emissions"].float().softmax(-1).cpu()
            scores.extend(probabilities[index, :len(feature["word_indices"])].tolist()
                          for index, feature in enumerate(group))
        yield aggregate_word_scores(record, features, scores)


def decode_word_scores(scores, crf, o_bias=0.0):
    offsets, raw = scores["word_offsets"], scores["probabilities"]
    probabilities = torch.as_tensor(raw, dtype=torch.float32).reshape(-1, 4, 3)
    if (len(offsets) != len(probabilities) or not torch.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=1e-5)):
        raise ValueError("cached word scores must be aligned, finite normalized BIO probabilities")
    if any(len(pair) != 2 or any(type(v) is not int for v in pair) or not 0 <= pair[0] < pair[1]
           or (index and offsets[index - 1][1] > pair[0]) for index, pair in enumerate(offsets)):
        raise ValueError("cached original word offsets must be ordered, nonoverlapping intervals")
    device = crf.transitions.device
    values = probabilities.clamp_min(torch.finfo(torch.float32).tiny).log().to(device)[None]
    tags = crf.decode(values, torch.ones((1, len(offsets)), dtype=torch.bool, device=device), o_bias)[0].cpu()
    spans = []
    for index, kind in enumerate(LABEL_TYPES):
        spans.extend(tags_to_char_spans(offsets, [BIO[int(value)] for value in tags[:, index]], kind))
    spans.sort(key=lambda span: (span["start"], span["end"], LABEL_TYPES.index(span["label_type"])))
    return {"id": scores["id"], "spans": spans}


def predict_records(model, records, tokenizer, config, device, o_bias=0.0):
    # Decoding short chains on CPU avoids hundreds of tiny accelerator kernels.
    crf = BatchedBIOCRF()
    crf.load_state_dict({key: value.detach().cpu() for key, value in model.crf.state_dict().items()})
    for scores in predict_word_scores(model, records, tokenizer, config, device):
        yield decode_word_scores(scores, crf, o_bias)


def save_checkpoint(path, model, tokenizer, config, metadata):
    validate_config(config)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.encoder.save_pretrained(path / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(path / "tokenizer")
    state = {f"classifier.{key}": value for key, value in model.classifier.state_dict().items()}
    state.update({f"crf.{key}": value for key, value in model.crf.state_dict().items()})
    save_file({key: value.detach().cpu().contiguous() for key, value in state.items()}, str(path / "head.safetensors"))
    write_json(path / "checkpoint.json", {"format_version": 1, "architecture": ARCHITECTURE,
        "label_types": list(LABEL_TYPES), "bio_order": list(BIO), "config": config, "metadata": metadata,
        "files_sha256": {file.relative_to(path).as_posix(): sha256_file(file)
                         for file in sorted(path.rglob("*")) if file.is_file()}})


def read_checkpoint(path):
    path = Path(path)
    saved = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if (saved.get("format_version") != 1 or saved.get("architecture") != ARCHITECTURE
            or saved.get("label_types") != list(LABEL_TYPES) or saved.get("bio_order") != list(BIO)):
        raise ValueError("not a compatible original-word structured checkpoint")
    validate_config(saved["config"])
    expected = saved.get("files_sha256", {})
    actual = {file.relative_to(path).as_posix() for file in path.rglob("*")
              if file.is_file() and file != path / "checkpoint.json"}
    if not expected or set(expected) != actual:
        raise ValueError("structured checkpoint file manifest differs")
    for name, checksum in expected.items():
        if sha256_file(path / name) != checksum:
            raise ValueError(f"structured checkpoint checksum mismatch: {name}")
    return saved


def load_crf(path):
    read_checkpoint(path)
    state = load_file(str(Path(path) / "head.safetensors"))
    crf = BatchedBIOCRF()
    crf.load_state_dict({key.removeprefix("crf."): value for key, value in state.items() if key.startswith("crf.")})
    return crf.eval()


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    saved = read_checkpoint(path)
    config = saved["config"]
    enc_config = AutoConfig.from_pretrained(path / "encoder", local_files_only=True)
    if enc_config.model_type == "modernbert":
        enc_config.reference_compile = False
    encoder = AutoModel.from_pretrained(path / "encoder", config=enc_config, local_files_only=True,
                                        attn_implementation=config["attention_implementation"])
    model = StructuredTransformer(encoder, config["dropout"])
    state = load_file(str(path / "head.safetensors"))
    model.classifier.load_state_dict({key.removeprefix("classifier."): value for key, value in state.items()
                                      if key.startswith("classifier.")})
    model.crf.load_state_dict({key.removeprefix("crf."): value for key, value in state.items() if key.startswith("crf.")})
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True, use_fast=True)
    core.validate_tokenizer(tokenizer, config, encoder.config)
    return model.to(device).eval(), tokenizer, saved
