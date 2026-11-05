"""Post-feedback response-only word-level RNN/CNN baselines.

These models are supplementary experiments, not the frozen V1 Transformer.
Only response words enter either encoder; labels are training targets only.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ragtruth_data.alignment import multi_hot, word_tokenize
from ragtruth_data.data import sha256_file
from ragtruth_data.schema import LABEL_TYPES

PAD, UNK = "<PAD>", "<UNK>"


def build_vocabulary(records, max_size=20000, min_frequency=2):
    """Fit exclusively on the supplied training records, with stable ties."""
    if type(max_size) is not int or max_size < 2 or type(min_frequency) is not int or min_frequency < 1:
        raise ValueError("invalid vocabulary limits")
    counts = Counter(token.casefold() for row in records for token in word_tokenize(row["response"])[0])
    words = sorted((word for word, n in counts.items() if n >= min_frequency),
                   key=lambda word: (-counts[word], word))[:max_size - 2]
    return {PAD: 0, UNK: 1, **{word: i + 2 for i, word in enumerate(words)}}


def make_examples(records, vocabulary, *, include_targets):
    examples, seen = [], set()
    for row in records:
        rid = row["id"]
        if not isinstance(rid, str) or not rid or rid in seen:
            raise ValueError("response IDs must be nonempty and unique")
        seen.add(rid)
        tokens, offsets = word_tokenize(row["response"])
        example = {"id": rid, "offsets": offsets,
                   "ids": [vocabulary.get(token.casefold(), 1) for token in tokens]}
        if include_targets:
            example["targets"] = multi_hot(offsets, row["labels"])
        examples.append(example)
    return examples


class ChunkDataset(torch.utils.data.Dataset):
    """Nonoverlapping chunks cover each word exactly once; no truncation."""
    def __init__(self, examples, chunk_size=256):
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.examples = examples
        self.chunk_size = chunk_size
        self.chunks = [(i, start) for i, ex in enumerate(examples)
                       for start in range(0, len(ex["ids"]), chunk_size)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        example_index, start = self.chunks[index]
        example = self.examples[example_index]
        stop = start + self.chunk_size
        item = {"example_index": example_index, "start": start, "ids": example["ids"][start:stop]}
        if "targets" in example:
            item["targets"] = example["targets"][start:stop]
        return item


def collate_chunks(items):
    if not items or any(not item["ids"] for item in items):
        raise ValueError("batch must contain nonempty chunks")
    lengths = torch.tensor([len(item["ids"]) for item in items], dtype=torch.long)
    ids = torch.zeros((len(items), int(lengths.max())), dtype=torch.long)
    mask = torch.arange(ids.shape[1])[None, :] < lengths[:, None]
    targets = torch.zeros((*ids.shape, len(LABEL_TYPES)), dtype=torch.float32)
    have_targets = ["targets" in item for item in items]
    if any(have_targets) and not all(have_targets):
        raise ValueError("mixed labelled/unlabelled batch")
    for i, item in enumerate(items):
        ids[i, :lengths[i]] = torch.tensor(item["ids"], dtype=torch.long)
        if all(have_targets):
            targets[i, :lengths[i]] = torch.tensor(item["targets"], dtype=torch.float32)
    return {"ids": ids, "lengths": lengths, "mask": mask,
            "targets": targets if all(have_targets) else None,
            "locations": [(item["example_index"], item["start"]) for item in items]}


class WordTagger(nn.Module):
    def __init__(self, architecture, vocabulary_size, embedding_size=128, hidden_size=128, dropout=0.1):
        super().__init__()
        if architecture not in {"rnn", "cnn"} or hidden_size < 2 or hidden_size % 2:
            raise ValueError("expected rnn/cnn and a positive even hidden_size")
        self.architecture = architecture
        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=0)
        if architecture == "rnn":
            self.encoder = nn.RNN(embedding_size, hidden_size // 2, batch_first=True,
                                  bidirectional=True, nonlinearity="tanh")
        else:
            self.convolutions = nn.ModuleList([
                nn.Conv1d(embedding_size, hidden_size // 2, kernel_size=k, padding=k // 2)
                for k in (3, 5)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, len(LABEL_TYPES))

    def forward(self, ids, lengths):
        embedded = self.embedding(ids)
        if self.architecture == "rnn":
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            encoded, _ = self.encoder(packed)
            encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=ids.shape[1])
        else:
            encoded = torch.cat([torch.relu(conv(embedded.transpose(1, 2)))
                                 for conv in self.convolutions], dim=1).transpose(1, 2)
        return self.classifier(self.dropout(encoded))


def masked_bce(logits, targets, mask):
    if logits.shape != targets.shape or mask.shape != logits.shape[:2] or not mask.any():
        raise ValueError("incompatible/empty BCE batch")
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return losses[mask].mean()


@torch.no_grad()
def predict_scores(model, examples, *, device, batch_size=32, chunk_size=256):
    model.eval()
    dataset = ChunkDataset(examples, chunk_size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_chunks)
    scores = [[[0.0] * len(LABEL_TYPES) for _ in ex["ids"]] for ex in examples]
    covered = [[False] * len(ex["ids"]) for ex in examples]
    for batch in loader:
        probabilities = model(batch["ids"].to(device), batch["lengths"]).sigmoid().cpu().tolist()
        for i, (example_index, start) in enumerate(batch["locations"]):
            length = int(batch["lengths"][i])
            if any(covered[example_index][start:start + length]):
                raise ValueError("a prediction word was decoded more than once")
            scores[example_index][start:start + length] = probabilities[i][:length]
            covered[example_index][start:start + length] = [True] * length
    if not all(all(row) for row in covered):
        raise ValueError("missing prediction words")
    return scores


def scores_to_predictions(examples, scores, threshold):
    if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
            or not math.isfinite(threshold) or not 0 < threshold < 1 or len(examples) != len(scores)):
        raise ValueError("invalid threshold/scores")
    result = []
    for example, word_scores in zip(examples, scores):
        if len(example["offsets"]) != len(word_scores):
            raise ValueError("word score count differs from original offsets")
        if any(len(row) != len(LABEL_TYPES) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in row)
               for row in word_scores):
            raise ValueError("word scores must be four finite probabilities")
        spans = []
        for column, kind in enumerate(LABEL_TYPES):
            active = None
            for (start, end), probabilities in zip(example["offsets"], word_scores):
                if probabilities[column] >= threshold:
                    if active is None:
                        active = {"start": start, "end": end, "label_type": kind}
                    else:
                        active["end"] = end
                elif active is not None:
                    spans.append(active)
                    active = None
            if active is not None:
                spans.append(active)
        spans.sort(key=lambda span: (span["start"], span["end"], LABEL_TYPES.index(span["label_type"])))
        result.append({"id": example["id"], "spans": spans})
    return result


def load_selected_model(directory, device="cpu"):
    directory = Path(directory)
    config = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    name = config["checkpoint"]
    if Path(name).name != name or config.get("label_types") != list(LABEL_TYPES):
        raise ValueError("invalid selected model metadata")
    checkpoint = directory / name
    if sha256_file(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    vocabulary = config["vocabulary"]
    if (vocabulary.get(PAD) != 0 or vocabulary.get(UNK) != 1
            or sorted(vocabulary.values()) != list(range(len(vocabulary)))):
        raise ValueError("invalid vocabulary")
    model = WordTagger(config["architecture"], len(vocabulary), config["embedding_size"],
                       config["hidden_size"], config["dropout"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, config
