"""Tiny stand-in for a Hugging Face *fast* tokenizer (standard library only).

It mimics the parts the data code uses: offset mapping, pair encoding with
special tokens, sequence_ids(), truncation="only_second". Long words are cut
into 4-character pieces so tokens do not always match words, like subwords.
Two layouts: "bert" ([CLS] A [SEP] B [SEP]) and "roberta" (<s> A </s></s> B </s>).
"""

import re

PIECE = re.compile(r"\w{1,4}|[^\w\s]")


class Encoding(dict):
    def __init__(self, data, seq_ids):
        super().__init__(data)
        self._seq_ids = seq_ids

    def sequence_ids(self, batch_index=0):
        return list(self._seq_ids)


class FakeFastTokenizer:
    is_fast = True
    pad_token_id = 0

    def __init__(self, style="bert"):
        self.style = style
        self.vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}

    def _tok(self, text):
        out = []
        for m in PIECE.finditer(text):
            out.append((self.vocab.setdefault(m.group(), len(self.vocab)), (m.start(), m.end())))
        return out

    def num_special_tokens_to_add(self, pair=False):
        if self.style == "bert":
            return 3 if pair else 2
        return 4 if pair else 2

    def __call__(self, text, text_pair=None, add_special_tokens=True, return_offsets_mapping=False,
                 truncation=False, max_length=None):
        a = self._tok(text)
        b = self._tok(text_pair) if text_pair is not None else None
        if truncation == "only_second" and max_length is not None and b is not None:
            room = max_length - len(a) - self.num_special_tokens_to_add(pair=True)
            if room < 0:
                raise ValueError("first sequence too long for only_second truncation")
            b = b[:room]
        ids, offs, seq = [], [], []

        def add(tokens, sid):
            for i, o in tokens:
                ids.append(i)
                offs.append(o)
                seq.append(sid)

        def special(n=1):
            for _ in range(n):
                ids.append(2)
                offs.append((0, 0))
                seq.append(None)

        if add_special_tokens:
            ids.append(1); offs.append((0, 0)); seq.append(None)
            add(a, 0)
            special(1 if self.style == "bert" or b is None else 2)
            if b is not None:
                add(b, 1)
                special()
        else:
            add(a, 0)
            if b is not None:
                add(b, 1)
        data = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if self.style == "bert":
            data["token_type_ids"] = [1 if s == 1 else 0 for s in seq]
        if return_offsets_mapping:
            data["offset_mapping"] = offs
        return Encoding(data, seq)
