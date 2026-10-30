"""2. kişi: label building, overlap policy and word/subword alignment (standard library only)."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_tokenizer import FakeFastTokenizer  # noqa: E402
from ragtruth_data.alignment import (  # noqa: E402
    aggregate_window_scores, bio_by_type, bio_tags, char_scores_to_spans, encode_with_windows,
    merge_intervals, multi_hot, select_windows, tags_to_char_spans, word_level_example, word_tokenize,
)
from ragtruth_data.data import build_labels  # noqa: E402
from ragtruth_data.features import build_features, pad_batch  # noqa: E402
from ragtruth_data.schema import LABEL_TYPES  # noqa: E402

# shortened from RAGTruth response 2: two overlapping Evident Conflict labels
RESP = "They are believed to have died before February 7, 2022. The exact dates remain unclear."
A = RESP.index("believed to have died before February 7")
B = RESP.index("February 7, 2022.")
LABELS = [
    {"start": A, "end": A + len("believed to have died before February 7"), "label_type": "Evident Conflict"},
    {"start": B, "end": B + len("February 7, 2022."), "label_type": "Evident Conflict"},
]


def lab(text, sub, typ, nth=0):
    i = -1
    for _ in range(nth + 1):
        i = text.index(sub, i + 1)
    return {"start": i, "end": i + len(sub), "label_type": typ}


def raw(text, sub, typ, nth=0):
    x = lab(text, sub, typ, nth)
    return dict(x, text=sub, meta="m", implicit_true=False, due_to_null=False)


class BuildLabelsTests(unittest.TestCase):
    def test_keeps_every_annotation_and_links_duplicates_and_overlaps(self):
        text = "The shop opens at 9 am on Sundays."
        rows = [raw(text, "9 am", "Evident Conflict"), raw(text, "opens at 9 am", "Evident Baseless Info"),
                raw(text, "9 am", "Evident Conflict")]
        out = build_labels(text, rows, "x")
        self.assertEqual(len(out), 3)
        self.assertEqual([l["orig_index"] for l in out], [1, 0, 2])
        self.assertEqual(out[2]["duplicate_of"], 1)
        self.assertEqual(out[0]["overlaps_with"], [1, 2])
        for l in out:
            self.assertEqual(text[l["start"]:l["end"]], l["text"])

    def test_rejects_offsets_that_do_not_match_text(self):
        bad = dict(raw("abc def", "def", "Evident Conflict"), text="xyz")
        with self.assertRaisesRegex(ValueError, "do not match"):
            build_labels("abc def", [bad], "x")

    def test_rejects_unknown_type(self):
        with self.assertRaisesRegex(ValueError, "unknown label type"):
            build_labels("abc def", [raw("abc def", "def", "Conflict")], "x")


class WordAlignmentTests(unittest.TestCase):
    def test_tokens_reproduce_original_text(self):
        text = "It wasn't 3.5 km — really!"
        toks, offs = word_tokenize(text)
        self.assertEqual(toks, ["It", "wasn't", "3", ".", "5", "km", "—", "really", "!"])
        self.assertTrue(all(text[s:e] == t for t, (s, e) in zip(toks, offs)))

    def test_merge_keeps_touching_intervals_apart(self):
        self.assertEqual(merge_intervals([(0, 5), (3, 8), (8, 10)]), [(0, 8), (8, 10)])

    def test_binary_tags_merge_overlap(self):
        toks, offs = word_tokenize(RESP)
        tags = bio_tags(offs, LABELS)
        self.assertEqual([t for t, g in zip(toks, tags) if g != "O"],
                         ["believed", "to", "have", "died", "before", "February", "7", ",", "2022", "."])
        self.assertEqual(tags.count("B"), 1)

    def test_cross_type_overlap_loses_nothing_per_type(self):
        text = "The shop opens at 9 am on Sundays and has free parking."
        labels = [lab(text, "opens at 9 am on Sundays", "Evident Baseless Info"),
                  lab(text, "9 am", "Evident Conflict"), lab(text, "free parking", "Subtle Baseless Info")]
        toks, offs = word_tokenize(text)
        per = bio_by_type(offs, labels)
        for l in labels:  # every label comes back exactly, in its own type
            back = tags_to_char_spans(offs, per[l["label_type"]], l["label_type"])
            self.assertIn({"start": l["start"], "end": l["end"], "label_type": l["label_type"]}, back)
        self.assertEqual(dict(zip(toks, multi_hot(offs, labels)))["9"], [1, 0, 1, 0])
        ex = word_level_example({"id": "1", "response": text, "labels": labels})
        self.assertEqual(set(ex["bio_by_type"]), set(LABEL_TYPES))

    def test_continuation_from_earlier_chunk_starts_with_I(self):
        toks, offs = word_tokenize(RESP)
        k = toks.index("have")
        self.assertEqual(bio_tags(offs[k:], LABELS, seq_start=offs[k][0])[0], "I")

    def test_partial_word_label_marks_whole_token(self):
        text = "unbelievable claims"
        toks, offs = word_tokenize(text)
        self.assertEqual(bio_tags(offs, [{"start": 2, "end": 6, "label_type": "Evident Conflict"}]), ["B", "O"])


class WindowTests(unittest.TestCase):
    SOURCE = " ".join(f"Passage {i}: The Local Butcher Shop phone number is (510) 845-{6300 + i}." for i in range(40))

    def check_windows(self, tok):
        feats = encode_with_windows(tok, RESP, self.SOURCE, LABELS, max_length=96, context_stride=16,
                                    min_context_tokens=24)
        self.assertGreater(len(feats), 3)
        covered = [False] * len(self.SOURCE)
        for f in feats:
            self.assertLessEqual(len(f["input_ids"]), 96)
            seen = set()
            for o in f["response_offsets"]:
                if o:
                    seen.update(range(*o))
            self.assertTrue(all(c in seen for c, ch in enumerate(RESP) if not ch.isspace()))
            s, e = f["source_char_span"]
            covered[s:e] = [True] * (e - s)
            for o, a, m, mask in zip(f["response_offsets"], f["labels_any"], f["labels_multi"], f["response_mask"]):
                if o is None:
                    self.assertEqual((a, m, mask), (-100, None, 0))
                else:
                    inside = any(l["start"] < o[1] and o[0] < l["end"] for l in LABELS)
                    self.assertEqual(a, int(inside))
                    self.assertEqual(m[0], int(inside))
        self.assertTrue(all(covered[c] for c, ch in enumerate(self.SOURCE) if not ch.isspace()))
        spans = sorted(f["source_char_span"] for f in feats)
        self.assertTrue(all(b[0] < a[1] for a, b in zip(spans, spans[1:])), "windows must overlap")

    def test_windows_cover_source_bert_layout(self):
        self.check_windows(FakeFastTokenizer("bert"))

    def test_windows_cover_source_roberta_layout(self):
        self.check_windows(FakeFastTokenizer("roberta"))

    def test_long_response_is_split_and_mapped_back(self):
        tok = FakeFastTokenizer()
        response = " ".join(["The restaurant opened in 1999 and serves Indian food."] * 12)
        labels = [lab(response, "opened in 1999", "Evident Conflict", 3),
                  lab(response, "serves Indian food. The restaurant opened", "Subtle Baseless Info", 6)]
        feats = encode_with_windows(tok, response, "Santa Barbara Indian kitchen, open since 2004. " * 30, labels,
                                    max_length=80, context_stride=8, min_context_tokens=20, response_overlap=10)
        self.assertGreater(len({f["response_chunk"] for f in feats}), 1)
        for ti, t in enumerate(LABEL_TYPES):
            scores = [[float(x != 0) if x != -100 else 0.0 for x in f["labels_bio_by_type"][ti]] for f in feats]
            pred = char_scores_to_spans(response, aggregate_window_scores(len(response), feats, scores), 0.5, t)
            gold = [l for l in labels if l["label_type"] == t]
            self.assertEqual(len(pred), len(gold))
            for l in gold:
                self.assertTrue(any(p["start"] <= l["start"] and l["end"] <= p["end"] for p in pred))

    def test_overlap_is_capped_so_windows_always_advance(self):
        # long response leaves little room for the source; a stride bigger than
        # that room must not make the windows crawl forward token by token
        tok = FakeFastTokenizer()
        response = "The restaurant opened in 1999 and serves Indian food. " * 6
        source = "Santa Barbara Indian kitchen, open since 2004. " * 60
        n_src = len(tok(source, add_special_tokens=False)["input_ids"])
        feats = encode_with_windows(tok, response, source, None, max_length=160, context_stride=128,
                                    min_context_tokens=40)
        budget = 160 - 3 - len(tok(response, add_special_tokens=False)["input_ids"])
        self.assertLessEqual(len(feats), n_src // (budget // 2) + 2)
        covered = [False] * len(source)
        for f in feats:
            s, e = f["source_char_span"]
            covered[s:e] = [True] * (e - s)
        self.assertTrue(all(covered[c] for c, ch in enumerate(source) if not ch.isspace()))

    def test_min_rule_clears_tokens_supported_in_any_window(self):
        feats = encode_with_windows(FakeFastTokenizer(), RESP, "Margot died before Anne. " * 30, LABELS,
                                    max_length=64, context_stride=8, min_context_tokens=16)
        self.assertGreater(len(feats), 1)
        scores = [[1.0 if f["window_index"] == 0 else 0.0] * len(f["input_ids"]) for f in feats]
        self.assertEqual(max(aggregate_window_scores(len(RESP), feats, scores, "min")), 0.0)
        self.assertEqual(max(aggregate_window_scores(len(RESP), feats, scores, "max")), 1.0)

    def test_best_window_uses_inputs_only(self):
        source = "Unrelated text about weather. " * 20 + "Anne Frank and Margot died in February 1945. " + "Filler. " * 20
        tok = FakeFastTokenizer()
        with_labels = select_windows(encode_with_windows(tok, RESP, source, LABELS, max_length=96,
                                                         context_stride=8, min_context_tokens=16), "best")
        without = select_windows(encode_with_windows(tok, RESP, source, None, max_length=96,
                                                     context_stride=8, min_context_tokens=16), "best")
        self.assertEqual(len(with_labels), 1)
        self.assertEqual(with_labels[0]["source_char_span"], without[0]["source_char_span"])
        s, e = with_labels[0]["source_char_span"]
        self.assertIn("February", source[s:e])

    def test_features_and_padding(self):
        rec = {"id": "x1", "response": RESP, "source_text": "Margot died before Anne. " * 30, "labels": LABELS}
        feats = build_features([rec], FakeFastTokenizer(), max_length=96, context_stride=8, min_context_tokens=16)
        self.assertGreater(len(feats), 1)
        batch = pad_batch(feats, 0)
        n = max(len(f["input_ids"]) for f in feats)
        self.assertTrue(all(len(row) == n for row in batch["input_ids"] + batch["labels"] + batch["response_mask"]))
        typed = pad_batch(feats, 0, label_key="labels_bio_by_type")
        self.assertEqual(len(typed["labels"]), 4)

    def test_char_scores_to_spans_trims_whitespace(self):
        text = "  alpha beta  gamma "
        scores = [0.0] * len(text)
        for a, b in [(2, 7), (8, 12), (14, 17)]:
            scores[a:b] = [1.0] * (b - a)
        scores[0] = 1.0
        self.assertEqual(char_scores_to_spans(text, scores, label_type="Evident Conflict"),
                         [{"start": 2, "end": 17, "label_type": "Evident Conflict"}])


try:  # optional: the same checks with real Hugging Face fast tokenizers, when installed
    import tokenizers  # noqa: F401
    import transformers  # noqa: F401
    HAVE_HF = True
except ImportError:
    HAVE_HF = False


@unittest.skipUnless(HAVE_HF, "tokenizers/transformers not installed")
class RealTokenizerTests(WindowTests):
    """Offline WordPiece and byte-level BPE tokenizers trained on a few sentences."""

    @staticmethod
    def _hf(kind):
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
        from transformers import PreTrainedTokenizerFast

        corpus = [RESP, WindowTests.SOURCE, "The restaurant opened in 1999 and serves Indian food.",
                  "Margot died before Anne. Santa Barbara Indian kitchen, open since 2004."] * 5
        if kind == "wordpiece":
            t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
            t.normalizer = normalizers.BertNormalizer(lowercase=True)
            t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            t.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=300, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
            t.post_processor = processors.TemplateProcessing(
                single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                special_tokens=[("[CLS]", t.token_to_id("[CLS]")), ("[SEP]", t.token_to_id("[SEP]"))])
            return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]",
                                           cls_token="[CLS]", sep_token="[SEP]")
        t = Tokenizer(models.BPE())
        t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        t.decoder = decoders.ByteLevel()
        t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        t.post_processor = processors.RobertaProcessing(("</s>", t.token_to_id("</s>")), ("<s>", t.token_to_id("<s>")), True, False)
        return PreTrainedTokenizerFast(tokenizer_object=t, bos_token="<s>", eos_token="</s>", sep_token="</s>",
                                       cls_token="<s>", pad_token="<pad>", unk_token="<unk>")

    def test_windows_cover_source_bert_layout(self):
        self.check_windows(self._hf("wordpiece"))

    def test_windows_cover_source_roberta_layout(self):
        self.check_windows(self._hf("bytelevel"))

    # the remaining inherited tests use the fake tokenizer and need not run twice
    test_long_response_is_split_and_mapped_back = None
    test_min_rule_clears_tokens_supported_in_any_window = None
    test_best_window_uses_inputs_only = None
    test_features_and_padding = None
    test_char_scores_to_spans_trims_whitespace = None


if __name__ == "__main__":
    unittest.main()
