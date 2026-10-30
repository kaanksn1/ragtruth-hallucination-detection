"""Offline person-4 checks. Tiny RANDOM ModernBERT; no pretrained weights or real training."""
import copy
import gc
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_tokenizer import FakeFastTokenizer
from ragtruth_data.data import sha256_file, write_jsonl
from ragtruth_data.schema import LABEL_TYPES
from scripts import transformer as cli
from scripts.baseline_crf import char_scores

HAVE_HF = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))
if HAVE_HF:
    import torch
    import transformer as tr
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast


def config():
    result = json.loads((ROOT / "configs/transformer.json").read_text(encoding="utf-8"))
    result.update(max_length=48, min_context_tokens=8, context_stride=4, response_overlap=4,
                  batch_size=2, dropout=0.0, gradient_accumulation_steps=2, epochs=1,
                  gradient_checkpointing=False, device="cpu")
    return result


def records():
    response = "The shop opened in 1999 and serves Indian food."
    labels = []
    for text, kind in zip(("opened in 1999", "opened", "1999", "Indian food"), LABEL_TYPES):
        start = response.index(text)
        labels.append({"start": start, "end": start + len(text), "label_type": kind})
    return [{"id": "overlap", "source_text": "The shop opened in 2004.", "response": response, "labels": labels},
            {"id": "negative", "source_text": "A shop.", "response": "A shop.", "labels": []}]


def tiny_model(vocab_size=256):
    enc = ModernBertConfig(vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=256,
                          local_attention=16, global_attn_every_n_layers=2,
                          pad_token_id=0, bos_token_id=1, eos_token_id=2,
                          reference_compile=False, attention_dropout=0.0,
                          embedding_dropout=0.0, mlp_dropout=0.0)
    enc._attn_implementation = "sdpa"
    return tr.HallucinationTransformer(ModernBertModel(enc), dropout=0.0)


def offline_tokenizer():
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}
    for row in records():
        for word in (row["response"] + " " + row["source_text"]).replace(".", " .").split():
            vocab.setdefault(word, len(vocab))
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", unk_token="[UNK]")


@unittest.skipUnless(HAVE_HF, "install requirements-transformer.txt for offline Transformer checks")
class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.cfg = config()
        tr.seed_everything(42)
        self.tokenizer = FakeFastTokenizer()
        self.features = [f for row in records() for f in tr.record_features(row, self.tokenizer, self.cfg)]
        self.batch = tr.collate_features(self.features, 0)

    def test_modernbert_forward_backward_and_optimizer_step(self):
        model = tiny_model()
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        before = model.classifier.weight.detach().clone()
        output = model(**self.batch)
        self.assertEqual(output["logits"].shape, (*self.batch["input_ids"].shape, 4))
        self.assertTrue(torch.isfinite(output["loss"]))
        output["loss"].backward()
        for component in (model.encoder, model.classifier):
            gradients = [p.grad for p in component.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(torch.count_nonzero(g) > 0 for g in gradients))
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_mask_excludes_source_specials_padding_even_with_invalid_targets(self):
        mask = self.batch["response_mask"].bool() & self.batch["attention_mask"].bool()
        self.assertTrue((self.batch["attention_mask"] == 0).any())
        logits = torch.randn_like(self.batch["labels"], requires_grad=True)
        labels = self.batch["labels"].clone()
        loss = tr.masked_bce_loss(logits, labels, self.batch["response_mask"], self.batch["attention_mask"])
        changed = logits.detach().clone()
        changed[~mask] = float("nan")
        labels[~mask] = float("nan")
        other = tr.masked_bce_loss(changed, labels, self.batch["response_mask"], self.batch["attention_mask"])
        torch.testing.assert_close(loss, other)
        loss.backward()
        self.assertTrue((logits.grad[~mask] == 0).all())
        self.assertTrue((logits.grad[mask] != 0).all())
        self.assertTrue(any(row.tolist() == [1, 0, 1, 0] for row in self.batch["labels"][mask]))
        # Even an erroneous response-mask bit on padding is suppressed by attention_mask.
        torch.testing.assert_close(loss, tr.masked_bce_loss(
            logits, self.batch["labels"], self.batch["response_mask"] | (1 - self.batch["attention_mask"]),
            self.batch["attention_mask"]))

    def test_all_zero_labels_remain_valid_negatives(self):
        logits = torch.zeros((1, 2, 4), requires_grad=True)
        loss = tr.masked_bce_loss(logits, torch.zeros_like(logits), torch.ones((1, 2)), torch.ones((1, 2)))
        self.assertAlmostEqual(loss.item(), 0.69314718, places=6)
        loss.backward()
        self.assertTrue((logits.grad > 0).all())

    def test_source_tokens_affect_answer_logits_without_becoming_loss_targets(self):
        model = tiny_model().eval()
        altered = {key: value.clone() for key, value in self.batch.items()}
        source_positions = torch.zeros_like(self.batch["response_mask"], dtype=torch.bool)
        for row, feature in enumerate(self.features):
            # This synthetic tokenizer uses token_type_id=1 exclusively for sequence B (source).
            for index, kind in enumerate(feature["token_type_ids"]):
                if kind == 1 and feature["input_ids"][index] != 2:
                    source_positions[row, index] = True
        self.assertTrue(source_positions.any())
        self.assertFalse(self.batch["response_mask"][source_positions].any())
        altered["input_ids"][source_positions] = 200
        answer = self.batch["response_mask"].bool()
        self.assertFalse(torch.equal(model(**self.batch)["logits"][answer], model(**altered)["logits"][answer]))

    def test_empty_batch_and_empty_response_backward(self):
        for batch in (tr.collate_features([], 0), dict(self.batch, response_mask=torch.zeros_like(self.batch["response_mask"]))):
            model = tiny_model()
            output = model(**batch)
            self.assertEqual(output["loss"].item(), 0)
            output["loss"].backward()
            self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in model.parameters() if p.grad is not None))

    def test_empty_answer_does_not_update_adamw_or_count_a_step(self):
        model = tiny_model()
        before = copy.deepcopy(model.state_dict())
        row = {"id": "empty", "source_text": "Valid source.", "response": "", "labels": []}
        loader = tr.make_loader([row], self.tokenizer, self.cfg, training=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.5)
        result = tr.run_epoch(model, loader, torch.device("cpu"), self.cfg, optimizer)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertIsNone(result["loss"])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_short_accumulation_group_is_token_weighted(self):
        first, second = tiny_model(), tiny_model()
        second.load_state_dict(first.state_dict())
        separate = [tr.collate_features([f], 0) for f in self.features]
        together = [self.batch]
        cfg = dict(self.cfg, gradient_accumulation_steps=10, max_grad_norm=1000)
        one = tr.run_epoch(first, separate, torch.device("cpu"), cfg, torch.optim.SGD(first.parameters(), lr=0.01))
        two = tr.run_epoch(second, together, torch.device("cpu"), cfg, torch.optim.SGD(second.parameters(), lr=0.01))
        self.assertEqual(one["optimizer_steps"], 1)
        self.assertEqual(two["optimizer_steps"], 1)
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_decoded_overlap_is_accepted_by_existing_character_scorer(self):
        row = records()[0]
        features = tr.record_features(row, self.tokenizer, self.cfg)
        # Oracle target probabilities validate conversion only, not trained-model accuracy.
        scores = [[target or [0.0] * 4 for target in f["labels_multi"]] for f in features]
        result = json.loads(json.dumps(tr.decode_record(row, features, scores, self.cfg)))
        self.assertEqual(set(result), {"id", "spans"})
        self.assertEqual(result["spans"], sorted(row["labels"], key=lambda x: (x["start"], x["end"], LABEL_TYPES.index(x["label_type"]))))
        metrics = char_scores({row["id"]: row["labels"]}, {result["id"]: result["spans"]})
        self.assertEqual(metrics["macro_f1_types"], 1)

    def test_long_source_and_answer_coverage_and_min_aggregation(self):
        row = records()[0]
        row["source_text"] *= 30
        row["response"] *= 6
        features = tr.record_features(row, self.tokenizer, self.cfg)
        self.assertGreater(len({f["response_chunk"] for f in features}), 1)
        response_chars, source_chars = set(), set()
        for f in features:
            self.assertLessEqual(len(f["input_ids"]), self.cfg["max_length"])
            source_chars.update(range(*f["source_char_span"]))
            for offset in f["response_offsets"]:
                if offset:
                    response_chars.update(range(*offset))
        for text, covered in ((row["response"], response_chars), (row["source_text"], source_chars)):
            self.assertTrue(all(i in covered for i, c in enumerate(text) if not c.isspace()))
        scores = [[[0.0 if f["window_index"] == 0 else 1.0] * 4 for _ in f["input_ids"]] for f in features]
        self.assertEqual(tr.decode_record(row, features, scores, self.cfg)["spans"], [])
        best = tr.record_features(row, self.tokenizer, self.cfg, training=True)
        unlabelled = tr.record_features(dict(row, labels=[]), self.tokenizer, self.cfg, training=True)
        self.assertEqual([f["input_ids"] for f in best], [f["input_ids"] for f in unlabelled])
        best_chars = {i for f in best for offset in f["response_offsets"] if offset for i in range(*offset)}
        self.assertEqual(response_chars, best_chars)

    def test_prediction_never_reads_labels_and_handles_empty_response(self):
        class Poison(dict):
            def __getitem__(self, key):
                if key in {"labels", "meta", "quality", "has_hallucination"}:
                    raise AssertionError("annotation read during prediction")
                return super().__getitem__(key)
        rows = [Poison(r) for r in records()] + [Poison(id="empty", response="", source_text="A shop.")]
        model = tiny_model()
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.fill_(8)
        predicted = list(tr.predict_records(model, rows, self.tokenizer, self.cfg, torch.device("cpu")))
        self.assertEqual(predicted[-1], {"id": "empty", "spans": []})
        self.assertEqual({s["label_type"] for s in predicted[0]["spans"]}, set(LABEL_TYPES))

    def test_checkpoint_roundtrip_and_label_order_guard(self):
        model, tokenizer = tiny_model().eval(), offline_tokenizer()
        batch = tr.collate_features(tr.record_features(records()[0], tokenizer, self.cfg), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            tr.save_checkpoint(path, model, tokenizer, self.cfg, {"synthetic": True})
            reloaded, tok, saved = tr.load_checkpoint(path)
            torch.testing.assert_close(model(**batch)["logits"], reloaded(**batch)["logits"], rtol=0, atol=0)
            self.assertEqual(tok(records()[0]["response"]), tokenizer(records()[0]["response"]))
            self.assertTrue(saved["metadata"]["synthetic"])
            with self.assertRaises(FileExistsError):
                tr.save_checkpoint(path, model, tokenizer, self.cfg, {})
            saved["label_types"].reverse()
            (path / "checkpoint.json").write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "label order"):
                tr.load_checkpoint(path)
            # HF safetensors may memory-map weights; release before Windows deletes the file.
            del reloaded
            gc.collect()

    def test_invalid_config_and_score_shapes_fail_early(self):
        for key, value in (("batch_size", 0), ("context_stride", -1), ("learning_rate", float("nan")),
                           ("threshold", 0), ("aggregation", "first"), ("precision", "fp16")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                tr.validate_config(dict(self.cfg, **{key: value}))
        with self.assertRaisesRegex(ValueError, "answer budget"):
            tr.validate_tokenizer(self.tokenizer, dict(self.cfg, max_length=8))
        with self.assertRaises(ValueError):
            tr.decode_record(records()[0], self.features, [], self.cfg)

    def test_changed_checkpoint_weights_are_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            tr.save_checkpoint(path, tiny_model(), offline_tokenizer(), self.cfg, {})
            changed = {key: value.clone() for key, value in tr.load_file(str(path / "head.safetensors")).items()}
            changed["bias"][0] += 1
            tr.save_file(changed, str(path / "head.safetensors"))
            with patch.object(tr.AutoModel, "from_pretrained", side_effect=AssertionError("model loaded before integrity check")):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    tr.load_checkpoint(path)


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.manifest = self.path / "manifest.csv"
        self.manifest.write_text("source_id,group_id,split\ntrain-src,train-group,train\nval-src,val-group,validation\n", encoding="utf-8")
        self.rows = {}
        for split, source in (("train", "train-src"), ("validation", "val-src")):
            self.rows[split] = [dict(row, id=split + row["id"], source_id=source, split=split, quality="good") for row in records()]
            write_jsonl(self.path / f"{split}.jsonl", self.rows[split])
        self.cfg = dict(config(), processed_dir=str(self.path), split_manifest=str(self.manifest),
                        split_summary=str(self.path / "summary.json"))
        (self.path / "summary.json").write_text(json.dumps({"processed_sha256": {
            split + ".jsonl": sha256_file(self.path / f"{split}.jsonl") for split in self.rows}}), encoding="utf-8")

    def test_manifest_membership_unique_ids_and_no_annotation_in_prediction_records(self):
        rows = cli.read_records(self.path, "train", self.manifest, with_labels=False)
        self.assertEqual(len(rows), 2)
        self.assertNotIn("labels", rows[0])
        write_jsonl(self.path / "train.jsonl", self.rows["train"] * 2)
        with self.assertRaisesRegex(ValueError, "unique"):
            cli.read_records(self.path, "train", self.manifest)
        write_jsonl(self.path / "train.jsonl", [dict(self.rows["train"][0], source_id="val-src")])
        with self.assertRaisesRegex(ValueError, "manifest"):
            cli.read_records(self.path, "train", self.manifest)

    def test_modified_data_is_rejected(self):
        cli.verify_data(self.cfg, ("train", "validation"))
        with (self.path / "train.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "differs from split summary"):
            cli.verify_data(self.cfg, ("train",))

    def test_cross_split_manifest_group_is_rejected_even_in_limited_prediction(self):
        self.manifest.write_text("source_id,group_id,split\ntrain-src,same-group,train\nval-src,same-group,validation\n",
                                 encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "group.*multiple splits"):
            cli.read_records(self.path, "train", self.manifest, with_labels=False, limit=1)

    def test_official_test_requires_frozen_run_before_data_or_model_access(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "frozen"):
                from types import SimpleNamespace
                cli.cmd_predict(SimpleNamespace(split="test", limit=None, invocation=["predict"]))

    @unittest.skipUnless(HAVE_HF, "install requirements-transformer.txt")
    def test_synthetic_train_validation_checkpoint_predict_cli_pipeline(self):
        # Only two synthetic records; AutoModel download replaced with tiny random ModernBERT.
        cfg_path = self.path / "config.json"
        cfg_path.write_text(json.dumps(dict(self.cfg, epochs=3)), encoding="utf-8")
        output = self.path / "run"
        with patch.object(tr.HallucinationTransformer, "from_pretrained", side_effect=lambda _: tiny_model()), \
             patch.object(tr.AutoTokenizer, "from_pretrained", return_value=offline_tokenizer()), redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["train", "--config", str(cfg_path), "--epochs", "1",
                                       "--output", str(output)]), 0)
        summary = json.loads((output / "transformer_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["config"]["epochs"], 1)
        self.assertEqual(len(summary["epochs"]), 1)
        self.assertEqual(summary["epochs"][0]["train"]["optimizer_steps"], 1)
        predicted = self.path / "again.jsonl"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["predict", "--checkpoint", str(output / summary["best_checkpoint"]),
                                       "--split", "validation", "--processed-dir", str(self.path),
                                       "--manifest", str(self.manifest), "--split-summary", self.cfg["split_summary"],
                                       "--device", "cpu", "--output", str(predicted)]), 0)
        self.assertEqual(predicted.read_bytes(), (output / "validation_predictions.jsonl").read_bytes())


@unittest.skipUnless(HAVE_HF and (ROOT / ".cache/modernbert-tokenizer/tokenizer.json").exists(),
                     "optional pinned ModernBERT tokenizer is not cached (no automatic downloads)")
class PinnedTokenizerTests(unittest.TestCase):
    def test_real_tokenizer_sample_and_long_unicode_coverage(self):
        from ragtruth_data.data import read_jsonl
        tokenizer = tr.AutoTokenizer.from_pretrained(ROOT / ".cache/modernbert-tokenizer", local_files_only=True)
        cfg = config()
        examples = list(read_jsonl(ROOT / "data/samples/sample_records.jsonl"))
        examples.append({"id": "unicode", "source_text": "İstanbul café 😊 — kanıt. " * 35,
                         "response": "İstanbul café 😊 — cevap. " * 15, "labels": []})
        for row in examples:
            # The committed examples use the production length; synthetic Unicode forces splitting.
            actual = dict(cfg, max_length=2048) if row["id"] != "unicode" else cfg
            features = tr.record_features(row, tokenizer, actual)
            for field in ("source_text", "response"):
                covered = set()
                for f in features:
                    self.assertLessEqual(len(f["input_ids"]), actual["max_length"])
                    offsets = [f["source_char_span"]] if field == "source_text" else f["response_offsets"]
                    for offset in offsets:
                        if offset:
                            covered.update(range(*offset))
                self.assertTrue(all(i in covered for i, ch in enumerate(row[field]) if not ch.isspace()), row["id"])
            scores = [[target or [0.] * 4 for target in f["labels_multi"]] for f in features]
            predicted = tr.decode_record(row, features, scores, actual)["spans"]
            for label in row["labels"]:
                covered = {i for span in predicted if span["label_type"] == label["label_type"]
                           for i in range(span["start"], span["end"])}
                self.assertTrue(all(i in covered for i in range(label["start"], label["end"])
                                    if not row["response"][i].isspace()))

    def test_real_tokenizer_with_random_tiny_modernbert_backward(self):
        tokenizer = tr.AutoTokenizer.from_pretrained(ROOT / ".cache/modernbert-tokenizer", local_files_only=True)
        features = [f for row in records() for f in tr.record_features(row, tokenizer, config())]
        model = tiny_model(len(tokenizer))
        loss = model(**tr.collate_features(features, tokenizer.pad_token_id))["loss"]
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.count_nonzero(model.classifier.weight.grad) > 0)


if __name__ == "__main__":
    unittest.main()
