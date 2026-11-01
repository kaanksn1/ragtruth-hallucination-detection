"""Synthetic structured-model checks; no downloaded assets or real data."""
from __future__ import annotations

import copy
import csv
import importlib.util
import itertools
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
from ragtruth_data.alignment import word_tokenize
from ragtruth_data.schema import LABEL_TYPES
from scripts import structured_transformer as cli

HAVE_HF = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"))
if HAVE_HF:
    import torch
    import structured_transformer as tr
    import test_transformer as fixtures


@unittest.skipUnless(HAVE_HF, "install requirements-transformer.txt for offline structured checks")
class StructuredTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(4)
        self.cfg = fixtures.config()
        self.tokenizer = FakeFastTokenizer()

    def model(self):
        return tr.StructuredTransformer(fixtures.tiny_model().encoder, 0.0)

    def test_partition_and_viterbi_match_exhaustive_paths_all_types(self):
        crf = tr.BatchedBIOCRF()
        for parameter in crf.parameters():
            torch.nn.init.normal_(parameter, std=0.3)
        emissions = torch.randn(2, 4, 4, 3, requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        actual = crf.log_partition(emissions, mask)
        decoded = crf.decode(emissions, mask)
        for row in range(2):
            length = int(mask[row].sum())
            for kind in range(4):
                candidates = []
                for tags in itertools.product(range(3), repeat=length):
                    if tags[0] == 2 or any(a == 0 and b == 2 for a, b in zip(tags, tags[1:])):
                        continue
                    score = crf.start_transitions[kind, tags[0]] + crf.end_transitions[kind, tags[-1]]
                    score = score + sum(emissions[row, index, kind, tag] for index, tag in enumerate(tags))
                    score = score + sum(crf.transitions[kind, a, b] for a, b in zip(tags, tags[1:]))
                    candidates.append((score, tags))
                expected = torch.logsumexp(torch.stack([score for score, _ in candidates]), 0)
                torch.testing.assert_close(actual[row, kind], expected)
                best = max(candidates, key=lambda item: float(item[0].detach()))[1]
                self.assertEqual(decoded[row, :length, kind].tolist(), list(best))
        self.assertTrue((decoded[1, 2:] == -1).all())

    def test_nll_gradients_legal_and_illegal_paths_padding(self):
        crf = tr.BatchedBIOCRF()
        emissions = torch.randn(2, 3, 4, 3, requires_grad=True)
        tags = torch.tensor([[[1, 0, 1, 0], [2, 1, 0, 1], [0, 2, 1, 2]],
                             [[0, 1, 0, 1], [-100] * 4, [-100] * 4]])
        mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
        loss = crf.neg_log_likelihood(emissions, tags, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(emissions.grad).all())
        self.assertTrue((emissions.grad[~mask] == 0).all())
        for parameter in crf.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue((crf.start_transitions.grad[:, 2] == 0).all())
        self.assertTrue((crf.transitions.grad[:, 0, 2] == 0).all())
        bad = tags.clone()
        bad[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "start with I"):
            crf.neg_log_likelihood(emissions, bad, mask)
        bad[0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "O-to-I"):
            crf.neg_log_likelihood(emissions, bad, mask)
        with self.assertRaisesRegex(ValueError, "prefix"):
            crf.decode(emissions, torch.tensor([[1, 0, 1], [1, 0, 0]]))

    def test_empty_chains_and_o_bias_direction(self):
        crf = tr.BatchedBIOCRF()
        emissions = torch.empty((2, 0, 4, 3), requires_grad=True)
        tags = torch.empty((2, 0, 4), dtype=torch.long)
        mask = torch.empty((2, 0), dtype=torch.bool)
        loss = crf.neg_log_likelihood(emissions, tags, mask)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertEqual(crf.decode(emissions, mask).shape, (2, 0, 4))
        emissions = torch.zeros(1, 3, 4, 3)
        emissions[..., 1] = 0.5
        mask = torch.ones(1, 3, dtype=torch.bool)
        self.assertTrue((crf.decode(emissions, mask, 1) == 0).all())
        self.assertTrue((crf.decode(emissions, mask, -1) != 0).all())

    def test_original_unicode_words_overlap_and_chunk_start_normalization(self):
        text = "İstanbul'da café 🧪 çokuzunbirkelime e\u0301lan."
        _, original = word_tokenize(text)
        record = {"id": "unicode", "source_text": "kaynak " * 40, "response": text,
                  "labels": [{"start": 1, "end": len(text) - 1, "label_type": kind} for kind in LABEL_TYPES]}
        config = dict(self.cfg, max_length=16, min_context_tokens=6, response_overlap=2)
        features = tr.record_features(record, self.tokenizer, config, training=True)
        self.assertGreater(len(features), 1)
        observed = set()
        for feature in features:
            for index, offset in zip(feature["word_indices"], feature["word_offsets"]):
                self.assertEqual(offset, original[index])
                observed.add(index)
            self.assertTrue(all(value in (0, 1) for value in feature["labels_bio"][0]))
            for bag in feature["word_token_indices"]:
                self.assertTrue(all(feature["response_mask"][index] == 1 for index in bag))
        self.assertEqual(observed, set(range(len(original))))
        self.assertTrue(any(sum(tag > 0 for tag in row) == 4 for feature in features for row in feature["labels_bio"]))

    def test_differentiable_pooling_excludes_source_special_and_padding(self):
        features = [feature for row in fixtures.records() for feature in tr.record_features(row, self.tokenizer, self.cfg)]
        batch = tr.collate_features(features, 0)
        model = self.model()
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        before = model.classifier.weight.detach().clone()
        output = model(**batch)
        self.assertEqual(output["emissions"].shape, (*batch["word_mask"].shape, 4, 3))
        output["loss"].backward()
        for component in (model.encoder, model.classifier, model.crf):
            gradients = [parameter.grad for parameter in component.parameters() if parameter.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertTrue(any((gradient != 0).any() for gradient in gradients))
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        self.assertFalse(torch.equal(before, model.classifier.weight))
        malformed = copy.deepcopy(features)
        malformed[0]["word_token_indices"][0].append(0)
        with self.assertRaisesRegex(ValueError, "source/special/padding"):
            tr.collate_features(malformed, 0)
        # Exact mean-pool oracle on synthetic token values, including ragged padding.
        hidden = torch.arange(batch["input_ids"].numel(), dtype=torch.float32).reshape(-1, 1)
        pooled = torch.nn.functional.embedding_bag(batch["pool_indices"], hidden, batch["pool_offsets"],
                                                  mode="mean", include_last_offset=True)
        words = batch["word_mask"].shape[1]
        for row, feature in enumerate(features):
            for index, bag in enumerate(feature["word_token_indices"]):
                expected = sum(row * batch["input_ids"].shape[1] + token for token in bag) / len(bag)
                self.assertAlmostEqual(pooled[row * words + index].item(), expected, places=5)

    def test_whole_probability_vector_selection_then_chunk_average(self):
        record = {"id": "x", "response": "a b"}
        features = [{"response_chunk": 0, "word_indices": [0, 1]},
                    {"response_chunk": 0, "word_indices": [0, 1]},
                    {"response_chunk": 1, "word_indices": [1]}]
        a = torch.tensor([[[0.2, 0.7, 0.1]] * 4, [[0.7, 0.1, 0.2]] * 4])
        b = torch.tensor([[[0.6, 0.1, 0.3]] * 4, [[0.1, 0.8, 0.1]] * 4])
        c = torch.tensor([[[0.3, 0.4, 0.3]] * 4])
        result = tr.aggregate_word_scores(record, features, [a, b, c])
        torch.testing.assert_close(torch.tensor(result["probabilities"])[0], b[0])
        torch.testing.assert_close(torch.tensor(result["probabilities"])[1], (a[1] + c[0]) / 2)
        with self.assertRaisesRegex(ValueError, "omitted"):
            tr.aggregate_word_scores(record, features[2:], [c])

    def test_decode_preserves_cross_class_overlaps_and_batch_equivalence(self):
        probabilities = [[[0.01, 0.98, 0.01]] * 4, [[0.01, 0.01, 0.98]] * 4]
        scores = {"id": "overlap", "word_offsets": [[0, 2], [3, 7]], "probabilities": probabilities}
        crf = tr.BatchedBIOCRF()
        prediction = tr.decode_word_scores(scores, crf)
        self.assertEqual(prediction["spans"], [{"start": 0, "end": 7, "label_type": kind} for kind in LABEL_TYPES])
        empty = {"id": "empty", "word_offsets": [], "probabilities": []}
        self.assertEqual(cli.decode_batch([scores, empty], crf, 0), [prediction, {"id": "empty", "spans": []}])
        with self.assertRaisesRegex(ValueError, "normalized"):
            tr.decode_word_scores(dict(scores, probabilities=[[[float("nan"), 0, 1]] * 4] * 2), crf)

    def test_checkpoint_roundtrip_and_gold_free_prediction(self):
        model = self.model().eval()
        tokenizer = fixtures.offline_tokenizer()
        records = fixtures.records()
        no_gold = [{key: value for key, value in row.items() if key != "labels"} for row in records]
        original = list(tr.predict_records(model, records, tokenizer, self.cfg, torch.device("cpu")))
        self.assertEqual(original, list(tr.predict_records(model, no_gold, tokenizer, self.cfg, torch.device("cpu"))))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model"
            tr.save_checkpoint(checkpoint, model, tokenizer, self.cfg, {"epoch": 1, "data_sha256": {"synthetic": "abc"}})
            loaded, tokenizer2, saved = tr.load_checkpoint(checkpoint)
            self.assertFalse(loaded.encoder.config.reference_compile)
            self.assertEqual(saved["architecture"], tr.ARCHITECTURE)
            self.assertEqual(original, list(tr.predict_records(loaded, no_gold, tokenizer2, self.cfg, torch.device("cpu"))))
            torch.testing.assert_close(tr.load_crf(checkpoint).transitions, model.crf.transitions)
            (checkpoint / "head.safetensors").write_bytes(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                tr.load_checkpoint(checkpoint)

    def test_empty_answer_training_does_not_update_optimizer(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.2)
        records = [{"id": "empty", "source_text": "source", "response": " ", "labels": []}]
        before = copy.deepcopy(model.state_dict())
        stats = tr.run_epoch(model, tr.make_loader(records, self.tokenizer, self.cfg), torch.device("cpu"), self.cfg, optimizer)
        self.assertIsNone(stats["loss"])
        self.assertEqual(stats["optimizer_steps"], 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_masked_nan_padding_and_empty_row_have_finite_zero_gradients(self):
        crf = tr.BatchedBIOCRF()
        mask = torch.tensor([[1, 1, 0], [0, 0, 0]], dtype=torch.bool)
        emissions = torch.randn(2, 3, 4, 3).masked_fill(~mask[..., None, None], float("nan")).requires_grad_()
        tags = torch.zeros(2, 3, 4, dtype=torch.long).masked_fill(~mask[..., None], -100)
        loss = crf.neg_log_likelihood(emissions, tags, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(emissions.grad).all())
        self.assertTrue((emissions.grad[~mask] == 0).all())
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in crf.parameters()))

    def test_synthetic_cli_train_all_epochs_calibrate_then_gold_free_predict(self):
        from argparse import Namespace
        from ragtruth_data.data import write_jsonl
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            processed.mkdir()
            original = fixtures.records()
            members = []
            for split in ("train", "validation"):
                rows = []
                for index, row in enumerate(original):
                    source = f"{split}-source-{index}"
                    rows.append(dict(row, id=f"{split}-{index}", source_id=source, split=split, quality="good"))
                    members.append({"source_id": source, "group_id": source, "split": split})
                write_jsonl(processed / f"{split}.jsonl", rows)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_id", "group_id", "split"])
                writer.writeheader()
                writer.writerows(members)
            summary = root / "split_summary.json"
            cli.write_json(summary, {"processed_sha256": {f"{split}.jsonl": cli.sha256_file(processed / f"{split}.jsonl")
                            for split in ("train", "validation")}})
            cfg = dict(self.cfg, processed_dir=str(processed), split_manifest=str(manifest), split_summary=str(summary), epochs=2)
            cfg_path = root / "config.json"
            cli.write_json(cfg_path, cfg)
            out = root / "outputs" / "run"
            args = Namespace(config=cfg_path, output=out, device=None, seed=None, epochs=None,
                             limit_train=None, limit_validation=None)
            with patch.object(tr.AutoTokenizer, "from_pretrained", return_value=fixtures.offline_tokenizer()):
                with patch.object(tr.StructuredTransformer, "from_pretrained", side_effect=lambda config: self.model()) as fresh:
                    result = cli.cmd_train(args)
            fresh.assert_called_once_with(cfg)
            self.assertEqual(result["saved_checkpoints"], ["checkpoints/epoch-1", "checkpoints/epoch-2"])
            self.assertEqual(result["status"], "complete")
            cal_out = root / "outputs" / "calibration"
            calibrate = Namespace(config=None, checkpoints=[out / name for name in result["saved_checkpoints"]], output=cal_out, device="cpu")
            with patch.object(cli, "ROOT", root), patch.object(cli, "CODE_PATHS", ()):
                artifact = cli.cmd_calibrate(calibrate)
            self.assertEqual(artifact["status"], "selected_on_validation")
            checkpoint = root / artifact["checkpoint"]
            cache = root / "predict-cache.jsonl"
            predictions = root / "prediction.jsonl"
            args = Namespace(checkpoint=checkpoint, split="validation", limit=None, processed_dir=processed,
                             manifest=manifest, split_summary=summary, output=predictions, cache_output=cache,
                             device="cpu", calibration=cal_out / "calibration.json")
            self.assertEqual(cli.cmd_predict(args), 2)
            self.assertEqual(predictions.read_bytes(), (cal_out / "validation_predictions.jsonl").read_bytes())
            meta = json.loads(cache.with_suffix(".jsonl.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["cache_sha256"], cli.sha256_file(cache))
            self.assertEqual(meta["identity"]["data_sha256"], result["data_sha256"])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                cli.cmd_predict(args)


class StructuredIntegrityTests(unittest.TestCase):
    def test_missing_holdout_summary_hash_requires_exact_gate_hash_null_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            processed.mkdir()
            (processed / "test.jsonl").write_text("synthetic holdout", encoding="utf-8")
            (processed / "validation.jsonl").write_text("synthetic validation", encoding="utf-8")
            summary, manifest = root / "summary.json", root / "manifest.csv"
            manifest.write_text("synthetic manifest", encoding="utf-8")
            config = {"processed_dir": str(processed), "split_summary": str(summary), "split_manifest": str(manifest)}
            good = {"validation.jsonl": cli.sha256_file(processed / "validation.jsonl")}
            cli.write_json(summary, {"processed_sha256": good})
            original_summary = summary.read_bytes()
            actual = cli.sha256_file(processed / "test.jsonl")
            with patch.object(cli, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "gate first"):
                    cli.verify_prediction_data(config, "test")
                with self.assertRaisesRegex(ValueError, "must declare"):
                    cli.verify_prediction_data(config, "test", {})
                with self.assertRaisesRegex(ValueError, "frozen job hash"):
                    cli.verify_prediction_data(config, "test", {"processed/test.jsonl": "wrong"})
                identity = cli.verify_prediction_data(config, "test", {"processed/test.jsonl": actual})
                self.assertEqual(identity["test.jsonl"], actual)
                self.assertEqual(summary.read_bytes(), original_summary)
                self.assertEqual(cli.verify_prediction_data(config, "validation")["validation.jsonl"], good["validation.jsonl"])
                for invalid in (None, "wrong"):
                    cli.write_json(summary, {"processed_sha256": dict(good, **{"test.jsonl": invalid})})
                    with self.assertRaisesRegex(ValueError, "differs from split summary"):
                        cli.verify_prediction_data(config, "test", {"processed/test.jsonl": actual})
                cli.write_json(summary, {"processed_sha256": {}})
                with self.assertRaisesRegex(ValueError, "differs from split summary"):
                    cli.verify_prediction_data(config, "validation", {"processed/validation.jsonl": good["validation.jsonl"]})

    def test_holdout_gate_runs_before_model_data_reads(self):
        args = type("Args", (), {"split": "test", "limit": None, "invocation": ["synthetic"]})()
        with patch.object(cli, "require_frozen_test_job", side_effect=ValueError("gate denied")) as gate:
            with patch.object(cli, "read_records", side_effect=AssertionError("must not read")):
                with self.assertRaisesRegex(ValueError, "gate denied"):
                    cli.cmd_predict(args)
            gate.assert_called_once()

    def test_frozen_inputs_require_code_and_model_manifest_before_reads(self):
        args = type("Args", (), {"processed_dir": ROOT / "absent", "manifest": ROOT / "absent.csv",
            "split_summary": ROOT / "absent.json", "checkpoint": ROOT / "absent-model", "calibration": None})()
        with self.assertRaisesRegex(ValueError, "must declare"):
            cli.verify_frozen_inputs(args, {})

    def test_bias_and_checkpoint_selection_exact_ties(self):
        metrics = {kind: {"f1": .1} for kind in LABEL_TYPES}
        selected = cli.select_biases([{"o_bias": bias, "metrics": metrics} for bias in cli.BIAS_GRID])
        self.assertEqual(selected, {kind: 8 for kind in LABEL_TYPES})
        candidates = [{"epoch": epoch, "calibrated_metrics": {"macro_f1_types": .2, "any": {"f1": any_f1}}}
                      for epoch, any_f1 in [(3, .4), (2, .4), (1, .3)]]
        self.assertEqual(cli.select_candidate(candidates)["epoch"], 2)

    def test_calibration_rejects_wrong_checkpoint_or_data(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "checkpoint.json").write_text("{}", encoding="utf-8")
            saved = {"architecture": "test", "metadata": {"data_sha256": {"train.jsonl": "abc"}}}
            artifact = {"schema_version": 1, "architecture": "test", "status": "selected_on_validation",
                "validation_split": "validation", "label_types": list(LABEL_TYPES),
                "checkpoint_sha256": cli.sha256_file(checkpoint / "checkpoint.json"),
                "data_sha256": saved["metadata"]["data_sha256"], "o_bias_grid": cli.BIAS_GRID,
                "o_bias_by_type": {kind: 0 for kind in LABEL_TYPES}}
            path = checkpoint / "calibration.json"
            cli.write_json(path, artifact)
            self.assertEqual(cli.load_calibration(path, checkpoint, saved), artifact)
            artifact["data_sha256"] = {"wrong": "identity"}
            cli.write_json(path, artifact)
            with self.assertRaisesRegex(ValueError, "data identity"):
                cli.load_calibration(path, checkpoint, saved)


if __name__ == "__main__":
    unittest.main()
