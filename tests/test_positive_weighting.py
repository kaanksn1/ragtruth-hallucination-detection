"""Train-only weighted BCE and checkpoint compatibility for one bounded variant."""
import copy
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_transformer as fixtures
from scripts import transformer as cli

if fixtures.HAVE_HF:
    import torch
    import transformer as tr


@unittest.skipUnless(fixtures.HAVE_HF, "install requirements-transformer.txt")
class PositiveWeightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_formula_floor_cap_zero_support_and_invalid_counts(self):
        weights = tr.weights_from_counts([1, 4, 9, 0], 10)
        self.assertEqual(weights[0], 3)
        self.assertAlmostEqual(weights[1], math.sqrt(1.5))
        self.assertEqual(weights[2:], [1., 50.])
        self.assertEqual(tr.weights_from_counts([1, 10, 0, 9], 10, 2), [2., 1., 2., 1.])
        for positive, total, cap in (([0] * 4, 0, 50), ([11, 0, 0, 0], 10, 50),
                                     ([True, 0, 0, 0], 10, 50), ([1] * 3, 10, 50),
                                     ([1] * 4, 10, float("nan")), ([1] * 4, 10, .5)):
            with self.subTest(positive=positive, total=total, cap=cap), self.assertRaises(ValueError):
                tr.weights_from_counts(positive, total, cap)

    def test_counting_uses_response_attention_intersection_and_keeps_overlaps(self):
        feature = {"input_ids": [1, 2, 3, 0], "response_mask": [0, 1, 1, 1], "attention_mask": [1, 1, 1, 0],
                   "labels_multi": [None, [1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]]}
        config = dict(fixtures.config(), positive_weighting="sqrt_capped", positive_weighting_cap=50)
        rows = [{"id": "train-only"}]
        state = torch.random.get_rng_state().clone()
        with patch.object(tr, "WindowDataset", return_value=iter([feature])) as dataset:
            result = tr.count_training_positive_weights(rows, None, config, split="train")
        dataset.assert_called_once_with(rows, None, config, training=True, epoch=0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(result["response_tokens"], 2)
        self.assertEqual(list(result["positive_response_tokens"].values()), [1, 1, 1, 0])
        self.assertEqual(result["weights_vector"], [1., 1., 1., 50.])
        for split in ("validation", "test"):
            with patch.object(tr, "WindowDataset", side_effect=AssertionError("held-out records accessed")):
                with self.subTest(split=split), self.assertRaisesRegex(ValueError, "only.*train"):
                    tr.count_training_positive_weights(rows, None, config, split=split)

    def test_counter_streams_features_instead_of_materializing_them(self):
        accessed = [False]
        class Feature(dict):
            def __getitem__(self, key):
                if key == "labels_multi":
                    accessed[0] = True
                return super().__getitem__(key)
        one = Feature(input_ids=[1], response_mask=[1], attention_mask=[1], labels_multi=[[1, 0, 0, 0]])
        def stream():
            yield one
            self.assertTrue(accessed[0], "features were collected before counting")
            yield one
        with patch.object(tr, "WindowDataset", return_value=stream()):
            result = tr.count_training_positive_weights([{}], None, fixtures.config(), split="train")
        self.assertEqual(result["windows"], 2)
        self.assertEqual(result["response_tokens"], 2)

    def test_weighted_loss_and_gradients_ignore_invalid_source_and_padding_targets(self):
        logits = torch.zeros((1, 3, 4), requires_grad=True)
        labels = torch.full_like(logits, float("nan"))
        labels[:, 1] = torch.tensor([1., 0., 1., 0.])
        loss = tr.masked_bce_loss(logits, labels, torch.tensor([[0, 1, 1]]), torch.tensor([[1, 1, 0]]),
                                  torch.tensor([2., 3., 5., 7.]))
        self.assertAlmostEqual(loss.item(), 9 * math.log(2) / 4, places=6)
        loss.backward()
        torch.testing.assert_close(logits.grad[:, 1], torch.tensor([[-.25, .125, -.625, .125]]))
        self.assertTrue((logits.grad[:, [0, 2]] == 0).all())

    def test_default_loss_matches_unit_weights_and_invalid_config_fails(self):
        logits = torch.randn(1, 3, 4)
        labels = torch.randint(0, 2, logits.shape).float()
        mask = torch.ones((1, 3))
        ordinary = tr.masked_bce_loss(logits, labels, mask, mask)
        unit = tr.masked_bce_loss(logits, labels, mask, mask, torch.ones(4))
        torch.testing.assert_close(ordinary, unit, rtol=0, atol=0)
        for changes in ({"positive_weighting": "unknown"}, {"positive_weighting_cap": float("nan")},
                        {"positive_weights": [2] * 4},
                        {"positive_weighting": "sqrt_capped", "positive_weights": [51] * 4},
                        {"positive_weighting": "sqrt_capped", "positive_weights": [1, 2, float("nan"), 4]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                tr.validate_config(dict(fixtures.config(), **changes))

    def test_weighted_checkpoint_restores_buffer_and_does_not_correct_inference_logits(self):
        config = dict(fixtures.config(), positive_weighting="sqrt_capped", positive_weighting_cap=50,
                      positive_weights=[2., 3., 5., 7.])
        model = fixtures.tiny_model().eval()
        model.positive_weights = torch.tensor(config["positive_weights"])
        plain = tr.HallucinationTransformer(copy.deepcopy(model.encoder), dropout=0).eval()
        plain.load_state_dict(model.state_dict())
        tokenizer = fixtures.offline_tokenizer()
        batch = tr.collate_features(tr.record_features(fixtures.records()[0], tokenizer, config), tokenizer.pad_token_id)
        torch.testing.assert_close(model(**batch)["logits"], plain(**batch)["logits"], rtol=0, atol=0)
        self.assertGreater(model(**batch)["loss"].item(), plain(**batch)["loss"].item())
        self.assertNotIn("positive_weights", model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint"
            with self.assertRaisesRegex(ValueError, "missing from the model"):
                tr.save_checkpoint(path, plain, tokenizer, config, {})
            self.assertFalse(path.exists())
            stats = {"split": "train", "weights_vector": config["positive_weights"]}
            tr.save_checkpoint(path, model, tokenizer, config, {"positive_weighting": stats})
            loaded, _, saved = tr.load_checkpoint(path)
            torch.testing.assert_close(loaded.positive_weights, model.positive_weights, rtol=0, atol=0)
            torch.testing.assert_close(loaded(**batch)["logits"], model(**batch)["logits"], rtol=0, atol=0)
            torch.testing.assert_close(loaded(**batch)["loss"], model(**batch)["loss"], rtol=0, atol=0)
            self.assertEqual(saved["metadata"]["positive_weighting"], stats)
            # Release HF's mapped weights before Windows removes the temp files.
            del loaded
            import gc
            gc.collect()


@unittest.skipUnless(fixtures.HAVE_HF, "install requirements-transformer.txt")
class WeightedTrainingCLITests(unittest.TestCase):
    setUp = fixtures.DataContractTests.setUp

    def test_cli_derives_counts_from_train_only_and_persists_the_same_validation_criterion(self):
        torch.set_num_threads(1)
        config = dict(self.cfg, positive_weighting="sqrt_capped", positive_weighting_cap=50)
        config_path = self.path / "weighted.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        output = self.path / "weighted-run"
        def tiny_from_config(effective):
            model = fixtures.tiny_model()
            model.positive_weights = torch.tensor(effective["positive_weights"])
            return model
        with patch.object(tr.HallucinationTransformer, "from_pretrained", side_effect=tiny_from_config), \
             patch.object(tr.AutoTokenizer, "from_pretrained", return_value=fixtures.offline_tokenizer()), \
             patch.object(tr, "count_training_positive_weights", wraps=tr.count_training_positive_weights) as count, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["train", "--config", str(config_path), "--output", str(output)]), 0)
        self.assertEqual({row["id"] for row in count.call_args.args[0]}, {row["id"] for row in self.rows["train"]})
        self.assertEqual(count.call_args.kwargs["split"], "train")
        stats = json.loads((output / "positive_weighting.json").read_text())
        summary = json.loads((output / "transformer_summary.json").read_text())
        saved = json.loads((output / summary["best_checkpoint"] / "checkpoint.json").read_text())
        self.assertEqual(summary["positive_weighting"], stats)
        self.assertEqual(saved["metadata"]["positive_weighting"], stats)
        self.assertEqual(saved["config"]["positive_weights"], stats["weights_vector"])
        self.assertIn("weighted", summary["selection_metric"])
        self.assertEqual(stats["records"], 2)
        self.assertEqual(stats["split"], "train")


if __name__ == "__main__":
    unittest.main()
