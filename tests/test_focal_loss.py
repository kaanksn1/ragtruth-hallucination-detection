"""Masked sigmoid focal loss, legacy compatibility and epoch retention checks."""
from contextlib import redirect_stdout
import gc
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
class FocalLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gamma_zero_matches_alpha_weighted_bce_and_gradients(self):
        logits = torch.tensor([[[1., -1., .5, -.5]]], requires_grad=True)
        targets = torch.tensor([[[1., 0., 1., 0.]]])
        mask = torch.ones((1, 1))
        actual = tr.masked_focal_loss(logits, targets, mask, mask, gamma=0, alpha=.75)
        other = logits.detach().clone().requires_grad_()
        expected = nn_bce(other, targets, pos_weight=torch.full((4,), 3.)) * .25
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(logits.grad, other.grad, atol=1e-7, rtol=1e-6)

    def test_default_focal_has_analytical_zero_logit_value_and_excludes_nan_targets(self):
        logits = torch.zeros((1, 3, 4), requires_grad=True)
        targets = torch.full_like(logits, float("nan"))
        targets[:, 1] = torch.tensor([1., 0., 1., 0.])
        response, attention = torch.tensor([[0, 1, 1]]), torch.tensor([[1, 1, 0]])
        with torch.no_grad():
            logits[:, [0, 2]] = float("nan")
        loss = tr.masked_focal_loss(logits, targets, response, attention)
        self.assertAlmostEqual(loss.item(), math.log(2) * .125, places=7)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad[:, [0, 2]] == 0).all())
        self.assertTrue((logits.grad[:, 1, [0, 2]] < 0).all())
        self.assertTrue((logits.grad[:, 1, [1, 3]] > 0).all())

    def test_large_finite_logits_and_fractional_gamma_have_finite_gradients(self):
        for gamma in (0., .5, 2.):
            logits = torch.tensor([[[1000., -1000., -1000., 1000.]]], requires_grad=True)
            labels = torch.tensor([[[1., 0., 1., 0.]]])
            mask = torch.ones((1, 1))
            loss = tr.masked_focal_loss(logits, labels, mask, mask, gamma=gamma)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertTrue((logits.grad[..., :2] == 0).all())

    def test_empty_mask_and_empty_batch_are_differentiable_zero(self):
        for shape in ((1, 2, 4), (0, 0, 4)):
            logits = torch.full(shape, float("nan"), requires_grad=True)
            labels = torch.full(shape, float("nan"))
            mask = torch.zeros(shape[:2])
            loss = tr.masked_focal_loss(logits, labels, mask, mask)
            self.assertEqual(loss.item(), 0.)
            loss.backward()
            self.assertTrue((logits.grad == 0).all())

    def test_config_and_active_nonfinite_targets_fail_before_training(self):
        config = fixtures.config()
        tr.validate_config(config)  # Missing loss fields remain ordinary BCE.
        changes = ({"loss": "unknown"}, {"loss": "focal", "positive_weighting": "sqrt_capped"},
                   {"focal_gamma": -1}, {"focal_gamma": float("inf")}, {"focal_gamma": float("nan")},
                   {"focal_gamma": True}, {"focal_alpha": 0}, {"focal_alpha": 1},
                   {"focal_alpha": float("nan")}, {"save_all_epochs": 1})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                tr.validate_config(dict(config, **change))
        with patch.object(tr.AutoConfig, "from_pretrained", side_effect=AssertionError("download attempted")):
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                tr.HallucinationTransformer.from_pretrained(dict(config, loss="focal", positive_weighting="sqrt_capped"))
        mask = torch.ones((1, 1))
        for logits, targets in ((torch.full((1, 1, 4), float("inf")), torch.zeros((1, 1, 4))),
                                (torch.zeros((1, 1, 4)), torch.full((1, 1, 4), .5))):
            with self.assertRaisesRegex(ValueError, "finite response logits and binary targets"):
                tr.masked_focal_loss(logits, targets, mask, mask)

    def test_focal_checkpoint_restores_loss_and_legacy_bce_config_stays_compatible(self):
        tokenizer = fixtures.offline_tokenizer()
        for mode in ("bce", "focal"):
            config = fixtures.config()
            if mode == "focal":
                config.update(loss="focal", focal_gamma=2., focal_alpha=.75)
            model = tiny_from_config(config).eval()
            batch = tr.collate_features(tr.record_features(fixtures.records()[0], tokenizer, config), tokenizer.pad_token_id)
            reference = model(**batch)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint"
                tr.save_checkpoint(path, model, tokenizer, config, {"fixture": True})
                loaded, _, saved = tr.load_checkpoint(path)
                self.assertEqual(loaded.loss_mode, mode)
                self.assertEqual(saved["metadata"]["loss_mode"], mode)
                torch.testing.assert_close(loaded(**batch)["loss"], reference["loss"], atol=0, rtol=0)
                torch.testing.assert_close(loaded(**batch)["logits"], reference["logits"], atol=0, rtol=0)
                if mode == "bce":
                    # Pre-extension v2 metadata also loads with ordinary BCE.
                    saved["metadata"].pop("loss_mode")
                    saved["metadata"].pop("loss_parameters")
                    (path / "checkpoint.json").write_text(json.dumps(saved), encoding="utf-8")
                    legacy, _, _ = tr.load_checkpoint(path)
                    self.assertEqual(legacy.loss_mode, "bce")
                    del legacy
                else:
                    self.assertEqual(saved["metadata"]["loss_parameters"], {"gamma": 2., "alpha": .75})
                    saved["metadata"]["loss_mode"] = "bce"
                    (path / "checkpoint.json").write_text(json.dumps(saved), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "loss metadata"):
                        tr.load_checkpoint(path)
                del loaded
                gc.collect()


def nn_bce(logits, labels, pos_weight=None):
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def tiny_from_config(config):
    return tr.HallucinationTransformer(fixtures.tiny_model().encoder, dropout=config["dropout"],
        positive_weights=config.get("positive_weights"), loss=config.get("loss", "bce"),
        focal_gamma=config.get("focal_gamma", 2.), focal_alpha=config.get("focal_alpha", .75))


@unittest.skipUnless(fixtures.HAVE_HF, "install requirements-transformer.txt")
class EpochRetentionTests(unittest.TestCase):
    setUp = fixtures.DataContractTests.setUp

    def test_save_all_retains_nonimproving_epoch_but_default_saves_only_improvements(self):
        torch.set_num_threads(1)
        for save_all in (False, True):
            config = dict(self.cfg, epochs=3)
            if save_all:
                config.update(save_all_epochs=True, loss="focal", focal_gamma=2., focal_alpha=.75)
            path = self.path / ("all.json" if save_all else "default.json")
            path.write_text(json.dumps(config), encoding="utf-8")
            output = self.path / ("all" if save_all else "default")
            phases = [{"loss": value} for value in (.5, .2, .4, .1, .3, .3)]
            with patch.object(tr.HallucinationTransformer, "from_pretrained", side_effect=tiny_from_config), \
                 patch.object(tr.AutoTokenizer, "from_pretrained", return_value=fixtures.offline_tokenizer()), \
                 patch.object(tr, "run_epoch", side_effect=phases), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["train", "--config", str(path), "--output", str(output)]), 0)
            summary = json.loads((output / "transformer_summary.json").read_text())
            expected = ["checkpoints/epoch-1", "checkpoints/epoch-2"] + (["checkpoints/epoch-3"] if save_all else [])
            self.assertEqual(summary["saved_checkpoints"], expected)
            self.assertEqual(summary["best_checkpoint"], "checkpoints/epoch-2")
            self.assertEqual(summary["loss_mode"], "focal" if save_all else "bce")
            self.assertEqual(summary["save_all_epochs"], save_all)
            for relative in expected:
                saved = json.loads((output / relative / "checkpoint.json").read_text())
                epoch = int(relative.rsplit("-", 1)[1])
                self.assertEqual(saved["metadata"]["epoch"], epoch)
                self.assertEqual(saved["metadata"]["validation_loss"], [.2, .1, .3][epoch - 1])
                self.assertEqual(saved["metadata"]["loss_mode"], summary["loss_mode"])
            if not save_all:
                self.assertFalse((output / "checkpoints/epoch-3").exists())


if __name__ == "__main__":
    unittest.main()
