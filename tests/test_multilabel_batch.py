"""Regression checks for the owner's explicitly assigned multi-hot batch fix."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_tokenizer import FakeFastTokenizer  # noqa: E402
from ragtruth_data.features import build_features, pad_batch  # noqa: E402
from ragtruth_data.schema import LABEL_TYPES  # noqa: E402


def example_features(with_labels=True):
    response = "The shop opened in 1999 and serves Indian food."

    def label(text, kind):
        start = response.index(text)
        return {"start": start, "end": start + len(text), "label_type": kind}

    records = [
        {"id": "overlap", "response": response, "source_text": "The shop opened in 2004.",
         "labels": [label("opened in 1999", LABEL_TYPES[0]), label("opened", LABEL_TYPES[1]),
                    label("1999", LABEL_TYPES[2]), label("Indian food", LABEL_TYPES[3])]},
        {"id": "negative", "response": "A shop.", "source_text": "A shop.", "labels": []},
    ]
    return build_features(records, FakeFastTokenizer(), max_length=64,
                          min_context_tokens=8, with_labels=with_labels)


class MultiLabelBatchTests(unittest.TestCase):
    def test_overlap_negatives_and_excluded_positions_survive_unequal_length_batch(self):
        features = example_features()
        original = copy.deepcopy(features)
        self.assertEqual([f["id"] for f in features], ["overlap", "negative"])
        self.assertGreater(len(features[0]["input_ids"]), len(features[1]["input_ids"]))
        batch = pad_batch(features, 0, label_key="labels_multi")
        width = len(batch["input_ids"][0])
        for labels, mask in zip(batch["labels"], batch["response_mask"]):
            self.assertEqual(len(labels), width)
            self.assertTrue(all(len(target) == 4 for target in labels))
            self.assertTrue(all(type(value) is float for target in labels for value in target))
            for target, active in zip(labels, mask):
                if not active:
                    self.assertEqual(target, [0.0] * 4)
        # "1999" belongs to both Evident Conflict and Evident Baseless Info.
        response = "The shop opened in 1999 and serves Indian food."
        start = response.index("1999")
        positions = [i for i, offset in enumerate(features[0]["response_offsets"])
                     if offset is not None and offset[0] <= start < offset[1]]
        self.assertTrue(positions)
        self.assertEqual(batch["labels"][0][positions[0]], [1.0, 0.0, 1.0, 0.0])
        self.assertTrue(all(any(target[t] for target in batch["labels"][0]) for t in range(4)))
        self.assertTrue(any(batch["response_mask"][1]))
        self.assertTrue(all(target == [0.0] * 4 for target in batch["labels"][1]))
        self.assertEqual(batch["response_mask"][1][len(features[1]["input_ids"]):],
                         [0] * (width - len(features[1]["input_ids"])))
        self.assertEqual(features, original)

    def test_inference_features_do_not_acquire_training_targets(self):
        batch = pad_batch(example_features(with_labels=False), 0, label_key="labels_multi")
        self.assertNotIn("labels", batch)
        self.assertIn("response_mask", batch)

    def test_malformed_targets_fail_before_tensor_conversion(self):
        features = example_features()
        position = features[0]["response_mask"].index(1)
        for invalid in (None, [1, 0], [0, 0, -100, 0]):
            with self.subTest(target=invalid):
                changed = copy.deepcopy(features)
                changed[0]["labels_multi"][position] = invalid
                with self.assertRaisesRegex(ValueError, "four binary values"):
                    pad_batch(changed, 0, label_key="labels_multi")
        features[0]["labels_multi"].pop()
        with self.assertRaisesRegex(ValueError, "align with input_ids"):
            pad_batch(features, 0, label_key="labels_multi")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_pytorch_bce_backward_masks_source_specials_and_padding(self):
        import torch
        from torch.nn import functional as F

        features = example_features()
        batch = pad_batch(features, 0, label_key="labels_multi", return_tensors="pt")
        self.assertEqual(batch["labels"].shape, (*batch["input_ids"].shape, 4))
        self.assertEqual(batch["labels"].dtype, torch.float32)
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["response_mask"].dtype, torch.long)
        mask = batch["response_mask"].bool()
        logits = torch.zeros_like(batch["labels"], requires_grad=True)
        loss = F.binary_cross_entropy_with_logits(logits[mask], batch["labels"][mask])
        changed = logits.detach().clone()
        changed[~mask] = 100
        changed_targets = batch["labels"].clone()
        changed_targets[~mask] = 1
        self.assertEqual(loss.item(), F.binary_cross_entropy_with_logits(
            changed[mask], changed_targets[mask]).item())
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertTrue((logits.grad[~mask] == 0).all().item())
        self.assertTrue((logits.grad[mask] != 0).all().item())
        # Existing CE/BIO targets keep their integer dtype and -100 convention.
        for key in ("labels_any", "labels_bio", "labels_bio_by_type"):
            with self.subTest(label_key=key):
                previous = pad_batch(features, 0, label_key=key, return_tensors="pt")
                self.assertEqual(previous["labels"].dtype, torch.long)
                targets = previous["labels"][0] if key == "labels_bio_by_type" else previous["labels"]
                self.assertTrue((targets[~mask] == -100).all().item())


if __name__ == "__main__":
    unittest.main()
