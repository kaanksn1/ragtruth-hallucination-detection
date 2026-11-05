"""Meaningful offline checks for supplementary word-level neural baselines."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAVE_TORCH = importlib.util.find_spec("torch") is not None
if HAVE_TORCH:
    import torch
    import neural_baselines as nb
    from scripts import baseline_neural as runner
    from ragtruth_data.data import sha256_file
    from ragtruth_data.schema import LABEL_TYPES
    from ragtruth_eval import evaluate_records


def record(rid, response, labels=None):
    return {"id": rid, "response": response, "labels": labels or []}


@unittest.skipUnless(HAVE_TORCH, "install requirements-baselines.txt for neural baseline checks")
class NeuralBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_vocabulary_is_train_only_deterministic_and_input_ignores_metadata(self):
        rows = [record("t1", "Alpha alpha beta"), record("t2", "beta gamma")]
        vocab = nb.build_vocabulary(rows, min_frequency=2)
        self.assertEqual(vocab, {"<PAD>": 0, "<UNK>": 1, "alpha": 2, "beta": 3})
        plain = record("v1", "Alpha private")
        labelled = {**plain, "source_text": "other", "labels": [{"malformed": "unused"}], "generator": "x"}
        self.assertEqual(nb.make_examples([plain], vocab, include_targets=False),
                         nb.make_examples([labelled], vocab, include_targets=False))
        self.assertEqual(nb.make_examples([plain], vocab, include_targets=False)[0]["ids"], [2, 1])
        self.assertNotIn("private", vocab)

    def test_chunks_cover_every_word_and_reassemble_unicode_overlapping_types(self):
        text = "İzmir çok güzel!"
        labels = [{"start": 0, "end": 9, "label_type": LABEL_TYPES[0]},
                  {"start": 6, "end": len(text), "label_type": LABEL_TYPES[2]}]
        rows = [record("x", text, labels), record("empty", " \n")]
        vocab = nb.build_vocabulary(rows, min_frequency=1)
        examples = nb.make_examples(rows, vocab, include_targets=True)
        dataset = nb.ChunkDataset(examples, 2)
        self.assertEqual([dataset[i]["start"] for i in range(len(dataset))], [0, 2])
        self.assertEqual(sum(len(dataset[i]["ids"]) for i in range(len(dataset))), 4)
        target = examples[0]["targets"]
        self.assertEqual(target[1], [1, 0, 1, 0])
        predictions = nb.scores_to_predictions(examples, [target, []], 0.5)
        self.assertEqual(predictions[0]["spans"], labels)
        self.assertEqual(predictions[1], {"id": "empty", "spans": []})
        self.assertEqual(evaluate_records(rows, predictions)["character"]["any"]["f1"], 1.0)

    def test_padding_has_no_bce_loss_or_gradient(self):
        logits = torch.zeros((2, 3, 4), requires_grad=True)
        targets = torch.zeros_like(logits)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        loss = nb.masked_bce(logits, targets, mask)
        altered = targets.clone()
        altered[~mask] = 1
        torch.testing.assert_close(loss, nb.masked_bce(logits, altered, mask))
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(logits.grad[~mask])), 0)
        self.assertGreater(int(torch.count_nonzero(logits.grad[mask])), 0)

    def test_both_encoders_are_padding_invariant_and_receive_gradients(self):
        for architecture in ("rnn", "cnn"):
            with self.subTest(architecture=architecture):
                torch.manual_seed(42)
                model = nb.WordTagger(architecture, 10, 8, 12, dropout=0)
                solo = model(torch.tensor([[2, 3]]), torch.tensor([2]))
                mixed = model(torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7]]), torch.tensor([2, 4]))
                torch.testing.assert_close(solo[0], mixed[0, :2], atol=1e-6, rtol=1e-6)
                before = model.classifier.weight.detach().clone()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                loss = nb.masked_bce(mixed, torch.ones_like(mixed),
                                     torch.tensor([[True, True, False, False], [True] * 4]))
                loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, model.classifier.weight))

    def test_inference_covers_all_chunks_and_matches_saved_model(self):
        for architecture in ("rnn", "cnn"):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as temporary:
                rows = [record("a", "one two three four five six seven"), record("b", "")]
                vocabulary = nb.build_vocabulary(rows, min_frequency=1)
                examples = nb.make_examples(rows, vocabulary, include_targets=False)
                model = nb.WordTagger(architecture, len(vocabulary), 8, 12, 0)
                scores = nb.predict_scores(model, examples, device="cpu", batch_size=2, chunk_size=3)
                self.assertEqual([len(row) for row in scores], [7, 0])
                directory = Path(temporary)
                torch.save(model.state_dict(), directory / "epoch-1.pt")
                config = {"architecture": architecture, "vocabulary": vocabulary,
                          "embedding_size": 8, "hidden_size": 12, "dropout": 0,
                          "checkpoint": "epoch-1.pt", "label_types": list(LABEL_TYPES),
                          "checkpoint_sha256": sha256_file(directory / "epoch-1.pt")}
                (directory / "model.json").write_text(json.dumps(config), encoding="utf-8")
                loaded, _ = nb.load_selected_model(directory)
                self.assertEqual(scores, nb.predict_scores(loaded, examples, device="cpu", batch_size=2, chunk_size=3))
                with (directory / "epoch-1.pt").open("ab") as handle:
                    handle.write(b"changed")
                with self.assertRaisesRegex(ValueError, "checksum"):
                    nb.load_selected_model(directory)

    def test_threshold_selection_uses_macro_then_any_and_rejects_invalid_scores(self):
        rows = [record("a", "bad good", [{"start": 0, "end": 3, "label_type": LABEL_TYPES[0]}])]
        vocabulary = nb.build_vocabulary(rows, min_frequency=1)
        examples = nb.make_examples(rows, vocabulary, include_targets=False)
        scores = [[[0.2, 0, 0, 0], [0.06, 0, 0, 0]]]
        (threshold, metrics), candidates = runner.choose_threshold(rows, examples, scores)
        self.assertEqual(threshold, 0.1)
        self.assertEqual(metrics["character"]["macro_f1_types"], 0.25)
        self.assertEqual(len(candidates), len(runner.THRESHOLDS))
        with self.assertRaises(ValueError):
            nb.scores_to_predictions(examples, [[[float("nan")] * 4] * 2], 0.5)
        with self.assertRaises(ValueError):
            nb.scores_to_predictions(examples, scores, 0)

    def test_tiny_training_reloads_selection_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            processed = directory / "processed"
            processed.mkdir()
            for split, rid, sid, response in (("train", "t", "s1", "wrong right wrong"),
                                               ("validation", "v", "s2", "wrong right")):
                row = {**record(rid, response, [{"start": 0, "end": 5, "label_type": LABEL_TYPES[0]}]),
                       "source_id": sid, "split": split, "quality": "good"}
                (processed / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            summary = {"processed_sha256": {f"{side}.jsonl": sha256_file(processed / f"{side}.jsonl")
                                            for side in ("train", "validation")}}
            (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (directory / "manifest.csv").write_text("source_id,group_id,split\ns1,g1,train\ns2,g2,validation\n", encoding="utf-8")
            argv = ["--model", "cnn", "--output-dir", str(directory / "run"), "--processed-dir", str(processed),
                    "--manifest", str(directory / "manifest.csv"), "--split-summary", str(directory / "summary.json"),
                    "--epochs", "1", "--embedding-size", "8", "--hidden-size", "12", "--batch-size", "2",
                    "--min-frequency", "1", "--threads", "1", "--device", "cpu"]
            runner.main(argv)
            saved = json.loads((directory / "run/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["validation_responses"], 1)
            self.assertFalse(saved["official_test_used"])
            self.assertTrue((directory / "run/validation_predictions.jsonl").is_file())
            with self.assertRaises(FileExistsError):
                runner.main(argv)


if __name__ == "__main__":
    unittest.main()
