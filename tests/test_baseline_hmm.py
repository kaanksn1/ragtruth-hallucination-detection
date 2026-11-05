"""Supplementary HMM math, overlap, train-only vocabulary and replay tests."""
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from ragtruth_data.alignment import word_level_example
from ragtruth_data.schema import BIO, LABEL_TYPES
from scripts import baseline_hmm as hmm


def example(rid, text, labels=()):
    return word_level_example({"id": rid, "response": text, "labels": list(labels)})


class HMMTests(unittest.TestCase):
    def setUp(self):
        self.training = [
            example("a", "safe false claim", [
                {"start": 5, "end": 16, "label_type": LABEL_TYPES[0]},
                {"start": 5, "end": 10, "label_type": LABEL_TYPES[2]}]),
            example("b", "safe safe safe"),
        ] * 1

    def test_viterbi_agrees_with_exhaustive_sequence_probability(self):
        model = hmm.fit(self.training, alpha=0.01, min_count=1)
        vocabulary, chains = hmm._compiled(model)
        symbols = [hmm.observation(w, vocabulary) for w in ("safe", "false", "claim")]
        chain = chains[LABEL_TYPES[0]]
        def probability(path):
            score = chain["initial"][path[0]] + chain["emission"][symbols[0]][path[0]]
            for i in range(1, len(path)):
                score += chain["transition"][path[i-1]][path[i]] + chain["emission"][symbols[i]][path[i]]
            return score + chain["end"][path[-1]]
        paths = list(itertools.product(range(3), repeat=3))
        expected = max(paths, key=probability)
        actual = hmm.viterbi(symbols, chain)
        self.assertEqual(actual, [BIO[index] for index in expected])
        self.assertFalse(actual[0] == "I")
        self.assertFalse(any(a == "O" and b == "I" for a, b in zip(actual, actual[1:])))

    def test_fit_predict_retains_cross_type_overlap_and_original_unicode_offsets(self):
        labels = [{"start": 3, "end": 8, "label_type": kind} for kind in LABEL_TYPES[:2]]
        training = [example(str(i), "é  ERROR ok", labels) for i in range(12)]
        model = hmm.fit(training, alpha=0.001, min_count=1)
        # Predict only tokens and offsets: no source, gold labels or metadata are read.
        output = hmm.predict(model, [{"id": "new", "tokens": ["é", "ERROR", "ok"],
                                      "offsets": [(0, 1), (3, 8), (9, 11)]}])
        self.assertEqual(output, [{"id": "new", "spans": [
            {"start": 3, "end": 8, "label_type": LABEL_TYPES[0]},
            {"start": 3, "end": 8, "label_type": LABEL_TYPES[1]}]}])

    def test_unseen_words_use_fixed_unknown_buckets_without_vocabulary_mutation(self):
        model = hmm.fit(self.training, min_count=2)
        original = json.dumps(model, sort_keys=True)
        self.assertNotIn("validation_only", model["vocabulary"])
        predictions = hmm.predict(model, [example("val", "validation_only 123 NASA Fresh !")])
        self.assertEqual(predictions[0]["id"], "val")
        self.assertEqual(json.dumps(model, sort_keys=True), original)
        self.assertEqual(hmm.observation("123", set()), "unk:number")
        self.assertEqual(hmm.observation("NASA", set()), "unk:upper")
        self.assertEqual(hmm.observation("Fresh", set()), "unk:title")
        self.assertEqual(hmm.observation("!", set()), "unk:punct")

    def test_serialized_model_replays_identical_predictions(self):
        model = hmm.fit(self.training, alpha=0.01, min_count=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            hmm._write_json(path, model)
            loaded = hmm.validate_model(json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual(hmm.predict(model, self.training), hmm.predict(loaded, self.training))
            with self.assertRaises(FileExistsError):
                hmm._write_json(path, model)

    def test_probabilities_are_normalized_with_structural_zeros(self):
        model = hmm.fit(self.training, min_count=1)
        _, chains = hmm._compiled(model)
        for chain in chains.values():
            self.assertAlmostEqual(sum(math.exp(p) for p in chain["initial"]), 1)
            for i in range(3):
                self.assertAlmostEqual(sum(math.exp(p) for p in chain["transition"][i])
                                       + math.exp(chain["end"][i]), 1)
                self.assertAlmostEqual(sum(math.exp(p[i]) for p in chain["emission"].values()), 1)
            self.assertEqual(chain["initial"][2], -math.inf)
            self.assertEqual(chain["transition"][0][2], -math.inf)

    def test_unsafe_settings_illegal_targets_and_test_access_rejected(self):
        for alpha in (0, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                hmm.fit(self.training, alpha=alpha)
        with self.assertRaises(ValueError):
            hmm.fit(self.training, min_count=0)
        illegal = example("bad", "wrong")
        illegal["bio_by_type"][LABEL_TYPES[0]] = ["I"]
        with self.assertRaises(ValueError):
            hmm.fit([illegal])
        with self.assertRaisesRegex(ValueError, "official test"):
            hmm.load_records(Path("nonexistent"), "test", Path("none"), Path("none"))

    def test_empty_response_and_duplicate_prediction_ids(self):
        model = hmm.fit(self.training)
        self.assertEqual(hmm.predict(model, [example("empty", "")]), [{"id": "empty", "spans": []}])
        with self.assertRaises(ValueError):
            hmm.predict(model, [example("same", "a"), example("same", "b")])
        with self.assertRaises(ValueError):
            hmm.fit(self.training + [self.training[0]])

    def test_cli_full_cohort_quality_filter_and_serialized_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            processed.mkdir()
            train = [{"id": ex["id"], "source_id": "source_train", "split": "train",
                      "quality": "good", "response": "safe false claim" if ex["id"] == "a" else "safe safe safe",
                      "labels": [{"start": 5, "end": 16, "label_type": LABEL_TYPES[0]}]
                      if ex["id"] == "a" else []} for ex in self.training]
            validation = [{"id": "validation", "source_id": "source_val", "split": "validation",
                           "quality": "good", "response": "safe false claim", "labels": []},
                          {"id": "excluded", "source_id": "source_val", "split": "validation",
                           "quality": "truncated", "response": "", "labels": []}]
            for split, rows in (("train", train), ("validation", validation)):
                hmm._write_predictions(processed / f"{split}.jsonl", rows)
            manifest = root / "manifest.csv"
            manifest.write_text("source_id,group_id,split\nsource_train,g_train,train\nsource_val,g_val,validation\n",
                                encoding="utf-8")
            summary = root / "summary.json"
            hmm._write_json(summary, {"processed_sha256": {f"{split}.jsonl":
                           hmm.sha256_file(processed / f"{split}.jsonl") for split in ("train", "validation")}})
            common = ["--processed-dir", str(processed), "--manifest", str(manifest), "--summary", str(summary)]
            run, replay = root / "run", root / "replay"
            with redirect_stdout(StringIO()):
                self.assertEqual(hmm.main(["train", *common, "--output", str(run)]), 0)
                self.assertEqual(hmm.main(["predict", *common, "--model", str(run / "model.json"),
                                           "--output", str(replay)]), 0)
            self.assertEqual((run / "validation_predictions.jsonl").read_bytes(),
                             (replay / "validation_predictions.jsonl").read_bytes())
            report = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(report["training_responses"], 2)
            self.assertEqual(report["validation"]["responses"], 1)
            self.assertEqual(report["evaluation"]["excluded_by_quality"], {"truncated": 1})
            self.assertFalse(report["official_test_used"])
            self.assertEqual(report["sha256"]["predictions"],
                             hmm.sha256_file(run / "validation_predictions.jsonl"))


if __name__ == "__main__":
    unittest.main()
