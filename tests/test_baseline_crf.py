"""3. kişi: CRF baseline features, decoding, selection metric and a tiny train/predict run."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("baseline_crf", ROOT / "scripts" / "baseline_crf.py")
CRF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CRF)

from ragtruth_data.alignment import word_level_example  # noqa: E402
from ragtruth_data.schema import LABEL_TYPES  # noqa: E402

try:
    import pycrfsuite
except ImportError:  # CI installs no model packages
    pycrfsuite = None

SOURCE = "The company was founded in Ankara in 2020. It makes solar panels."
RESPONSE = "The company was founded in Istanbul in 2018. It makes solar panels."


def record(rid, response, labels, source=SOURCE):
    return {"id": rid, "source_text": source, "response": response, "labels": labels, "quality": "good"}


def span(text, sub, label_type):
    i = text.index(sub)
    return {"start": i, "end": i + len(sub), "label_type": label_type}


class FeatureTests(unittest.TestCase):
    def test_source_features_flag_words_and_numbers_missing_from_source(self):
        tokens = word_level_example(record("1", RESPONSE, []))["tokens"]
        feats = CRF.token_features(tokens, CRF.source_index(SOURCE))
        by_word = dict(zip(tokens, feats))
        self.assertIn("src:content_new", by_word["Istanbul"])
        self.assertIn("src:num_new", by_word["2018"])
        self.assertIn("src:content_seen", by_word["solar"])
        # company, founded, Istanbul, 2018: two of four content words are new
        self.assertIn("src:sentence_new=mid", by_word["founded"])
        self.assertIn("src:sentence_new_number", by_word["founded"])
        self.assertIn("src:sentence_new=0", by_word["panels"])

    def test_response_only_view_has_no_source_features(self):
        tokens = word_level_example(record("1", RESPONSE, []))["tokens"]
        feats = CRF.token_features(tokens, None)
        self.assertFalse(any(f.startswith(("src:", "-1:src:", "1:src:")) for item in feats for f in item))

    def test_features_do_not_depend_on_labels_or_metadata(self):
        labelled = record("1", RESPONSE, [span(RESPONSE, "Istanbul", "Evident Conflict")])
        labelled.update(task_type="QA", generator="gpt-4")
        plain = record("2", RESPONSE, [])
        a = CRF.token_features(word_level_example(labelled)["tokens"], CRF.source_index(labelled["source_text"]))
        b = CRF.token_features(word_level_example(plain)["tokens"], CRF.source_index(plain["source_text"]))
        self.assertEqual(a, b)

    def test_sentence_ids_ignore_decimal_points(self):
        tokens = ["It", "rose", "3", ".", "5", "%", ".", "Then", "fell", "."]
        self.assertEqual(CRF.sentence_ids(tokens), [0, 0, 0, 0, 0, 0, 0, 1, 1, 1])


class DecodingTests(unittest.TestCase):
    def test_threshold_tags_start_every_run_with_b(self):
        self.assertEqual(CRF.threshold_tags([0.1, 0.6, 0.7, 0.2, 0.9], 0.5), ["O", "B", "I", "O", "B"])

    def test_invalid_starts_count_i_without_preceding_tag(self):
        self.assertEqual(CRF.invalid_starts(["I", "I", "O", "I", "B", "I"]), 2)

    def test_spans_keep_overlapping_types(self):
        offsets = [(0, 3), (4, 11), (12, 15)]
        tags = {t: ["O", "O", "O"] for t in LABEL_TYPES}
        tags["Evident Conflict"] = ["B", "I", "O"]
        tags["Subtle Baseless Info"] = ["O", "B", "I"]
        spans = CRF.tags_to_spans(offsets, tags)
        self.assertEqual(spans, [
            {"start": 0, "end": 11, "label_type": "Evident Conflict"},
            {"start": 4, "end": 15, "label_type": "Subtle Baseless Info"},
        ])

    def test_decodings_put_viterbi_first_then_thresholds_descending(self):
        names = [d["name"] for d in CRF.decodings({"decoding": {"posterior_thresholds": [0.1, 0.5, 0.3]}})]
        self.assertEqual(names, ["viterbi", "posterior_0.5", "posterior_0.3", "posterior_0.1"])


class MetricTests(unittest.TestCase):
    def test_character_scores_per_type_and_union(self):
        gold = {"1": [{"start": 0, "end": 10, "label_type": "Evident Conflict"}]}
        pred = {"1": [{"start": 5, "end": 15, "label_type": "Evident Baseless Info"}]}
        scores = CRF.char_scores(gold, pred)
        self.assertAlmostEqual(scores["any"]["precision"], 0.5)
        self.assertAlmostEqual(scores["any"]["recall"], 0.5)
        self.assertEqual(scores["Evident Conflict"]["f1"], 0.0)

    def test_overlapping_gold_labels_are_not_double_counted(self):
        gold = {"1": [{"start": 0, "end": 10, "label_type": "Evident Conflict"},
                      {"start": 5, "end": 10, "label_type": "Evident Conflict"}]}
        self.assertEqual(CRF.char_scores(gold, gold)["Evident Conflict"]["gold_chars"], 10)

    def test_selection_uses_any_f1_and_keeps_first_on_ties(self):
        def cand(name, f1):
            return {"name": name, "validation": {"any": {"f1": f1}}}
        candidates = [cand("a", 0.2), cand("b", 0.3), cand("c", 0.3)]
        self.assertEqual(CRF.select_candidate(candidates)["name"], "b")


@unittest.skipIf(pycrfsuite is None, "python-crfsuite not installed (pip install -r requirements-crf.txt)")
class TinyTrainingTests(unittest.TestCase):
    def test_train_and_predict_round_trip(self):
        rows = []
        for k in range(30):
            if k % 2:
                text = f"The company was founded in Istanbul in {2000 + k}. It makes solar panels."
                labels = [span(text, f"Istanbul in {2000 + k}", "Evident Conflict")]
            else:
                text, labels = "The company was founded in Ankara in 2020. It makes solar panels.", []
            rows.append(record(str(k), text, labels))
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp) / "processed"
            processed.mkdir()
            for split, part in (("train", rows[:20]), ("validation", rows[20:])):
                with open(processed / f"{split}.jsonl", "w", encoding="utf-8") as f:
                    f.writelines(json.dumps(r) + "\n" for r in part)
                with open(processed / f"crf_{split}.jsonl", "w", encoding="utf-8") as f:
                    f.writelines(json.dumps({**word_level_example(r), "quality": "good"}) + "\n" for r in part)
            config = json.loads((ROOT / "configs" / "crf.json").read_text(encoding="utf-8"))
            config.update(processed_dir=str(processed), grid=[{"c1": 0.0, "c2": 0.01}])
            config["training"]["max_iterations"] = 50
            config["training"]["min_feature_frequency"] = 1
            config_path = Path(tmp) / "crf.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            out = Path(tmp) / "run"
            CRF.main(["train", "--config", str(config_path), "--output", str(out)])

            summary = json.loads((out / "crf_summary.json").read_text(encoding="utf-8"))
            self.assertGreater(summary["selected"]["validation"]["Evident Conflict"]["f1"], 0.9)
            self.assertIn("response_only_control", summary)
            first = [json.loads(line) for line in open(out / "validation_predictions.jsonl", encoding="utf-8")]
            self.assertEqual([p["id"] for p in first], [r["id"] for r in rows[20:]])

            CRF.main(["predict", "--model-dir", str(out / "models" / "source"), "--split", "validation",
                      "--processed-dir", str(processed), "--output", str(out / "again.jsonl")])
            again = [json.loads(line) for line in open(out / "again.jsonl", encoding="utf-8")]
            self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()
