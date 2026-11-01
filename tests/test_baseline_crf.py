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


class SelectBestConfigurationTests(unittest.TestCase):
    @staticmethod
    def metrics(any_f1, macro_f1):
        return {"any": {"f1": any_f1}, "macro_f1_types": macro_f1}

    def test_detection_f1_outranks_macro_f1_which_outranks_declaration_order(self):
        # A per-type-style configuration can raise macro F1 while lowering detection F1
        # (exactly what the real CRF run showed: macro 19.99%->20.84%, any 39.08%->38.84%).
        # Both 'global' and 'per_type' here have a higher macro F1 than 'default', but a
        # lower detection F1 -- they must still lose to whichever has the best detection F1.
        candidates = {"default": self.metrics(0.40, 0.15), "global": self.metrics(0.391, 0.20),
                      "per_type": self.metrics(0.388, 0.208), "joint": self.metrics(0.40, 0.19)}
        # 'default' and 'joint' tie on detection F1 (0.40); 'joint' wins on the macro F1 tie-break.
        self.assertEqual(CRF.select_best_configuration(candidates), "joint")
        exact_tie = {"global": self.metrics(0.5, 0.2), "per_type": self.metrics(0.5, 0.2)}
        self.assertEqual(CRF.select_best_configuration(exact_tie), "global")  # declared first

    def test_empty_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            CRF.select_best_configuration({})


class PerTypeThresholdTests(unittest.TestCase):
    """The shared-threshold grid picks one posterior cutoff from the combined 'any' F1,
    which a rare type's few gold characters barely move. `tune_type_thresholds` instead
    searches each type independently against its own characters, with a false-positive
    guardrail (MIN_PRECISION_FLOOR) so flooding a rare type is never preferred merely for
    a marginal F1 gain over predicting nothing."""

    RARE = LABEL_TYPES[1]
    RARE_OTHER = LABEL_TYPES[0]

    def test_each_type_is_tuned_against_its_own_characters(self):
        offsets = [(0, 2), (3, 5)]
        gold = {"r": [{"start": 0, "end": 2, "label_type": self.RARE}]}
        run = {"probs": {"r": {t: ([0.6, 0.1] if t == self.RARE else [0.0, 0.0]) for t in LABEL_TYPES}}}
        examples = [{"id": "r", "offsets": offsets}]
        thresholds, table = CRF.tune_type_thresholds(examples, run, gold, [0.05, 0.3, 0.5])
        self.assertEqual(thresholds[self.RARE], 0.5)
        best = next(row for row in table[self.RARE] if row["threshold"] == 0.5)
        self.assertEqual(best["f1"], 1.0)
        for other in LABEL_TYPES:
            if other != self.RARE:
                self.assertEqual(thresholds[other], 0.5)  # never predicts; ties keep the highest threshold

    def test_joint_search_beats_independent_per_type_when_they_disagree(self):
        # 12 single-character tokens. Evident Conflict's own-best threshold (0.5) recalls its
        # 2 gold characters but also floods 8 unrelated characters; Subtle Conflict is clean.
        # Independently, EC's own F1 at 0.5 (0.333) beats predicting nothing (0.0), so
        # tune_type_thresholds keeps it -- but for the combined 'any' metric, dropping EC
        # entirely (predicting nothing, threshold 0.7) raises detection F1 from 0.5 to 0.667.
        offsets = [(i, i + 1) for i in range(12)]
        gold = {"r": [{"start": 0, "end": 2, "label_type": self.RARE_OTHER},
                      {"start": 10, "end": 12, "label_type": self.RARE}]}
        probs = {self.RARE_OTHER: [0.6] * 10 + [0.0, 0.0], self.RARE: [0.0] * 10 + [0.6, 0.6]}
        probs.update({t: [0.0] * 12 for t in LABEL_TYPES if t not in (self.RARE, self.RARE_OTHER)})
        run = {"probs": {"r": probs}}
        examples = [{"id": "r", "offsets": offsets}]
        grid = [0.5, 0.7]

        independent, _ = CRF.tune_type_thresholds(examples, run, gold, grid)
        self.assertEqual(independent[self.RARE_OTHER], 0.5)  # own F1 0.333 beats 0.0 at 0.7
        independent_spans, _ = CRF.decode_per_type(examples, run, independent)
        independent_any_f1 = CRF.char_scores(gold, independent_spans)["any"]["f1"]
        self.assertAlmostEqual(independent_any_f1, 0.5)

        joint, info = CRF.tune_joint_thresholds(examples, run, gold, grid)
        self.assertEqual(joint[self.RARE_OTHER], 0.7)  # joint search drops the flooding type
        self.assertEqual(joint[self.RARE], 0.5)
        self.assertAlmostEqual(info["selected"]["any"]["f1"], 2 / 3)
        self.assertGreater(info["selected"]["any"]["f1"], independent_any_f1)  # the guarantee this exists for

    def test_flooding_threshold_is_rejected_below_the_precision_floor(self):
        offsets = [(2 * i, 2 * i + 1) for i in range(200)]  # 200 single-char tokens, one true positive
        gold = {"r": [{"start": 0, "end": 1, "label_type": self.RARE}]}
        probs = {self.RARE: [0.02] * 200}
        probs.update({t: [0.0] * 200 for t in LABEL_TYPES if t != self.RARE})
        run = {"probs": {"r": probs}}
        examples = [{"id": "r", "offsets": offsets}]
        thresholds, table = CRF.tune_type_thresholds(examples, run, gold, [0.01, 0.5])
        flooded = next(row for row in table[self.RARE] if row["threshold"] == 0.01)
        self.assertLess(flooded["precision"], CRF.MIN_PRECISION_FLOOR)
        self.assertGreater(flooded["f1"], 0)  # the rejected option is not simply the worst F1
        # Consecutive flagged tokens decode as one bridged span (same rule as viterbi runs),
        # so 200 scattered single-char tokens flood into one 399-character span here.
        self.assertEqual(flooded["predicted_chars"], 399)
        self.assertEqual(thresholds[self.RARE], 0.5)  # conservative fallback, not the degenerate flood
        spans, _ = CRF.decode_per_type(examples, run, thresholds)
        self.assertEqual(spans["r"], [])


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
            fixture_summary = Path(tmp) / "split_summary.json"
            fixture_summary.write_text(json.dumps({"processed_sha256": {
                p.name: CRF.sha256_file(p) for p in processed.glob("*.jsonl")
            }}), encoding="utf-8")
            config.update(processed_dir=str(processed), split_summary=str(fixture_summary),
                          grid=[{"c1": 0.0, "c2": 0.01}])
            config["training"]["max_iterations"] = 50
            config["training"]["min_feature_frequency"] = 1
            config_path = Path(tmp) / "crf.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            out = Path(tmp) / "Masaüstü" / "run"
            CRF.main(["train", "--config", str(config_path), "--output", str(out)])

            summary = json.loads((out / "crf_summary.json").read_text(encoding="utf-8"))
            self.assertGreater(summary["selected"]["validation"]["Evident Conflict"]["f1"], 0.9)
            self.assertIn("response_only_control", summary)
            first = list(CRF.read_jsonl(out / "validation_predictions.jsonl"))
            self.assertEqual([p["id"] for p in first], [r["id"] for r in rows[20:]])

            CRF.main(["predict", "--model-dir", str(out / "models" / "source"), "--split", "validation",
                      "--processed-dir", str(processed), "--output", str(out / "again.jsonl")])
            again = list(CRF.read_jsonl(out / "again.jsonl"))
            self.assertEqual(first, again)
            metadata = json.loads((out / "models/source/model_config.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["label_types"], list(LABEL_TYPES))
            self.assertEqual(len(metadata["models_sha256"]), 4)
            CRF.main(["predict", "--model-dir", str(out / "models/source"), "--split", "validation",
                      "--processed-dir", str(processed), "--limit", "3", "--output", str(out / "limited.jsonl")])
            self.assertEqual(list(CRF.read_jsonl(out / "limited.jsonl")), first[:3])


if __name__ == "__main__":
    unittest.main()
