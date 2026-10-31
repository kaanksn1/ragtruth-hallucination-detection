"""Checks for support counts and honest validation error example selection."""
import unittest

from ragtruth_data.schema import LABEL_TYPES
from scripts.error_analysis import fenced, select_examples, support


def record(rid, labels):
    return {"id": rid, "response": "abc def", "source_text": "Original source.",
            "task_type": "QA", "generator": "fixture", "labels": labels}


class ErrorAnalysisTests(unittest.TestCase):
    def test_overlap_counts_characters_once_but_retains_raw_span_support(self):
        kind = LABEL_TYPES[0]
        rows = [record("x", [{"start": 0, "end": 3, "label_type": kind},
                              {"start": 1, "end": 3, "label_type": kind, "implicit_true": True}])]
        word = [{"id": "x", "tokens": ["abc", "def"],
                 "bio_by_type": {t: ["B", "O"] if t == kind else ["O", "O"] for t in LABEL_TYPES}}]
        result = support(rows, word)
        self.assertEqual(result[kind]["raw_spans"], 2)
        self.assertEqual(result[kind]["characters"], 3)
        self.assertEqual(result[kind]["positive_word_fraction"], .5)
        self.assertEqual(result[kind]["implicit_true_spans"], 1)
        self.assertEqual(result[LABEL_TYPES[1]]["characters"], 0)
        with self.assertRaisesRegex(ValueError, "word-view IDs"):
            support(rows, [])

    def test_error_categories_are_distinct_deterministic_and_original_text_is_preserved(self):
        rare = {"start": 0, "end": 3, "label_type": LABEL_TYPES[1]}
        ordinary = {"start": 0, "end": 7, "label_type": LABEL_TYPES[0]}
        rows = [record("rare", [rare]), record("miss", [ordinary]), record("false-alarm", [])]
        predictions = [{"id": "rare", "spans": []}, {"id": "miss", "spans": []},
                       {"id": "false-alarm", "spans": [ordinary]}]
        result = select_examples(rows, predictions)
        self.assertEqual([r["category"] for r in result],
                         ["missed_subtle_conflict", "false_negative", "false_positive"])
        self.assertEqual(len({r["id"] for r in result}), 3)
        self.assertEqual(result, select_examples(list(reversed(rows)), list(reversed(predictions))))
        self.assertEqual(result[0]["response"], "abc def")

    def test_perfect_predictions_do_not_invent_error_examples(self):
        rows = [record("x", [{"start": 0, "end": 3, "label_type": LABEL_TYPES[0]}])]
        self.assertEqual(select_examples(rows, [{"id": "x", "spans": rows[0]["labels"]}]), [])
        with self.assertRaisesRegex(ValueError, "IDs"):
            select_examples(rows, [])

    def test_dataset_backticks_cannot_close_report_code_block(self):
        text = "```\n<script>sample</script>\n````"
        value = fenced(text)
        self.assertTrue(value.startswith("`````text\n"))
        self.assertTrue(value.endswith("\n`````"))

    def test_same_error_category_fills_requested_count_and_invalid_limit_is_rejected(self):
        span = {"start": 0, "end": 3, "label_type": LABEL_TYPES[0]}
        rows = [record(str(i), []) for i in range(4)]
        predictions = [{"id": row["id"], "spans": [span]} for row in rows]
        examples = select_examples(rows, predictions, 3)
        self.assertEqual(len(examples), 3)
        self.assertEqual(len({row["id"] for row in examples}), 3)
        for value in (0, -1, True):
            with self.subTest(limit=value), self.assertRaisesRegex(ValueError, "positive integer"):
                select_examples(rows, predictions, value)


if __name__ == "__main__":
    unittest.main()
