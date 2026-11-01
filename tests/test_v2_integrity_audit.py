"""Hand-computed cases for the independent set-based audit scorer."""
import unittest
from scripts.audit_v2_integrity import independent_scores


class IntegrityAuditTests(unittest.TestCase):
    def test_overlapping_types_unicode_and_exact_bounds(self):
        gold = [{"id": "a", "response": "İş 😀!", "labels": [
            {"start": 0, "end": 3, "label_type": "Evident Conflict"},
            {"start": 2, "end": 4, "label_type": "Subtle Conflict"}]}]
        prediction = [{"id": "a", "spans": [
            {"start": 0, "end": 3, "label_type": "Evident Conflict"},
            {"start": 3, "end": 5, "label_type": "Subtle Conflict"}]}]
        metrics = independent_scores(gold, prediction)
        self.assertEqual(metrics["character"]["any"]["tp"], 4)
        self.assertEqual(metrics["character"]["any"]["fp"], 1)
        self.assertEqual(metrics["character"]["any"]["fn"], 0)
        self.assertEqual(metrics["character"]["Subtle Conflict"]["f1"], 0.5)
        self.assertEqual(metrics["character"]["macro_f1_types"], 0.375)
        self.assertEqual(metrics["span_exact"]["typed"]["f1"], 0.5)

    def test_empty_gold_and_missing_prediction_ids(self):
        gold = [{"id": "a", "labels": []}]
        metrics = independent_scores(gold, [{"id": "a", "spans": []}])
        self.assertEqual(metrics["character"]["any"]["f1"], 0)
        self.assertEqual(metrics["span_exact"]["typed"]["f1"], 0)
        with self.assertRaises(AssertionError):
            independent_scores(gold, [])


if __name__ == "__main__":
    unittest.main()
