"""Saved-output demo checks: offsets, overlap, strict IDs and safe HTML embedding."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data.data import sha256_file, write_jsonl
from ragtruth_data.schema import LABEL_TYPES
from scripts import build_demo as demo


def records():
    return [{"id": "one", "source_id": "source-one", "split": "validation", "quality": "good",
             "task_type": "QA", "generator": "generator-a", "source_text": "The source supports BC.",
             "response": "A😊BC de", "labels": [
                 {"start": 1, "end": 4, "label_type": LABEL_TYPES[0], "meta": "annotation-only"},
                 {"start": 2, "end": 6, "label_type": LABEL_TYPES[1]}]},
            {"id": "two", "source_id": "source-two", "split": "validation", "quality": "good",
             "task_type": "Summary", "generator": "generator-b", "source_text": "A shop.",
             "response": "A shop.", "labels": []}]


def predictions(rows=None):
    return [{"id": row["id"], "spans": [dict(start=s["start"], end=s["end"], label_type=s["label_type"])
                                             for s in row["labels"]]} for row in (records() if rows is None else rows)]


class DemoTests(unittest.TestCase):
    def test_unicode_offsets_and_overlap_reconstruct_every_original_character(self):
        row = records()[0]
        spans = row["labels"] + [row["labels"][0]]
        segmented = demo.span_segments(row["response"], spans)
        self.assertEqual("".join(segment["text"] for segment in segmented), row["response"])
        self.assertEqual([(s["text"], s["types"]) for s in segmented],
                         [("A", []), ("😊", [0]), ("BC", [0, 1]), (" d", [1]), ("e", [])])
        self.assertTrue(all(row["response"][s["start"]:s["end"]] == s["text"] for s in segmented))
        self.assertEqual(demo.span_segments("", []), [])

    def test_html_script_closing_markup_and_unicode_are_only_data(self):
        rows = records()
        attack = '</script><script>alert("x")</script><img src=x onerror=alert(1)> & __JAVASCRIPT__\u2028'
        rows[0]["source_text"] = attack
        rows[1]["response"] = attack + "😊"
        model_name = '<svg onload="alert(2)">'
        payload = demo.build_payload(rows, {model_name: predictions(rows)})
        document = demo.build_document(payload, title="<script>bad title</script>")
        self.assertNotIn(attack, document)
        self.assertNotIn(model_name, document)
        self.assertIn("&lt;script&gt;bad title&lt;/script&gt;", document)
        embedded = re.search(r'<script id="demo-data" type="application/json">(.*?)</script>', document, re.S).group(1)
        restored = json.loads(embedded)
        self.assertEqual(restored["examples"][0]["source_text"], attack)
        self.assertEqual(restored["examples"][1]["response"], attack + "😊")
        self.assertEqual(restored["models"], [model_name])
        self.assertNotIn("innerHTML", document)
        self.assertIn("document.createTextNode", document)

    def test_missing_extra_and_duplicate_prediction_ids_fail_before_display_limit(self):
        rows = records()
        variants = [predictions()[:1], predictions() + [{"id": "extra", "spans": []}], predictions() * 2]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "missing|extra|unique"):
                demo.build_payload(rows, {"Model": variant}, max_examples=1)

    def test_bad_offsets_and_unknown_types_are_rejected(self):
        for start, end, kind in ((True, 2, LABEL_TYPES[0]), (-1, 2, LABEL_TYPES[0]),
                                 (2, 2, LABEL_TYPES[0]), (0, 100, LABEL_TYPES[0]), (0, 1, "Other")):
            rows = predictions()
            rows[0]["spans"] = [{"start": start, "end": end, "label_type": kind}]
            with self.subTest(start=start, end=end, kind=kind), self.assertRaisesRegex(ValueError, "offsets|unknown"):
                demo.build_payload(records(), {"Model": rows})

    def test_test_or_mixed_files_are_rejected_even_for_excluded_quality_rows(self):
        rows = records() + [dict(records()[1], id="never-read-test", split="test", quality="truncated")]
        with self.assertRaisesRegex(ValueError, "validation records only"):
            demo.build_payload(rows, {"Model": predictions()}, max_examples=1)

    def test_negatives_model_variants_and_gold_are_distinct(self):
        predicted = predictions()
        alternate = [{"id": row["id"], "spans": []} for row in records()]
        payload = demo.build_payload(records(), {"Transformer": predicted, "CRF": alternate})
        first = payload["examples"][0]
        self.assertTrue(first["exact_match"]["Transformer"])
        self.assertFalse(first["exact_match"]["CRF"])
        self.assertEqual(first["predictions"]["CRF"]["spans"], [])
        self.assertEqual(len(first["gold"]["spans"]), 2)
        self.assertNotIn("annotation-only", json.dumps(payload))
        self.assertTrue(all("meta" not in span for span in first["gold"]["spans"]))
        self.assertTrue(payload["examples"][1]["exact_match"]["CRF"])
        with self.assertRaisesRegex(ValueError, "positive"):
            demo.build_payload(records(), {"Model": predicted}, max_examples=0)

    def test_cli_builds_a_complete_offline_file_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            record_file, prediction_file = folder / "records.jsonl", folder / "predictions.jsonl"
            output = folder / "demo.html"
            write_jsonl(record_file, records())
            write_jsonl(prediction_file, predictions())
            command = ["--records", str(record_file), "--predictions", f"Transformer={prediction_file}",
                       "--predictions", f"CRF={prediction_file}", "--output", str(output)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(demo.main(command), 0)
            document = output.read_text(encoding="utf-8")
            self.assertIn("Show human annotations", document)
            self.assertIn("offline viewer", document)
            self.assertNotRegex(document, r'<script[^>]+src=|<link[^>]+href=')
            original = sha256_file(output)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                demo.main(command)
            self.assertEqual(sha256_file(output), original)
            duplicate = ["--records", str(record_file), "--predictions", f"Same={prediction_file}",
                         "--predictions", f"Same={prediction_file}", "--output", str(folder / "duplicate.html")]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                demo.main(duplicate)


if __name__ == "__main__":
    unittest.main()
