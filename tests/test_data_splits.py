"""2. kişi: grouped split rules on synthetic sources, and checks on the prepared files when present."""

import csv
import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data import data, splits  # noqa: E402
from ragtruth_data.alignment import bio_tags, tags_to_char_spans, word_tokenize  # noqa: E402


def fixture(n_per_task=40, n_test=6):
    sources, responses = {}, []
    sid = 0
    for task, ds in (("QA", "MARCO"), ("Summary", "CNN/DM"), ("Data2txt", "Yelp")):
        for k in range(n_per_task + n_test):
            sid += 1
            info = {"question": f"q{sid}", "passages": f"p{sid}"} if task == "QA" else (
                {"name": f"biz{sid}", "address": "a", "city": "c"} if task == "Data2txt" else f"news {sid}")
            sources[str(sid)] = {"source_id": str(sid), "task_type": task, "source": ds,
                                 "source_info": info, "prompt": f"prompt {sid}"}
            for m in range(6):
                responses.append({"id": f"{sid}-{m}", "source_id": str(sid),
                                  "split": "test" if k < n_test else "train", "labels": []})
    # a duplicate of source 10 under another id, and one duplicating an official test source
    sources["900"] = dict(sources["10"], source_id="900")
    sources["901"] = dict(sources["1"], source_id="901")
    for m in range(6):
        responses.append({"id": f"900-{m}", "source_id": "900", "split": "train", "labels": []})
        responses.append({"id": f"901-{m}", "source_id": "901", "split": "train", "labels": []})
    return sources, responses


class SplitRuleTests(unittest.TestCase):
    def setUp(self):
        self.sources, self.responses = fixture()
        self.split_of, self.group_of = splits.assign_splits(self.sources, self.responses, 0.15, 42)

    def test_official_test_untouched_and_duplicates_follow_it(self):
        for r in self.responses:
            if r["split"] == "test":
                self.assertEqual(self.split_of[r["source_id"]], "test")
        self.assertEqual(self.split_of["901"], "test")  # same content as an official test source

    def test_same_content_sources_share_a_side(self):
        self.assertEqual(self.group_of["900"], self.group_of["10"])
        self.assertEqual(self.split_of["900"], self.split_of["10"])
        splits.check_no_leakage(self.split_of, self.group_of, self.responses)

    def test_fraction_per_stratum_and_determinism(self):
        per = defaultdict(Counter)
        for sid, side in self.split_of.items():
            if side != "test":
                per[self.sources[sid]["task_type"]][side] += 1
        for c in per.values():
            self.assertAlmostEqual(c["validation"] / (c["train"] + c["validation"]), 0.15, delta=0.03)
        again, _ = splits.assign_splits(self.sources, self.responses, 0.15, 42)
        self.assertEqual(again, self.split_of)

    def test_labels_do_not_influence_the_split(self):
        labelled = [dict(r, labels=[{"start": 0, "end": 1}]) for r in self.responses]
        self.assertEqual(splits.assign_splits(self.sources, labelled, 0.15, 42)[0], self.split_of)

    def test_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.csv"
            splits.write_manifest(p, self.sources, self.responses, self.split_of, self.group_of)
            self.assertEqual(splits.read_manifest(p), self.split_of)

    def test_verify_raw_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, "download_data.py"):
                data.verify_raw(d)


PROC = ROOT / "data/processed"
MANIFEST = ROOT / "data/splits/split_manifest_v1.csv"


@unittest.skipUnless((PROC / "train.jsonl").exists() and (ROOT / "data/raw/response.jsonl").exists(),
                     "run scripts/download_data.py and scripts/prepare_data.py first")
class PreparedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = list(data.read_jsonl(PROC / "train.jsonl")) + list(data.read_jsonl(PROC / "validation.jsonl"))
        with open(MANIFEST, encoding="utf-8") as f:
            cls.manifest = {row["source_id"]: row for row in csv.DictReader(f)}

    def test_counts_match_summary_and_manifest(self):
        summary = json.loads((ROOT / "data/splits/split_summary_v1.json").read_text(encoding="utf-8"))
        got = Counter(r["split"] for r in self.records)
        for side in ("train", "validation"):
            self.assertEqual(got[side], summary["responses_per_split"][side])
        for r in self.records:
            self.assertEqual(self.manifest[r["source_id"]]["split"], r["split"])
            self.assertEqual(r["official_split"], "train")
        by_group = defaultdict(set)
        for row in self.manifest.values():
            by_group[row["group_id"]].add(row["split"])
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))
        for name, h in summary["processed_sha256"].items():
            self.assertEqual(data.sha256_file(PROC / name), h, name)

    def test_original_text_offsets_types_and_overlaps_kept(self):
        raw = {str(r["id"]): r for r in data.read_jsonl(ROOT / "data/raw/response.jsonl") if r["split"] == "train"}
        self.assertEqual(len(raw), len(self.records))
        for r in self.records:
            orig = raw[r["id"]]
            self.assertEqual(r["response"], orig["response"])
            self.assertEqual(sorted((l["start"], l["end"], l["label_type"], l["text"]) for l in orig["labels"]),
                             sorted((l["start"], l["end"], l["label_type"], l["text"]) for l in r["labels"]))

    def test_word_tags_cover_every_gold_character(self):
        for r in self.records:
            if not r["labels"]:
                continue
            _, offs = word_tokenize(r["response"])
            back = tags_to_char_spans(offs, bio_tags(offs, r["labels"]))
            for l in r["labels"]:
                for c in range(l["start"], l["end"]):
                    if not r["response"][c].isspace():
                        self.assertTrue(any(b["start"] <= c < b["end"] for b in back), (r["id"], l))


if __name__ == "__main__":
    unittest.main()
