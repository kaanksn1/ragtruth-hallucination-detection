"""Fresh V2 holdout preparation on synthetic sources only."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_v2_data import (
    MANIFEST, SUMMARY, STUDY_ID, assign_v2_manifest, prepare_v2,
)
from ragtruth_data.schema import LABEL_TYPES


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8"))


def fixture(source):
    members, rows = [], {"train": [], "validation": []}
    for task in ("QA", "Summary", "Data2txt"):
        for number in range(13):
            side = "train" if number < 11 else "validation"
            group = f"{task}-{number}"
            # Distinct source IDs with the same content group must travel together.
            for copy in range(2 if number == 0 else 1):
                sid = f"{group}-{copy}"
                members.append(dict(source_id=sid, group_id=group, task_type=task,
                    source_dataset="synthetic", official_split="train", split=side, n_responses="2"))
                for generator in range(2):
                    response = "İstanbul has blue skies."
                    labels = [dict(start=i, end=i+8, label_type=label, text=response[i:i+8],
                                   meta="kept", implicit_true=False, due_to_null=False)
                              for i, label in enumerate(LABEL_TYPES)]
                    rows[side].append(dict(format_version="ragtruth-data-v1", id=f"{sid}-{generator}",
                        source_id=sid, split=side, official_split="train", task_type=task,
                        source_dataset="synthetic", generator=f"model-{generator}", quality="good",
                        source_text=f"Passage for {group}", response=response, labels=labels,
                        n_labels=4, has_hallucination=True, source_info={"nested": "retained"}))
    members.append(dict(source_id="original-test-only", group_id="old-test", task_type="QA",
        source_dataset="synthetic", official_split="test", split="test", n_responses="6"))
    path = source / "data/splits/split_manifest_v1.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(members[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(members)
    for side, records in rows.items():
        jsonl(source / f"data/processed/{side}.jsonl", records)
    # Deliberately unreadable JSON is a sentinel: V2 prep may not touch this file.
    (source / "data/processed/test.jsonl").write_bytes(b"\xffDO NOT OPEN ORIGINAL TEST\x00")
    return members, rows


class V2DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.source, self.root = base / "source", base / "v2"
        self.root.mkdir()
        self.members, self.rows = fixture(self.source)
        self.inputs = list((self.source / "data/processed").glob("*.jsonl")) + [self.source / "data/splits/split_manifest_v1.csv"]
        self.before = {path: digest(path) for path in self.inputs}

    def git(self, *args):
        return subprocess.check_output(["git", "-c", f"safe.directory={self.root.as_posix()}", *args],
                                       cwd=self.root, stderr=subprocess.PIPE)

    def freeze(self, status="frozen", commit=True):
        if not (self.root / ".git").exists():
            self.git("init")
        path = self.root / "docs/V2_FINAL_PROTOCOL.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((json.dumps({"status": status, "study_id": STUDY_ID}) + "\n").encode())
        if commit:
            self.git("add", "docs/V2_FINAL_PROTOCOL.json", MANIFEST, SUMMARY)
            self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                     "-c", "core.autocrlf=false", "commit", "-m", "freeze synthetic study")
        return path

    def test_group_stratification_is_deterministic_and_excludes_old_test(self):
        first = assign_v2_manifest(self.members)
        self.assertEqual(first, assign_v2_manifest(list(reversed(self.members))))
        self.assertNotIn("original-test-only", {row["source_id"] for row in first})
        for task in ("QA", "Summary", "Data2txt"):
            holdout_groups = {r["group_id"] for r in first if r["task_type"] == task and r["split"] == "test"}
            self.assertEqual(len(holdout_groups), 1)
        groups = {}
        for row in first:
            groups.setdefault(row["group_id"], set()).add(row["split"])
            if row["original_project_split"] == "validation":
                self.assertEqual(row["split"], "validation")
        self.assertTrue(all(len(sides) == 1 for sides in groups.values()))

    def test_prepare_preserves_annotations_dev_and_inputs_without_holdout(self):
        summary = prepare_v2(self.source, root=self.root)
        self.assertEqual(summary["seed"], 20261101)
        self.assertFalse(summary["original_official_test_annotations_read"])
        self.assertNotIn("test.jsonl", summary["processed_sha256"])
        for name in ("test.jsonl", "crf_test.jsonl"):
            self.assertFalse((self.root / "data/processed" / name).exists())
        original = {r["id"]: r for records in self.rows.values() for r in records}
        for side in ("train", "validation"):
            output = self.root / f"data/processed/{side}.jsonl"
            self.assertNotIn(b"\r\n", output.read_bytes())
            written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            for row in written:
                self.assertEqual(row["study_id"], STUDY_ID)
                before = original[row["id"]]
                self.assertEqual({k: v for k, v in row.items() if k not in {"study_id", "original_project_split"}},
                                 dict(before, split=side))
            if side == "validation":
                self.assertEqual({r["id"] for r in written}, {r["id"] for r in self.rows[side]})
        self.assertEqual(self.before, {path: digest(path) for path in self.inputs})
        self.assertNotIn(b"\r\n", (self.root / MANIFEST).read_bytes())

    def test_rerun_is_byte_identical_and_conflicts_refused_before_writes(self):
        prepare_v2(self.source, root=self.root)
        files = [p for p in (self.root / "data").rglob("*") if p.is_file()]
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
        prepare_v2(self.source, root=self.root)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files})
        (self.root / SUMMARY).write_bytes(b"conflicting bytes\n")
        with self.assertRaisesRegex(FileExistsError, "conflicting V2 output"):
            prepare_v2(self.source, root=self.root)
        self.assertEqual((self.root / "data/processed/train.jsonl").read_bytes(),
                         before[self.root / "data/processed/train.jsonl"][0])

    def test_annotation_changes_cannot_change_manifest_selection(self):
        prepare_v2(self.source, root=self.root)
        first = (self.root / MANIFEST).read_bytes()
        for side, records in self.rows.items():
            jsonl(self.source / f"data/processed/{side}.jsonl", [dict(r, labels=[], n_labels=0, has_hallucination=False) for r in records])
        another = self.root.parent / "another-v2"
        prepare_v2(self.source, root=another)
        self.assertEqual(first, (another / MANIFEST).read_bytes())

    def test_holdout_requires_committed_frozen_protocol(self):
        prepare_v2(self.source, root=self.root)
        with self.assertRaisesRegex(ValueError, "committed"):
            prepare_v2(self.source, root=self.root, include_holdout=True)
        self.freeze(status="draft")
        with self.assertRaisesRegex(ValueError, "frozen"):
            prepare_v2(self.source, root=self.root, include_holdout=True)
        path = self.freeze()
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "committed bytes"):
            prepare_v2(self.source, root=self.root, include_holdout=True)
        self.assertFalse((self.root / "data/processed/test.jsonl").exists())

    def test_frozen_holdout_is_fresh_original_train_and_summary_stays_immutable(self):
        prepare_v2(self.source, root=self.root)
        before = {name: (self.root / name).read_bytes() for name in (MANIFEST, SUMMARY)}
        self.freeze()
        prepare_v2(self.source, root=self.root, include_holdout=True)
        for name, payload in before.items():
            self.assertEqual((self.root / name).read_bytes(), payload)
        records = [json.loads(line) for line in (self.root / "data/processed/test.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(records)
        self.assertTrue(all(r["split"] == "test" and r["official_split"] == "train"
                            and r["original_project_split"] == "train" and r["study_id"] == STUDY_ID for r in records))
        self.assertEqual(self.before, {path: digest(path) for path in self.inputs})

    def test_original_destination_and_duplicate_response_rejected(self):
        with self.assertRaisesRegex(ValueError, "differ"):
            prepare_v2(self.source, root=self.source)
        rows = list(self.rows["train"])
        rows[1] = dict(rows[1], id=rows[0]["id"])
        jsonl(self.source / "data/processed/train.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "unique strings"):
            prepare_v2(self.source, root=self.root)
        self.assertFalse((self.root / MANIFEST).exists())

    def test_cross_split_source_group_and_count_disagreement_rejected(self):
        members = list(self.members)
        members[-2] = dict(members[-2], group_id=members[0]["group_id"])
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            assign_v2_manifest(members)
        rows = self.rows["train"][:-1]
        jsonl(self.source / "data/processed/train.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "response count"):
            prepare_v2(self.source, root=self.root)


if __name__ == "__main__":
    unittest.main()
