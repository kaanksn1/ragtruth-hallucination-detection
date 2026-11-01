"""Fixture-only checks for label audit identity gates and descriptive limits."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import audit_label_quality as audit


def label(kind="Evident Conflict", meta="EVIDENT CONFLICT", **changes):
    result = {"start": 0, "end": 3, "text": "abc", "label_type": kind,
              "implicit_true": False, "due_to_null": False, "meta": meta}
    result.update(changes)
    return result


def record(rid, sid, split, labels=None, text="abc def", source_text=None):
    labels = [] if labels is None else labels
    return {"format_version": "ragtruth-data-v1", "id": rid, "source_id": sid,
            "split": split, "official_split": "train", "task_type": "QA", "source_dataset": "fixture",
            "generator": "fixture-model", "quality": "good", "source_text": source_text or "context " + sid,
            "response": text, "labels": labels, "n_labels": len(labels), "has_hallucination": bool(labels)}


class Fixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.train = [record("r1", "s1", "train", [label()])]
        self.validation = [record("r2", "s2", "validation", [label("Subtle Baseless Info", "LOW INTRO OF NEW INFO")])]
        self.seal()

    def put(self, name, value):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))
        return target

    def put_json(self, name, value):
        return self.put(name, json.dumps(value, ensure_ascii=False) + "\n")

    def rebind(self):
        """Freeze deliberately changed fixtures so record checks, not hash checks, run."""
        source = json.loads((self.root / "configs/source.json").read_text(encoding="utf-8"))
        data = {name: audit.digest((self.root / name).read_bytes()) for name in audit.INPUT_NAMES if name != "configs/source.json"}
        data.update({"data/raw/" + name: value for name, value in source["sha256"].items()})
        self.put_json("docs/FINAL_PROTOCOL.json", {"status": "frozen", "data_sha256": data,
            "code_sha256": {"configs/source.json": audit.digest((self.root / "configs/source.json").read_bytes())}})

    def seal(self, train=None, validation=None, members=None):
        self.train = copy.deepcopy(self.train if train is None else train)
        self.validation = copy.deepcopy(self.validation if validation is None else validation)
        rows = []
        for side, records in (("train", self.train), ("validation", self.validation)):
            self.put("data/processed/" + side + ".jsonl", "".join(json.dumps(r) + "\n" for r in records))
            for sid in sorted({r["source_id"] for r in records}):
                rows.append({"source_id": sid, "group_id": "g" + sid, "task_type": "QA", "source_dataset": "fixture",
                             "official_split": "train", "split": side, "n_responses": sum(r["source_id"] == sid for r in records)})
        # Test membership metadata is allowed. Its annotation file is forbidden.
        rows.append({"source_id": "test-source", "group_id": "test-group", "task_type": "QA", "source_dataset": "fixture",
                     "official_split": "test", "split": "test", "n_responses": 1})
        if members is not None:
            rows = members
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=("source_id", "group_id", "task_type", "source_dataset", "official_split", "split", "n_responses"))
        writer.writeheader()
        writer.writerows(rows)
        self.put("data/splits/split_manifest_v1.csv", output.getvalue())
        raw = {name: str(index) * 64 for index, name in enumerate(("response.jsonl", "source_info.jsonl", "LICENSE"), 1)}
        self.put_json("configs/source.json", {"revision": "a" * 40, "sha256": raw})
        self.put_json("data/splits/split_summary_v1.json", {"format_version": "ragtruth-data-v1", "revision": "a" * 40,
            "raw_sha256": {name: raw[name] for name in ("response.jsonl", "source_info.jsonl")},
            "processed_sha256": {side + ".jsonl": audit.digest((self.root / "data/processed" / (side + ".jsonl")).read_bytes()) for side in audit.SPLITS},
            "responses_per_split": {"train": len(self.train), "validation": len(self.validation)},
            "sources_per_split": {side: {"total": len({r["source_id"] for r in records})}
                                  for side, records in (("train", self.train), ("validation", self.validation))}})
        self.rebind()
        self.put("data/processed/test.jsonl", "FORBIDDEN TEST ANNOTATIONS")
        self.put("data/raw/response.jsonl", "FORBIDDEN MIXED-SPLIT RAW ANNOTATIONS")


class IdentityTests(Fixture):
    def test_only_fixed_train_dev_inputs_are_read_and_preserved(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        real_read = Path.read_bytes
        touched = []
        def guarded_read(path):
            touched.append(path)
            if path.name == "test.jsonl" or "raw" in path.parts:
                raise AssertionError("official-test or raw labels must not be opened")
            return real_read(path)
        with patch.object(Path, "read_bytes", guarded_read):
            report = audit.build_report(self.root)
        self.assertEqual(report["responses"], 2)
        self.assertEqual(report["spans"], 2)
        self.assertFalse(report["provenance"]["official_test_annotations_opened"])
        self.assertFalse(report["provenance"]["raw_files_opened"])
        self.assertEqual(set(report["provenance"]["inputs_sha256"]), set(audit.INPUT_NAMES))
        self.assertEqual(report["provenance"]["cross_split_source_text_overlaps"], {"exact": 0, "casefold_whitespace": 0})
        self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))
        self.assertTrue(any(p.name == "train.jsonl" for p in touched))

    def test_hash_mismatch_fails_before_parsing_bad_json(self):
        self.put("data/processed/train.jsonl", "not json")
        with self.assertRaisesRegex(ValueError, "frozen SHA-256"):
            audit.build_report(self.root)

    def test_missing_frozen_identity_and_source_declaration_mismatch(self):
        protocol = json.loads((self.root / "docs/FINAL_PROTOCOL.json").read_text())
        del protocol["data_sha256"]["data/processed/validation.jsonl"]
        self.put_json("docs/FINAL_PROTOCOL.json", protocol)
        with self.assertRaisesRegex(ValueError, "missing frozen SHA"):
            audit.build_report(self.root)
        self.seal()
        protocol = json.loads((self.root / "docs/FINAL_PROTOCOL.json").read_text())
        protocol["data_sha256"]["data/raw/response.jsonl"] = "f" * 64
        self.put_json("docs/FINAL_PROTOCOL.json", protocol)
        with self.assertRaisesRegex(ValueError, "source SHA declaration"):
            audit.build_report(self.root)

    def test_manifest_count_and_summary_hash_disagreement_are_rejected(self):
        target = self.root / "data/splits/split_manifest_v1.csv"
        target.write_bytes(target.read_bytes().replace(b"s1,gs1,QA,fixture,train,train,1", b"s1,gs1,QA,fixture,train,train,2"))
        self.rebind()
        with self.assertRaisesRegex(ValueError, "response count differs from manifest"):
            audit.build_report(self.root)
        self.seal()
        summary = json.loads((self.root / "data/splits/split_summary_v1.json").read_text())
        summary["processed_sha256"]["train.jsonl"] = "a" * 64
        self.put_json("data/splits/split_summary_v1.json", summary)
        self.rebind()
        with self.assertRaisesRegex(ValueError, "processed SHA differs"):
            audit.build_report(self.root)

    def test_cross_split_group_and_source_text_leaks_are_rejected(self):
        target = self.root / "data/splits/split_manifest_v1.csv"
        target.write_bytes(target.read_bytes().replace(b"s2,gs2,", b"s2,gs1,"))
        self.rebind()
        with self.assertRaisesRegex(ValueError, "group occurs in multiple splits"):
            audit.build_report(self.root)
        self.seal()
        validation = copy.deepcopy(self.validation)
        validation[0]["source_text"] = self.train[0]["source_text"].upper() + "  "
        self.seal(validation=validation)
        with self.assertRaisesRegex(ValueError, "source text overlaps"):
            audit.build_report(self.root)


class RecordValidationTests(Fixture):
    def test_boolean_metadata_flags_and_bounds_are_strict(self):
        cases = [("start", True), ("end", 3.0), ("implicit_true", "false"), ("due_to_null", 1),
                 ("meta", False), ("meta", 1), ("label_type", "unknown"), ("text", "mismatch"), ("end", 99)]
        baseline = copy.deepcopy(self.train)
        for field, value in cases:
            rows = copy.deepcopy(baseline)
            rows[0]["labels"][0][field] = value
            self.seal(train=rows)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit.build_report(self.root)

    def test_record_flags_counts_unknown_quality_and_split_are_rejected(self):
        baseline = copy.deepcopy(self.train)
        for field, value in (("has_hallucination", 1), ("has_hallucination", False), ("n_labels", True),
                             ("n_labels", 3), ("quality", "unknown"), ("split", "test"), ("response", False)):
            rows = copy.deepcopy(baseline)
            rows[0][field] = value
            self.seal(train=rows)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                audit.build_report(self.root)

    def test_duplicate_ids_across_splits_and_excluded_bad_rows_fail(self):
        rows = copy.deepcopy(self.validation)
        rows[0]["id"] = self.train[0]["id"]
        self.seal(validation=rows)
        with self.assertRaisesRegex(ValueError, "duplicate response ID"):
            audit.build_report(self.root)
        self.seal(validation=[record("r2", "s2", "validation")])
        rows = copy.deepcopy(self.train)
        rows[0]["quality"] = "truncated"
        rows[0]["labels"][0]["end"] = 100
        self.seal(train=rows)
        with self.assertRaisesRegex(ValueError, "label bounds"):
            audit.build_report(self.root)

    def test_empty_dataset_and_missing_classes_have_defined_null_summaries(self):
        self.seal(train=[], validation=[])
        report = audit.build_report(self.root)
        self.assertEqual(report["responses"], 0)
        self.assertEqual(report["spans"], 0)
        self.assertEqual(set(report["types"]), set(audit.TYPES))
        for row in report["types"].values():
            self.assertEqual(row["spans"], 0)
            self.assertIsNone(row["median_words"])
            self.assertIsNone(row["p90_words"])
            self.assertIsNone(row["max_words"])
        self.assertIsNone(report["note_type_vs_label_type"]["disagreement_pct"])
        self.assertIsNone(report["identical_response_pairs"]["same_binary_decision_pct"])
        self.seal(train=[record("a", "s1", "train", [label(meta=None)])])
        report = audit.build_report(self.root)
        self.assertEqual(report["note_type_vs_label_type"]["unparsed"], 1)
        self.assertEqual(report["types"]["Subtle Conflict"]["spans"], 0)


class DescriptiveTests(unittest.TestCase):
    def test_parser_accepts_only_explicit_unambiguous_headers(self):
        accepted = {"EVIDENT CONFLICT": "Evident Conflict", "Subtle Conflict: explanation": "Subtle Conflict",
                    "**Evident Baseless Info**": "Evident Baseless Info", "Type: Subtle Baseless Info": "Subtle Baseless Info",
                    "HIGH INTRO OF NEW INFO": "Evident Baseless Info", "LOW INTRODUCTION OF INFORMATION": "Subtle Baseless Info"}
        for text, kind in accepted.items():
            with self.subTest(text=text):
                self.assertEqual(audit.note_type({"meta": text}), kind)
        for text in (None, "", "This is not an evident conflict", "Subtle Conflict and Evident Conflict",
                     "HIGH INTRO OF NEW INTRO", "Evident conflict may not be the correct category", "HIGH INTRODUCITON OF NEW INFORMATION"):
            with self.subTest(text=text):
                self.assertIsNone(audit.note_type({"meta": text}))

    def test_parser_reports_missing_empty_and_ambiguous_separately(self):
        labels = [label(meta=None), label(meta=""), label(meta="Not an evident conflict"),
                  label(meta="SUBTLE CONFLICT")]
        result = audit.note_agreement([record("a", "s1", "train", labels)])
        self.assertEqual(result["parsed"], 1)
        self.assertEqual(result["unparsed"], 3)
        self.assertEqual(result["disagreements"], 1)
        self.assertEqual(result["disagreement_pct"], 100)
        self.assertEqual(result["unparsed_reasons"], {"missing_or_null": 1, "empty": 1, "unrecognized_or_ambiguous": 1})

    def test_duplicate_rules_keep_source_identity_content_and_group_distinct(self):
        rows = [record("a", "s1", "train", [label()], source_text="context"),
                record("b", "s1", "train", [], source_text="context"),
                record("c", "s2", "train", [label()], source_text="context"),
                record("d", "s3", "train", [], source_text="different context")]
        members = {sid: {"group_id": "same-group"} for sid in ("s1", "s2", "s3")}
        same_id = audit.identical_response_agreement(rows)
        same_text = audit.identical_response_agreement(rows, "source_text")
        same_group = audit.identical_response_agreement(rows, "manifest_group", members)
        self.assertEqual(same_id["pairs"], 1)
        self.assertEqual(same_text["pairs"], 3)
        self.assertEqual(same_text["cross_source_id_pairs"], 2)
        self.assertEqual(same_group["pairs"], 6)
        self.assertEqual(same_group["different_source_text_pairs"], 3)
        self.assertEqual(same_id["any_character_agreement_f1"], 0)
        self.assertEqual(same_id["same_binary_decision_pct"], 0)
        self.assertTrue(any("neither IAA" in text for text in audit.METHOD_LIMITS))


class OutputTests(Fixture):
    def test_explicit_output_is_required_and_existing_report_is_preserved(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            audit.main([], root=self.root)
        self.assertEqual(error.exception.code, 2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(["--output", "outputs/new.json"], root=self.root), 0)
        output = self.root / "outputs/new.json"
        before = output.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            audit.main(["--output", "outputs/new.json"], root=self.root)
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(output.read_bytes(), before)

    def test_output_cannot_write_annotations_or_leave_root(self):
        for output in ("data/processed/train.jsonl", "../elsewhere.json", "outputs/bad.txt"):
            with self.subTest(output=output), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                audit.main(["--output", output], root=self.root)


if __name__ == "__main__":
    unittest.main()
