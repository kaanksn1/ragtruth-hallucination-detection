"""Frozen-cohort integrity and hand-calculated independent metrics, synthetic only."""
from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import independent_rescore as audit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)).encode("utf-8"))


def expected_metrics(responses):
    # Hand fixture: one perfectly matched EC span plus optional clean responses.
    return {"responses": responses,
        "character": {"any": {"precision": 1., "recall": 1., "f1": 1.},
                      "macro_f1_types": .25, **{kind: {"f1": float(i == 0)} for i, kind in enumerate(audit.TYPES)}},
        "span_exact": {"typed": {"f1": 1.}}}


class ArchiveFixture:
    def __init__(self, root):
        self.root = root
        label = {"start": 0, "end": 3, "label_type": audit.TYPES[0], "text": "İ🙂 ",
                 "implicit_true": True, "due_to_null": True}
        self.rows = [
            dict(id="tr", source_id="train", split="train", quality="good", response="train", labels=[]),
            dict(id="v1", source_id="val", split="train", quality="good", response="İ🙂 a", labels=[label, dict(label)]),
            dict(id="v2", source_id="val", split="train", quality="good", response="clean", labels=[]),
            dict(id="v3", source_id="val", split="train", quality="truncated", response="short", labels=[]),
            dict(id="v4", source_id="val", split="train", quality="incorrect_refusal", response="refusal", labels="[]"),
            dict(id="te", source_id="test", split="test", quality="good", response="X", labels=[
                {"start": 0, "end": 1, "label_type": audit.TYPES[0], "text": "X"}]),
        ]
        self.predictions = {
            "validation": [{"id": "v1", "spans": [{"start": 0, "end": 3, "label_type": audit.TYPES[0]}]},
                           {"id": "v2", "spans": []}],
            "test": [{"id": "te", "spans": [{"start": 0, "end": 1, "label_type": audit.TYPES[0]}]}],
        }
        self.refresh()

    def refresh(self, *, opaque_test=False):
        root = self.root
        raw = root / "data/raw/response.jsonl"
        write_rows(raw, self.rows)
        if opaque_test:
            lines = raw.read_text(encoding="utf-8").splitlines()
            lines[-1] = '{"labels":[NOT_JSON],"response":NOT_JSON,"quality":NOT_JSON,"id":"te","source_id":"test","split":"test"}'
            raw.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        write_rows(root / "data/raw/source_info.jsonl", [])
        (root / "data/raw/LICENSE").write_bytes(b"synthetic license\n")
        source = {"revision": audit.REVISION, "sha256": {name: audit.sha256(root / "data/raw" / name)
                    for name in ("response.jsonl", "source_info.jsonl", "LICENSE")}}
        write_json(root / "configs/source.json", source)
        manifest = root / "data/splits/split_manifest_v1.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source_id", "group_id", "official_split", "split", "n_responses"], lineterminator="\n")
            writer.writeheader()
            writer.writerows([
                dict(source_id="train", group_id="g-tr", official_split="train", split="train", n_responses=1),
                dict(source_id="val", group_id="g-val", official_split="train", split="validation", n_responses=4),
                dict(source_id="test", group_id="g-test", official_split="test", split="test", n_responses=1),
            ])
        summary = {"revision": audit.REVISION, "raw_sha256": source["sha256"],
            "processed_sha256": {"validation.jsonl": "1" * 64},
            "responses_per_split": {"train": 1, "validation": 4, "test": 1}}
        write_json(root / "data/splits/split_summary_v1.json", summary)
        decision = {"status": "frozen", "models": {"transformer": {}},
            "code_sha256": {"configs/source.json": audit.sha256(root / "configs/source.json")},
            "data_sha256": {name: audit.sha256(root / name) for name in (
                "data/raw/response.jsonl", "data/raw/source_info.jsonl", "data/raw/LICENSE",
                "data/splits/split_manifest_v1.csv", "data/splits/split_summary_v1.json")}}
        write_json(root / "docs/FINAL_PROTOCOL.json", decision)
        for split, predictions in self.predictions.items():
            pred = root / f"predictions/{split}.jsonl"
            write_rows(pred, predictions)
            is_dev = split == "validation"
            data = {"quality_filter": "good", "included_responses": 2 if is_dev else 1,
                "file_responses": 4 if is_dev else 1,
                "excluded_by_quality": {"truncated": 1, "incorrect_refusal": 1} if is_dev else {},
                "sha256": {"gold": ("1" if is_dev else "2") * 64,
                    "split_manifest": audit.sha256(manifest),
                    "split_summary": audit.sha256(root / "data/splits/split_summary_v1.json")}}
            reference = {"split": split, "metric_version": "ragtruth-project-eval-v1", "data": data,
                "metric_policy": {"whitespace": "included", "macro_labels": list(audit.TYPES)},
                "models": {"transformer": {"predictions_path": f"predictions/{split}.jsonl",
                    "predictions_sha256": audit.sha256(pred), "metrics": expected_metrics(2 if is_dev else 1)}}}
            write_json(root / f"results/{split}_comparison.json", reference)
        write_json(root / "configs/final/final_evaluate.protocol.json", {
            "status": "frozen", "scientific_decision_sha256": audit.sha256(root / "docs/FINAL_PROTOCOL.json"),
            "inputs_sha256": {"data/processed/test.jsonl": "2" * 64,
                "predictions/test.jsonl": audit.sha256(root / "predictions/test.jsonl")}})
        self.trusted = {name: audit.sha256(root / name) for name in audit.TRUSTED_REFERENCES}


class IndependentRescoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "fixture"
        self.fixture = ArchiveFixture(self.root)

    def score(self, **kwargs):
        with patch.dict(audit.TRUSTED_REFERENCES, self.fixture.trusted, clear=True):
            return audit.rescore(root=self.root, **kwargs)

    def test_full_cohort_quality_duplicates_unicode_and_opaque_test(self):
        self.fixture.refresh(opaque_test=True)
        result = self.score(boundary=True)
        data = result["validation"]
        self.assertEqual(data["responses"], 2)
        self.assertEqual(data["data"]["file_responses"], 4)
        self.assertFalse(result["provenance"]["original_official_test_labels_decoded"])
        model = data["models"]["transformer"]
        self.assertEqual(model["any_f1"], 1.)
        self.assertEqual(model["macro_f1"], .25)
        self.assertEqual(model["gold_spans"], 1)  # exact raw duplicates counted once
        self.assertEqual(model["exact_typed_f1"], 1.)
        self.assertIn("many-to-many", result["metric_policy"]["overlap_typed_f1"])
        self.assertEqual(model["boundary"]["gold_and_pred_stripped_exact"], 1.)

    def test_gate_rejects_test_and_test_boundaries_before_any_data_read(self):
        with patch.object(audit, "provenance") as provenance, patch.object(audit, "raw_gold") as gold:
            with self.assertRaisesRegex(ValueError, "explicit"):
                audit.rescore(root=self.root, split="test")
            with self.assertRaisesRegex(ValueError, "validation-only"):
                audit.rescore(root=self.root, split="test", allow_test=True, boundary=True)
            provenance.assert_not_called()
            gold.assert_not_called()
        result = self.score(split="test", allow_test=True)
        self.assertTrue(result["provenance"]["original_official_test_labels_decoded"])
        self.assertNotIn("boundary", result["test"]["models"]["transformer"])
        with patch.object(Path, "open") as opened:
            with self.assertRaisesRegex(ValueError, "explicit"):
                audit.raw_gold(self.root, "test", {}, {}, {})
            opened.assert_not_called()

    def test_missing_duplicate_and_foreign_predictions_never_shrink_cohort(self):
        for mutation in ("missing", "duplicate", "foreign", "excluded"):
            with self.subTest(mutation=mutation):
                rows = list(self.fixture.predictions["validation"])
                if mutation == "missing":
                    rows.pop()
                elif mutation == "duplicate":
                    rows.append(rows[0])
                else:
                    rows[1] = {"id": "outside" if mutation == "foreign" else "v3", "spans": []}
                self.fixture.predictions["validation"] = rows
                self.fixture.refresh()
                with self.assertRaisesRegex(ValueError, "IDs|ID"):
                    self.score()
                self.fixture.predictions["validation"] = [
                    {"id": "v1", "spans": [{"start": 0, "end": 3, "label_type": audit.TYPES[0]}]},
                    {"id": "v2", "spans": []}]

    def test_invalid_bounds_types_and_prediction_schema_rejected(self):
        for span in ({"start": True, "end": 2, "label_type": audit.TYPES[0]},
                     {"start": -1, "end": 2, "label_type": audit.TYPES[0]},
                     {"start": 1, "end": 1, "label_type": audit.TYPES[0]},
                     {"start": 0, "end": 5, "label_type": audit.TYPES[0]},
                     {"start": 0, "end": 1, "label_type": "other"}):
            with self.subTest(span=span):
                self.fixture.predictions["validation"][0]["spans"] = [span]
                self.fixture.refresh()
                with self.assertRaisesRegex(ValueError, "span"):
                    self.score()
        self.fixture.predictions["validation"][0]["extra"] = 1
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.score()

    def test_raw_quality_annotation_text_and_duplicate_ids_rejected(self):
        for mutation in ("quality", "text", "duplicate"):
            with self.subTest(mutation=mutation):
                fixture = ArchiveFixture(self.root)
                if mutation == "quality":
                    fixture.rows[1]["quality"] = "unknown"
                elif mutation == "text":
                    fixture.rows[1]["labels"][0]["text"] = "changed"
                else:
                    fixture.rows[2]["id"] = fixture.rows[1]["id"]
                fixture.refresh()
                self.fixture = fixture
                with self.assertRaises(ValueError):
                    self.score()

    def test_hash_changes_rejected_before_gold_read(self):
        for name in ("data/raw/response.jsonl", "data/splits/split_manifest_v1.csv", "configs/source.json",
                     "results/validation_comparison.json", "predictions/validation.jsonl"):
            with self.subTest(name=name):
                self.fixture.refresh()
                path = self.root / name
                path.write_bytes(path.read_bytes() + b" ")
                with patch.object(audit, "raw_gold") as gold:
                    with self.assertRaisesRegex(ValueError, "hash changed"):
                        self.score()
                    gold.assert_not_called()

    def test_test_prediction_must_match_existing_frozen_job(self):
        protocol = self.root / "configs/final/final_evaluate.protocol.json"
        value = audit.read_json(protocol)
        value["inputs_sha256"]["predictions/test.jsonl"] = "0" * 64
        write_json(protocol, value)
        # Even when a fixture protocol is trusted, its internal frozen hash binding matters.
        self.fixture.trusted[protocol.relative_to(self.root).as_posix()] = audit.sha256(protocol)
        with patch.object(audit, "raw_gold") as gold:
            with self.assertRaisesRegex(ValueError, "not frozen"):
                self.score(split="test", allow_test=True)
            gold.assert_not_called()

    def test_reference_and_prediction_cannot_be_tampered_together(self):
        path = self.root / "predictions/validation.jsonl"
        path.write_bytes(b'{"id":"v1","spans":[]}\n')
        reference = self.root / "results/validation_comparison.json"
        value = audit.read_json(reference)
        value["models"]["transformer"]["predictions_sha256"] = audit.sha256(path)
        write_json(reference, value)
        with self.assertRaisesRegex(ValueError, "frozen reference hash"):
            self.score()

    def test_prediction_change_during_scoring_cannot_get_stale_provenance(self):
        original = audit.load_pairs
        def changing_predictions(path, gold):
            value = original(path, gold)
            path.write_bytes(path.read_bytes() + b" ")
            return value
        with patch.object(audit, "load_pairs", side_effect=changing_predictions):
            with self.assertRaisesRegex(ValueError, "changed during audit"):
                self.score()

    def test_portable_prediction_pack_with_explicit_remapping(self):
        pack = self.root.parent / "portable-pack"
        pack.mkdir()
        shutil.copyfile(self.root / "predictions/validation.jsonl", pack / "relocated.jsonl")
        result = self.score(pack=pack, predictions=[("transformer", "relocated.jsonl")])
        self.assertEqual(result["validation"]["responses"], 2)
        self.assertEqual(result["provenance"]["predictions"]["transformer"]["path"], "relocated.jsonl")
        with self.assertRaisesRegex(ValueError, "duplicate model"):
            self.score(pack=pack, predictions=[("transformer", "relocated.jsonl")] * 2)
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.score(predictions=[("newly-trained-model", "relocated.jsonl")])

    def test_unicode_overlap_math_and_empty_boundary_do_not_create_false_matches(self):
        ec, other = audit.TYPES[:2]
        # 3 union gold characters, 2 predicted; one typed span duplicated, one overlapping type.
        gold = [(0, 2, ec), (0, 2, ec), (1, 3, other)]
        pred = [(1, 3, ec)]
        result = audit.summary([(gold, pred)])
        self.assertEqual(result["any_p"], 1.)
        self.assertAlmostEqual(result["any_r"], 2/3)
        self.assertAlmostEqual(result["any_f1"], .8)
        self.assertEqual(result["exact_typed_f1"], 0.)
        self.assertAlmostEqual(result["macro_f1"], .125)
        self.assertIsNone(audit.strip_edges((0, 2, ec), " ."))
        diagnostics = audit.boundary_diagnostics([([(0, 2, ec)], [(0, 2, ec)], " .")])
        self.assertEqual(diagnostics["gold_and_pred_stripped_exact"], 0.)
        self.assertEqual(audit.summary([])["any_f1"], 0.)
        self.assertEqual(audit.response_level([])["accuracy"], 0.)

    def test_overlap_is_labeled_coverage_and_not_exact_matching(self):
        kind = audit.TYPES[0]
        pairs = [([(0, 1, kind), (2, 3, kind)], [(0, 3, kind)])]
        self.assertEqual(audit.overlap(pairs), 1.)  # many-to-many coverage, not 1:1 matching
        self.assertEqual(audit.exact(pairs), 0.)

    def test_cli_output_is_new_lf_json_and_invalid_paths_never_read_data(self):
        with patch.dict(audit.TRUSTED_REFERENCES, self.fixture.trusted, clear=True), redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(["--root", str(self.root), "--output", "outputs/new.json"]), 0)
        output = self.root / "outputs/new.json"
        before = output.read_bytes()
        self.assertNotIn(b"\r\n", before)
        for target in ("outputs/new.json", "results/overwrite.json", "../outside.json"):
            with self.subTest(target=target), patch.object(audit, "rescore") as scorer, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    audit.main(["--root", str(self.root), "--output", target])
                scorer.assert_not_called()
        self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
