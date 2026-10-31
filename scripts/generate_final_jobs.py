"""Hash-only final job generation after a committed scientific decision.

This helper never parses test annotations, executes a model, or evaluates a
score. Generated job/protocol pairs must be committed before run_job uses them.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import METRIC_FILES
from ragtruth_data.data import sha256_file
from scripts.run_job import git

MODEL_NAMES = ("crf_source", "crf_response_only", "transformer")
OPTIONAL_REFERENCE = "transformer_reference"
COMMON_FILES = ("scripts/generate_final_jobs.py", "scripts/run_job.py", "ragtruth_data/data.py",
                "ragtruth_data/schema.py", "ragtruth_data/alignment.py",
                "data/splits/split_manifest_v1.csv", "data/splits/split_summary_v1.json")
CRF_CODE_FILES = ("scripts/baseline_crf.py", "requirements-crf.txt")
TRANSFORMER_CODE_FILES = ("transformer.py", "scripts/transformer.py", "scripts/calibrate_transformer.py",
                          "scripts/baseline_crf.py", "ragtruth_data/features.py",
                          "requirements-transformer.txt", "requirements-crf.txt")
CRF_MODEL_FILES = ("model_config.json", "evident_conflict.crfsuite", "subtle_conflict.crfsuite",
                   "evident_baseless_info.crfsuite", "subtle_baseless_info.crfsuite")
TRANSFORMER_MINIMUM_FILES = ("checkpoint.json", "head.safetensors", "encoder/config.json",
                             "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                             "tokenizer/special_tokens_map.json")
TEST_GOLD = "data/processed/test.jsonl"
TEST_CRF = "data/processed/crf_test.jsonl"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(root, value, *, must_exist=True, directory=False):
    _require(isinstance(value, (str, Path)), "paths must be repository-relative strings")
    value = str(value).replace("\\", "/")
    _require(value and not PureWindowsPath(value).drive and not PurePosixPath(value).is_absolute()
             and ".." not in PurePosixPath(value).parts, "absolute/traversing paths are not allowed")
    result = (root / value).resolve()
    _require(result != root and result.is_relative_to(root), "path leaves repository")
    if must_exist:
        _require(result.is_dir() if directory else result.is_file(), "missing required input: " + value)
    return result


def _relative(root, path):
    path = path.resolve()
    _require(path.is_relative_to(root) and path != root, "input leaves repository")
    return path.relative_to(root).as_posix()


def _json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _model_files(root, directory, required):
    for name in required:
        _path(root, directory.relative_to(root) / name)
    files = []
    for path in sorted(directory.rglob("*")):
        # Resolve every child before hashing, including symlinked files.
        _require(path.resolve().is_relative_to(root), "model artifact leaves repository")
        if path.is_file():
            files.append(_relative(root, path))
    _require(files, "model directory has no files")
    return files


def _model_names(decision):
    models = decision.get("models")
    _require(isinstance(models, dict) and set(models) in (set(MODEL_NAMES), set(MODEL_NAMES) | {OPTIONAL_REFERENCE}),
             "scientific decision must name crf_source, crf_response_only and transformer, plus only optional transformer_reference")
    return MODEL_NAMES + ((OPTIONAL_REFERENCE,) if OPTIONAL_REFERENCE in models else ())


def load_decision(decision_path, *, root=ROOT):
    """Validate committed decision and every explicitly frozen artifact/code/data hash."""
    root = Path(root).resolve()
    path = _path(root, decision_path)
    relative = _relative(root, path)
    try:
        committed = git(root, "show", "HEAD:" + relative)
    except subprocess.CalledProcessError as exc:
        raise ValueError("scientific decision must already be committed at HEAD") from exc
    _require(committed == path.read_bytes(), "scientific decision differs from committed HEAD bytes")
    decision = json.loads(committed)
    _require(isinstance(decision, dict) and decision.get("status") == "frozen", "scientific decision must be frozen")
    model_names = _model_names(decision)
    _require(isinstance(decision.get("artifacts_sha256"), dict) and decision["artifacts_sha256"],
             "scientific decision must freeze model artifacts_sha256")
    declared = {}
    for field in ("artifacts_sha256", "code_sha256", "data_sha256"):
        values = decision.get(field, {})
        _require(isinstance(values, dict), field + " must map relative paths to SHA-256")
        for name, digest in values.items():
            _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid SHA-256 in " + field)
            source = _path(root, name)
            canonical = _relative(root, source)
            _require(sha256_file(source) == digest, "frozen scientific input changed: " + name)
            _require(canonical not in declared or declared[canonical] == digest, "conflicting frozen input hashes")
            declared[canonical] = digest
    artifacts = {_relative(root, _path(root, name)) for name in decision["artifacts_sha256"]}
    model_files, directories = {}, {}
    for name in MODEL_NAMES[:2]:
        entry = decision["models"][name]
        _require(isinstance(entry, dict) and "model_dir" in entry, name + " needs model_dir")
        directory = _path(root, entry["model_dir"], directory=True)
        directories[name] = _relative(root, directory)
        model_files[name] = _model_files(root, directory, CRF_MODEL_FILES)
    for name in model_names[2:]:
        entry = decision["models"][name]
        _require(isinstance(entry, dict) and "checkpoint" in entry and "calibration" in entry,
                 name + " needs checkpoint and calibration")
        checkpoint = _path(root, entry["checkpoint"], directory=True)
        directories[name] = _relative(root, checkpoint)
        model_files[name] = _model_files(root, checkpoint, TRANSFORMER_MINIMUM_FILES)
        _require(any(filename.startswith(directories[name] + "/encoder/") and filename.endswith(".safetensors")
                     for filename in model_files[name]), "checkpoint is missing encoder safetensors")
        calibration = _path(root, entry["calibration"])
        directories[name + "_calibration"] = _relative(root, calibration)
        model_files[name].append(directories[name + "_calibration"])
    directories["calibration"] = directories["transformer_calibration"]  # Backward-compatible primary alias.
    if OPTIONAL_REFERENCE in model_names:
        _require((directories["transformer"], directories["transformer_calibration"]) !=
                 (directories[OPTIONAL_REFERENCE], directories[OPTIONAL_REFERENCE + "_calibration"]),
                 "transformer_reference must not duplicate the primary checkpoint/calibration pair")
    for names in model_files.values():
        _require(set(names) <= artifacts, "every selected model/checkpoint/calibration file must be in artifacts_sha256")
    return decision, relative, declared, directories, model_files


def _pair(root, output_dir, stem, job, decision_relative):
    config_path = output_dir / (stem + ".json")
    protocol_path = output_dir / (stem + ".protocol.json")
    job["frozen_protocol"] = _relative(root, protocol_path)
    encoded = _json_bytes(job)
    inputs = {name: sha256_file(_path(root, name)) for name in job["inputs"]}
    protocol = {"status": "frozen", "scientific_decision": decision_relative,
                "scientific_decision_sha256": sha256_file(root / decision_relative),
                "test_config_sha256": hashlib.sha256(encoded).hexdigest(), "inputs_sha256": inputs}
    return (config_path, encoded), (protocol_path, _json_bytes(protocol))


def _write_pairs(root, pairs):
    paths = [path for pair in pairs for path, _ in pair]
    _require(len(paths) == len(set(paths)), "generated paths collide")
    for path in paths:
        _require(path.is_relative_to(root), "output leaves repository")
        if path.exists() or path.is_symlink():
            raise FileExistsError("refusing to overwrite generated file: " + str(path))
    for pair in pairs:
        for path, content in pair:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)
    return [{"job": _relative(root, pair[0][0]), "protocol": _relative(root, pair[1][0])} for pair in pairs]


def generate_predictions(decision_path, output_dir="configs/final", *, root=ROOT):
    root = Path(root).resolve()
    decision, decision_relative, declared, directories, model_files = load_decision(decision_path, root=root)
    destination = _path(root, output_dir, must_exist=False, directory=True)
    seed = decision.get("seed", 42)
    _require(type(seed) is int and seed >= 0, "scientific seed must be a nonnegative integer")
    pairs = []
    for name in _model_names(decision):
        inputs = set(COMMON_FILES) | set(declared) | {decision_relative, TEST_GOLD} | set(model_files[name])
        if name.startswith("crf_"):
            inputs.update((*CRF_CODE_FILES, TEST_CRF))
            command = ["{python}", "scripts/baseline_crf.py", "predict", "--model-dir", directories[name],
                       "--split", "test", "--processed-dir", "data/processed", "--output", "{run_dir}/predictions.jsonl"]
        else:
            inputs.update(TRANSFORMER_CODE_FILES)
            command = ["{python}", "scripts/transformer.py", "predict", "--checkpoint", directories[name],
                       "--calibration", directories[name + "_calibration"], "--split", "test", "--device", "cuda",
                       "--processed-dir", "data/processed", "--manifest", "data/splits/split_manifest_v1.csv",
                       "--split-summary", "data/splits/split_summary_v1.json", "--output", "{run_dir}/predictions.jsonl"]
        job = {"schema_version": 1, "name": "final-predict-" + name.replace("_", "-"),
               "stage": "predict", "split": "test", "seed": seed, "inputs": sorted(inputs), "command": command,
               "expected_outputs": ["predictions.jsonl"], "timeout_seconds": 10800}
        pairs.append(_pair(root, destination, "predict_" + name, job, decision_relative))
    return _write_pairs(root, pairs)


def generate_evaluation(decision_path, predictions, output_dir="configs/final", *, root=ROOT):
    root = Path(root).resolve()
    decision, decision_relative, declared, _, _ = load_decision(decision_path, root=root)
    model_names = _model_names(decision)
    _require(isinstance(predictions, Mapping) and set(predictions) == set(model_names),
             "evaluation needs exactly one prediction file for each frozen model")
    prediction_paths = {name: _relative(root, _path(root, predictions[name])) for name in model_names}
    _require(len(set(prediction_paths.values())) == len(model_names), "model prediction files must be distinct")
    destination = _path(root, output_dir, must_exist=False, directory=True)
    inputs = set(COMMON_FILES) | set(METRIC_FILES) | set(declared) | {decision_relative, TEST_GOLD}
    inputs.update(prediction_paths.values())
    command = ["{python}", "evaluate.py", "--split", "test", "--gold", TEST_GOLD,
               "--manifest", "data/splits/split_manifest_v1.csv", "--split-summary", "data/splits/split_summary_v1.json"]
    for name in model_names:
        command.extend(["--predictions", name + "=" + prediction_paths[name]])
    command.extend(["--output", "{run_dir}/comparison.json", "--report", "{run_dir}/comparison.md",
                    "--csv", "{run_dir}/comparison.csv"])
    seed = decision.get("seed", 42)
    _require(type(seed) is int and seed >= 0, "scientific seed must be a nonnegative integer")
    job = {"schema_version": 1, "name": "final-evaluate", "stage": "evaluate", "split": "test", "seed": seed,
           "inputs": sorted(inputs), "command": command,
           "expected_outputs": ["comparison.json", "comparison.md", "comparison.csv"], "timeout_seconds": 3600}
    return _write_pairs(root, [_pair(root, destination, "final_evaluate", job, decision_relative)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("predict", "evaluate"))
    parser.add_argument("--decision", default="docs/FINAL_PROTOCOL.json")
    parser.add_argument("--output-dir", default="configs/final")
    parser.add_argument("--predictions", action="append", default=[], metavar="NAME=FILE")
    args = parser.parse_args(argv)
    try:
        if args.phase == "predict":
            _require(not args.predictions, "prediction phase does not accept prediction files")
            result = generate_predictions(args.decision, args.output_dir)
        else:
            pairs = []
            for value in args.predictions:
                _require("=" in value, "use NAME=FILE for evaluation predictions")
                pairs.append(value.split("=", 1))
            _require(len({name for name, _ in pairs}) == len(pairs), "duplicate model prediction argument")
            result = generate_evaluation(args.decision, dict(pairs), args.output_dir)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Final job generation failed: {exc}\n")
    print(json.dumps({"generated": result, "next_step": "commit generated jobs/protocols before running final test"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
