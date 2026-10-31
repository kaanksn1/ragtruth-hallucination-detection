"""Local Windows supervisor for the bounded weighted trial and technical finish.

No network/push/presentation work. Official test is disabled unless --final-test
is explicit; scientific decisions and each machine protocol are committed first.
Existing weights, outputs, scientific decisions and generated jobs are not replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import METRIC_FILES
from ragtruth_data.data import sha256_file
from ragtruth_data.schema import LABEL_TYPES
from ragtruth_eval import METRIC_VERSION
from scripts import generate_final_jobs as final_jobs
from scripts.run_job import git, source_state

BASE_INPUTS = ("requirements-transformer.txt", "requirements-crf.txt", "transformer.py", "scripts/transformer.py",
    "scripts/run_job.py", "scripts/calibrate_transformer.py", "scripts/baseline_crf.py", "ragtruth_data/data.py",
    "ragtruth_data/features.py", "ragtruth_data/alignment.py", "ragtruth_data/schema.py",
    "data/splits/split_manifest_v1.csv", "data/splits/split_summary_v1.json",
    "data/processed/train.jsonl", "data/processed/validation.jsonl")
DATA_FILES = ("data/raw/response.jsonl", "data/raw/source_info.jsonl", "data/raw/LICENSE",
    "data/processed/train.jsonl", "data/processed/crf_train.jsonl", "data/processed/validation.jsonl",
    "data/processed/crf_validation.jsonl", "data/splits/split_manifest_v1.csv", "data/splits/split_summary_v1.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def windows_to_wsl(path):
    """Pure conversion; argv callers need no shell quoting, even with spaces/Unicode."""
    path = PureWindowsPath(path)
    require(path.is_absolute() and len(path.drive) == 2 and path.drive[1] == ":"
            and path.drive[0].isalpha(), "expected an absolute Windows drive path")
    require(".." not in path.parts, "traversing Windows path")
    return "/mnt/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])


def wsl_record_to_windows(value, windows_root):
    root = PureWindowsPath(windows_root)
    path = PurePosixPath(value)
    require(".." not in path.parts, "traversing WSL record path")
    try:
        relative = path.relative_to(PurePosixPath(windows_to_wsl(root)))
    except ValueError as exc:
        raise ValueError("WSL record is outside the repository") from exc
    require(relative.parts[:2] == ("outputs", "runs") and len(relative.parts) == 4 and relative.name == "run.json",
            "job record must be under outputs/runs")
    return root.joinpath(*relative.parts)


def path_inside(root, value, *, exists=True):
    value = str(value).replace("\\", "/")
    require(value and not PureWindowsPath(value).drive and not value.startswith("/")
            and ".." not in PurePosixPath(value).parts, "expected a contained repository-relative path")
    path = (root / value).resolve()
    require(path != root and path.is_relative_to(root), "path leaves repository")
    require(not exists or path.exists(), "missing input: " + value)
    return path


def relative(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root), "path leaves repository")
    return path.relative_to(root).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_state(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def ensure_clean(root):
    require(not source_state(root)["working_tree_status"], "pipeline requires a clean committed repository")


def unreadable_cache_hashes(root, names):
    """Hash NTFS WSL symlinks from WSL; native Windows cannot open these aliases."""
    require(all(name.startswith(".cache/hf-hub/") for name in names), "WSL fallback is restricted to HF cache")
    program = ("import hashlib,json,sys;from pathlib import Path;root=Path.cwd().resolve();result={};"
               "\nfor name in json.loads(sys.argv[1]):"
               "\n p=(root/name).resolve();assert p.is_relative_to(root),'cache path leaves repository'"
               "\n with p.open('rb') as handle: result[name]=hashlib.file_digest(handle,'sha256').hexdigest()"
               "\nprint(json.dumps(result))")
    command = ["wsl.exe", "--distribution", "Ubuntu-24.04", "--user", "serda", "--cd", windows_to_wsl(str(root)),
               "--", "/opt/banking77-venv/bin/python", "-c", program, json.dumps(names)]
    child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    result = json.loads(child.stdout.strip().splitlines()[-1])
    require(isinstance(result, dict) and set(result) == set(names), "incomplete WSL cache hash result")
    return result


def verify_run_inputs(record, *, root=ROOT, completed=True):
    """Verify immutable declared inputs; do not parse data/model artifacts."""
    root = Path(root).resolve()
    require(record.get("status") in {"running", "completed", "failed", "interrupted"}, "invalid job status")
    if completed:
        require(record["status"] == "completed" and record.get("exit_code") == 0, "job did not complete successfully")
    require(isinstance(record.get("inputs_sha256"), dict) and record["inputs_sha256"], "job lacks input hashes")
    unreadable = []
    for name, digest in record["inputs_sha256"].items():
        try:
            actual = sha256_file(path_inside(root, name, exists=False))
        except OSError:
            if not name.startswith(".cache/hf-hub/"):
                raise
            unreadable.append(name)
            continue
        require(actual == digest, "declared input changed: " + name)
    if unreadable:
        hashes = unreadable_cache_hashes(root, unreadable)
        for name in unreadable:
            require(hashes[name] == record["inputs_sha256"][name], "declared input changed: " + name)
    return record


def verify_run_outputs(run_dir, record=None):
    """Historical runs retain old code input hashes; only their saved outputs are checked."""
    run_dir = Path(run_dir).resolve()
    record = record or read_json(run_dir / "run.json")
    require(record.get("status") == "completed" and record.get("exit_code") == 0, "job did not complete successfully")
    require(isinstance(record.get("outputs_sha256"), dict) and record["outputs_sha256"], "job lacks output hashes")
    for name, digest in record["outputs_sha256"].items():
        require(sha256_file(path_inside(run_dir, name)) == digest, "job output checksum mismatch: " + name)
    return record


def verify_weighted_summary(summary):
    config, stats = summary.get("config", {}), summary.get("positive_weighting", {})
    epochs = summary.get("epochs", [])
    require(summary.get("status") == "complete" and config.get("epochs") == 3 and len(epochs) == 3,
            "weighted trial did not complete three epochs")
    require(summary.get("counts") == {"train": 12702, "validation": 2240}
            and summary.get("limits") == {"train": None, "validation": None}, "weighted trial was incomplete or limited")
    require(config.get("positive_weighting") == stats.get("method") == "sqrt_capped" and stats.get("split") == "train",
            "positive weights were not derived solely from train")
    require(summary.get("label_types") == stats.get("label_types") == list(LABEL_TYPES), "weighted label order mismatch")
    vector = config.get("positive_weights")
    require(isinstance(vector, list) and len(vector) == len(LABEL_TYPES) and vector == stats.get("weights_vector")
            and all(type(v) in (int, float) and math.isfinite(v) and 1 <= v <= config.get("positive_weighting_cap", 50) for v in vector),
            "train-derived weight vector differs from model config")
    require(stats.get("records") == 12702 and type(stats.get("response_tokens")) is int and stats["response_tokens"] > 0,
            "invalid train-derived weight support")
    require(all(epoch.get("train", {}).get("answer_tokens") == stats["response_tokens"] for epoch in epochs),
            "training target counts differ from weight-count pass")
    return summary


def saved_improving_checkpoints(run_dir, summary=None):
    """Only saved strict validation-loss improvements, in declared epoch order."""
    run_dir = Path(run_dir)
    summary = summary or read_json(run_dir / "transformer_summary.json")
    epochs = summary.get("epochs")
    require(summary.get("status") == "complete" and isinstance(epochs, list) and 1 <= len(epochs) <= 3,
            "expected one to three completed weighted epochs")
    best, selected = float("inf"), []
    for index, row in enumerate(epochs, 1):
        loss = row.get("validation", {}).get("loss")
        require(row.get("epoch") == index and type(loss) in (int, float) and math.isfinite(loss) and loss >= 0,
                "invalid weighted epoch/loss history")
        if loss < best:
            checkpoint = run_dir / "checkpoints" / f"epoch-{index}"
            require((checkpoint / "checkpoint.json").is_file(), "missing saved improving checkpoint")
            selected.append(checkpoint)
            best = loss
    require(selected, "no saved improving checkpoint candidates")
    return selected


def choose_experiment(base, weighted):
    require(base.get("label_types") == weighted.get("label_types") == list(LABEL_TYPES), "calibration label order mismatch")
    require(base.get("data_sha256") and base["data_sha256"] == weighted.get("data_sha256"), "calibration data identity mismatch")
    def key(artifact):
        require(artifact.get("status") == "selected_on_validation" and artifact.get("validation_split") == "validation",
                "experiment decoder was not selected on validation")
        metrics = artifact.get("validation_metrics", {})
        values = (metrics.get("macro_f1_types"), metrics.get("any", {}).get("f1"))
        require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values),
                "invalid validation selection metrics")
        return values
    return "weighted" if key(weighted) > key(base) else "base"


def parse_job_record(stdout, *, root=ROOT):
    root = Path(root).resolve()
    for line in reversed(stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "record" in result:
            require(result.get("status") == "completed" and result.get("exit_code") == 0, "central job reported failure")
            value = result["record"]
            require(isinstance(value, str), "invalid job record path")
            path = Path(str(wsl_record_to_windows(value, str(root)))) if value.startswith("/mnt/") else Path(value)
            path = path.resolve()
            require(path.is_relative_to(root / "outputs/runs") and path.name == "run.json", "unsafe job record path")
            return path
    raise ValueError("central job did not return a successful record")


def commit_new_paths(root, paths, message):
    names = [relative(root, path) for path in paths]
    require(len(names) == len(set(names)) and all((root / name).is_file() for name in names), "invalid commit path list")
    require(not git(root, "diff", "--cached", "--name-only").strip(), "unrelated staged changes before scoped commit")
    require(not git(root, "diff", "--name-only").strip(), "tracked changes before scoped commit")
    untracked = set(git(root, "ls-files", "--others", "--exclude-standard").decode("utf-8").splitlines())
    require(untracked == set(names), "unrelated/unexpected untracked files before scoped commit")
    git(root, "add", "--", *names)
    git(root, "commit", "-m", message)
    ensure_clean(root)


class Pipeline:
    def __init__(self, args, *, root=ROOT):
        self.root = Path(root).resolve()
        self.args = args
        self.output = path_inside(self.root, args.output, exists=False)
        require(self.output.is_relative_to(self.root / "outputs"), "pipeline output must be under outputs/")
        self.state_path = self.output.with_suffix(".status.json")
        self.work = self.output.parent / (self.output.stem + "_work")
        require(not self.output.exists() and not self.state_path.exists() and not self.work.exists(),
                "refusing to overwrite an existing pipeline")
        ensure_clean(self.root)
        self.work.mkdir(parents=True)
        self.cpu = self.root / ".venv/Scripts/python.exe"
        require(self.cpu.is_file(), "dedicated Windows P2 interpreter missing")
        self.result = {"schema_version": 1, "status": "running", "started_utc": utc(), "stages": {},
                       "official_test_enabled": args.final_test, "official_test_evaluated": False}

    def progress(self, phase, **details):
        self.result.update(phase=phase, updated_utc=utc())
        self.result["stages"][phase] = details
        write_state(self.state_path, self.result)
        print(json.dumps({"phase": phase, **details}), flush=True)

    def local(self, command, name):
        self.progress(name)
        process = subprocess.run(command, cwd=self.root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (self.work / (name + ".log")).write_text(process.stdout + process.stderr, encoding="utf-8")
        require(process.returncode == 0, name + " failed; see its pipeline log")
        return process.stdout

    def job(self, config, name, gpu=False):
        config_name = relative(self.root, config)
        while (self.root / "outputs/.run.lock").exists():
            self.progress(name + "_waiting_for_queue")
            time.sleep(5)
        if gpu:
            command = ["wsl.exe", "--distribution", "Ubuntu-24.04", "--user", "serda", "--cd", windows_to_wsl(str(self.root)),
                "--", "env", "HSA_ENABLE_DXG_DETECTION=1", "HF_HUB_CACHE=.cache/hf-hub", "HF_HUB_OFFLINE=1",
                "TRANSFORMERS_OFFLINE=1", "/opt/banking77-venv/bin/python", "scripts/run_job.py", config_name]
        else:
            command = [str(self.cpu), "scripts/run_job.py", config_name]
        stdout = self.local(command, name)
        record_path = parse_job_record(stdout, root=self.root)
        record = verify_run_inputs(read_json(record_path), root=self.root)
        verify_run_outputs(record_path.parent, record)
        self.result["stages"][name]["record"] = relative(self.root, record_path)
        return record_path.parent

    def config(self, name, command, inputs, outputs, *, stage="predict", split="validation"):
        path = self.work / (name + ".json")
        write_new(path, {"schema_version": 1, "name": name.replace("_", "-"), "stage": stage, "split": split,
                        "seed": 42, "inputs": sorted(set(inputs)), "command": command,
                        "expected_outputs": outputs, "timeout_seconds": 10800})
        return path

    def files(self, directory):
        files = []
        for path in sorted(directory.rglob("*")):
            require(path.resolve().is_relative_to(self.root), "artifact leaves repository")
            if path.is_file():
                files.append(relative(self.root, path))
        return files

    def wait_for_weighted(self, run_dir):
        last_inputs_verified = False
        while True:
            try:
                record = read_json(run_dir / "run.json")
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            require(record.get("name") == "transformer-weighted-train", "not the declared weighted trial")
            if not last_inputs_verified:
                verify_run_inputs(record, root=self.root, completed=False)
                last_inputs_verified = True
            if record["status"] != "running":
                verify_run_inputs(record, root=self.root)
                verify_run_outputs(run_dir, record)
                return
            self.progress("waiting_for_weighted_training", run=relative(self.root, run_dir))
            time.sleep(30)

    def calibration(self, weighted_run):
        policy_name = "configs/calibration_weighted_policy.json"
        policy_path = self.root / policy_name
        require(policy_path.read_bytes() == git(self.root, "show", "HEAD:" + policy_name), "weighted calibration policy is not committed")
        policy = read_json(policy_path)
        summary = verify_weighted_summary(read_json(weighted_run / "transformer_summary.json"))
        require(summary["positive_weighting"] == read_json(weighted_run / "positive_weighting.json"),
                "training weight statistics file differs from summary")
        candidates = saved_improving_checkpoints(weighted_run, summary)
        require(len(candidates) <= policy.get("max_checkpoint_candidates", 0) == 3, "checkpoint policy mismatch")
        policy["checkpoints"] = [relative(self.root, path) for path in candidates]
        config = self.work / "weighted_calibration_config.json"
        write_new(config, policy)
        inputs = [*BASE_INPUTS, policy_name, policy["training_config"], relative(self.root, config)]
        inputs += [filename for path in candidates for filename in self.files(path)]
        outputs = ["calibration.json", "calibration_report.json", "validation_predictions.jsonl"]
        for index in range(1, len(candidates) + 1):
            outputs += [f"validation_predictions_candidate_{index}.jsonl", f"validation_predictions_default_{index}.jsonl",
                        f"probabilities/candidate-{index}.jsonl.gz", f"probabilities/candidate-{index}.jsonl.gz.meta.json"]
        job = self.config("weighted-calibration", ["{python}", "scripts/calibrate_transformer.py", "--config",
                          relative(self.root, config), "--split", "validation", "--device", "cuda", "--output", "{run_dir}"], inputs, outputs)
        return self.job(job, "weighted_calibration", gpu=True)

    def validation(self, selected_dir, base_calibration, weighted_calibration, base_train, crf_run, winner):
        calibration = read_json(selected_dir / "calibration.json")
        checkpoint = path_inside(self.root, calibration["checkpoint"])
        calibration_name = relative(self.root, selected_dir / "calibration.json")
        inputs = [*BASE_INPUTS, *self.files(checkpoint), calibration_name]
        config = self.config("selected-validation-replay", ["{python}", "scripts/transformer.py", "predict", "--checkpoint",
            relative(self.root, checkpoint), "--calibration", calibration_name, "--split", "validation", "--device", "cuda",
            "--output", "{run_dir}/predictions.jsonl"], inputs, ["predictions.jsonl"])
        replay = self.job(config, "selected_validation_replay", gpu=True)
        require((replay / "predictions.jsonl").read_bytes() == (selected_dir / "validation_predictions.jsonl").read_bytes(),
                "selected validation replay is not byte-identical to calibrated cache predictions")
        predictions = {"crf_source": crf_run / "validation_predictions.jsonl",
            "crf_response_only": crf_run / "validation_predictions_response_only.jsonl",
            "transformer_default": base_train / "validation_predictions.jsonl", "transformer": replay / "predictions.jsonl"}
        predictions["transformer_reference" if winner == "weighted" else "weighted_trial"] = (
            base_calibration if winner == "weighted" else weighted_calibration) / "validation_predictions.jsonl"
        command = ["{python}", "evaluate.py", "--split", "validation"]
        for name, path in predictions.items():
            command += ["--predictions", name + "=" + relative(self.root, path)]
        command += ["--output", "{run_dir}/comparison.json", "--report", "{run_dir}/comparison.md", "--csv", "{run_dir}/comparison.csv"]
        config = self.config("technical-validation-comparison", command,
            [*METRIC_FILES, *BASE_INPUTS, *[relative(self.root, path) for path in predictions.values()]],
            ["comparison.json", "comparison.md", "comparison.csv"], stage="evaluate")
        return self.job(config, "validation_comparison")

    def final_test(self, selected_dir, base_calibration, crf_run, winner, selection, validation_run):
        self.progress("freezing_scientific_decision")
        ensure_clean(self.root)
        decision_path = self.root / "docs/FINAL_PROTOCOL.json"
        require(not decision_path.exists(), "scientific final decision already exists; refusing another final test")
        selected = read_json(selected_dir / "calibration.json")
        models = {"crf_source": {"model_dir": relative(self.root, crf_run / "models/source")},
                  "crf_response_only": {"model_dir": relative(self.root, crf_run / "models/response_only")},
                  "transformer": {"checkpoint": selected["checkpoint"], "calibration": relative(self.root, selected_dir / "calibration.json")}}
        if winner == "weighted":
            base = read_json(base_calibration / "calibration.json")
            models["transformer_reference"] = {"checkpoint": base["checkpoint"],
                "calibration": relative(self.root, base_calibration / "calibration.json")}
        artifacts = set()
        for model in models.values():
            artifacts.update(self.files(path_inside(self.root, model.get("model_dir", model.get("checkpoint")))))
            if "calibration" in model:
                artifacts.add(model["calibration"])
        tracked = git(self.root, "ls-files").decode("utf-8").splitlines()
        code = [name for name in tracked if not name.startswith("configs/final/") and
                (name.endswith(".py") or name.startswith("configs/") and name.endswith(".json")
                 or Path(name).name.startswith("requirements") and name.endswith(".txt"))]
        decision = {"schema_version": 1, "status": "frozen", "seed": 42, "frozen_utc": utc(), "models": models,
            "selection": selection, "validation_comparison": relative(self.root, validation_run / "comparison.json"),
            "validation_comparison_sha256": sha256_file(validation_run / "comparison.json"),
            "metric_policy": {"version": METRIC_VERSION, "character_whitespace": "included", "exact_spans": "raw deduplicated boundaries",
                              "quality_filter": "good", "implicit_true": "retain", "due_to_null": "retain"},
            "test_policy": "one fixed prediction per declared model; no selection or changes from official-test results",
            "artifacts_sha256": {name: sha256_file(self.root / name) for name in sorted(artifacts)},
            "code_sha256": {name: sha256_file(self.root / name) for name in sorted(code)},
            "data_sha256": {name: sha256_file(self.root / name) for name in DATA_FILES}}
        write_new(decision_path, decision)
        commit_new_paths(self.root, [decision_path], "Freeze validation-selected scientific final-test decisions")
        final_jobs.load_decision("docs/FINAL_PROTOCOL.json", root=self.root)
        self.local([str(self.cpu), "scripts/prepare_data.py", "--include-test"], "prepare_frozen_test")
        final_jobs.load_decision("docs/FINAL_PROTOCOL.json", root=self.root)
        ensure_clean(self.root)
        pairs = final_jobs.generate_predictions("docs/FINAL_PROTOCOL.json", root=self.root)
        commit_new_paths(self.root, [self.root / pair[key] for pair in pairs for key in ("job", "protocol")],
                         "Freeze final prediction job byte identities")
        prediction_paths = {}
        for pair in pairs:
            name = Path(pair["job"]).stem.removeprefix("predict_")
            run = self.job(self.root / pair["job"], "final_" + name, gpu=name.startswith("transformer"))
            prediction_paths[name] = relative(self.root, run / "predictions.jsonl")
        pairs = final_jobs.generate_evaluation("docs/FINAL_PROTOCOL.json", prediction_paths, root=self.root)
        commit_new_paths(self.root, [self.root / pair[key] for pair in pairs for key in ("job", "protocol")],
                         "Freeze final evaluation job and prediction hashes")
        final_run = self.job(self.root / pairs[0]["job"], "final_evaluation")
        self.result.update(official_test_evaluated=True, final_comparison=relative(self.root, final_run / "comparison.json"),
                           scientific_decision="docs/FINAL_PROTOCOL.json", test_predictions=prediction_paths)

    def run(self):
        weighted = path_inside(self.root, self.args.weighted_run)
        base_calibration = path_inside(self.root, self.args.base_calibration)
        base_train = path_inside(self.root, self.args.base_train)
        crf_run = path_inside(self.root, self.args.crf_run)
        for historical_run in (base_calibration, base_train, crf_run):
            verify_run_outputs(historical_run)
        self.wait_for_weighted(weighted)
        weighted_calibration = self.calibration(weighted)
        base, weighted_artifact = read_json(base_calibration / "calibration.json"), read_json(weighted_calibration / "calibration.json")
        require(path_inside(self.root, base["checkpoint"]).is_relative_to(base_train / "checkpoints"), "wrong base checkpoint run")
        require(path_inside(self.root, weighted_artifact["checkpoint"]).is_relative_to(weighted / "checkpoints"), "wrong weighted checkpoint run")
        winner = choose_experiment(base, weighted_artifact)
        selected_dir = weighted_calibration if winner == "weighted" else base_calibration
        selection = {"winner": winner, "rule": "higher validation macro F1, then any F1; exact tie prefers base",
            "base_metrics": base["validation_metrics"], "weighted_metrics": weighted_artifact["validation_metrics"],
            "base_calibration": relative(self.root, base_calibration / "calibration.json"),
            "weighted_calibration": relative(self.root, weighted_calibration / "calibration.json"),
            "selected_calibration": relative(self.root, selected_dir / "calibration.json"), "official_test_used": False}
        write_new(self.work / "selection.json", selection)
        self.progress("selected_on_validation", winner=winner)
        validation_run = self.validation(selected_dir, base_calibration, weighted_calibration, base_train, crf_run, winner)
        self.result.update(selection=selection, weighted_calibration=relative(self.root, weighted_calibration),
                           validation_comparison=relative(self.root, validation_run / "comparison.json"))
        if self.args.final_test:
            self.final_test(selected_dir, base_calibration, crf_run, winner, selection, validation_run)
        self.result.update(status="completed", finished_utc=utc())
        write_new(self.output, self.result)
        self.progress("completed", output=relative(self.root, self.output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weighted-run", default="outputs/runs/20261031T142051826651Z-transformer-weighted-train")
    parser.add_argument("--base-calibration", default="outputs/runs/20261031T140016156258Z-transformer-calibration")
    parser.add_argument("--base-train", default="outputs/runs/20261031T110122793234Z-transformer-train")
    parser.add_argument("--crf-run", default="outputs/runs/20261031T133357203274Z-crf-validation")
    parser.add_argument("--output", default="outputs/technical_completion.json")
    parser.add_argument("--final-test", action="store_true", help="Commit scientific/machine protocols then run the already authorized final test once")
    args = parser.parse_args(argv)
    pipeline = None
    try:
        pipeline = Pipeline(args)
        pipeline.run()
    except (Exception, KeyboardInterrupt) as exc:
        if pipeline is not None:
            pipeline.result.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_utc=utc())
            if not pipeline.output.exists():
                write_new(pipeline.output, pipeline.result)
            write_state(pipeline.state_path, pipeline.result)
        parser.exit(1, f"Technical completion stopped: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
