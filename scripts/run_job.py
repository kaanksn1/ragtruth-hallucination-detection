"""Sequential owner-run jobs with input/code provenance and retained failure logs.

This is a reproducibility wrapper, not a sandbox for untrusted training code.
It does not implement teammates' models, splits, token alignment or scoring.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "hardware_smoke": {"synthetic"},
    "integration_check": {"synthetic"},
    "pilot": {"train"}, "train": {"train"},
    "predict": {"train", "validation", "test"},
    "evaluate": {"validation", "test"},
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Expected a nonempty repository-relative path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Path leaves the repository.")
    return path


def git(root, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root.as_posix()}", *args],
        cwd=root, stderr=subprocess.PIPE,
    )


def source_state(root):
    return {
        "commit": git(root, "rev-parse", "HEAD").decode().strip(),
        "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
        "working_tree_status": git(root, "status", "--porcelain", "--untracked-files=all").decode().splitlines(),
    }


def environment_snapshot():
    versions = {}
    for package in ("torch", "transformers", "tokenizers", "numpy", "safetensors",
                    "huggingface-hub", "scikit-learn", "sklearn-crfsuite", "python-crfsuite"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    runtime_keys = ("HSA_ENABLE_DXG_DETECTION", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                    "HF_HUB_CACHE", "HF_HUB_DISABLE_XET",
                    "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    return {"python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "packages": versions,
            "runtime_environment": {key: os.environ.get(key) for key in runtime_keys}}


def validate(root, config_path):
    config_path = config_path.resolve()
    if not config_path.is_relative_to(root.resolve()):
        raise ValueError("Job config must be inside this repository.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected job schema_version 1.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", config.get("name", "")):
        raise ValueError("Use a short lowercase job name.")
    if config.get("split") not in STAGES.get(config.get("stage"), set()):
        raise ValueError("Stage/split combination is not allowed.")
    if type(config.get("seed")) is not int:
        raise ValueError("An integer seed is required; the child command must use it.")
    argv = config.get("command")
    if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv):
        raise ValueError("command must be a nonempty argv list; no shell string.")
    if not isinstance(config.get("inputs"), list):
        raise ValueError("inputs must list every data manifest/checkpoint needed by the job.")
    inputs = {name: sha256(contained(root, name)) for name in config["inputs"]}
    if config["stage"] not in {"hardware_smoke", "integration_check"} and not inputs:
        raise ValueError("Model/data jobs must declare their hashed inputs.")
    outputs = config.get("expected_outputs", [])
    if not isinstance(outputs, list):
        raise ValueError("expected_outputs must be a list of paths relative to run_dir.")
    for output in outputs:
        contained(root, output)  # Also reject traversal before a run directory exists.
    timeout = config.get("timeout_seconds", 3600)
    if type(timeout) is not int or not 1 <= timeout <= 86400:
        raise ValueError("timeout_seconds must be an integer from 1 to 86400.")
    state = source_state(root)
    protocol_info = None
    if config["split"] == "test":
        if state["working_tree_status"]:
            raise ValueError("Official test requires a clean committed working tree.")
        protocol = contained(root, config.get("frozen_protocol"))
        relative = protocol.relative_to(root).as_posix()
        committed = git(root, "show", f"HEAD:{relative}")
        if committed != protocol.read_bytes():
            raise ValueError("Official test protocol differs from its committed version.")
        frozen = json.loads(committed)
        if frozen.get("status") != "frozen":
            raise ValueError("Official test requires an explicitly frozen JSON protocol.")
        if frozen.get("test_config_sha256") != sha256(config_path):
            raise ValueError("Frozen protocol does not match this exact test config.")
        if frozen.get("inputs_sha256") != inputs:
            raise ValueError("Frozen protocol does not match the declared input bytes.")
        protocol_info = {"path": relative, "sha256": sha256(protocol),
                         "commit": git(root, "log", "-1", "--format=%H", "--", relative).decode().strip()}
    return config, inputs, state, protocol_info


def require_frozen_test_job(root, invocation):
    """Confirm a test predictor is the exact child command of a validated test job.

    This repeats the existing protocol checks; it is not an untrusted-code sandbox.
    """
    root = root.resolve()
    value = os.environ.get("RAGTRUTH_RUN_RECORD")
    if not value:
        raise ValueError("official test predictions require run_job and a frozen committed protocol")
    record_path = Path(value).resolve()
    if not record_path.is_relative_to(root / "outputs/runs"):
        raise ValueError("test run record must be under outputs/runs")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    config_path = contained(root, record["config_path"])
    config, inputs, state, protocol = validate(root, config_path)
    command = record.get("command", [])
    if (record.get("status") != "running" or config["split"] != "test" or config["stage"] != "predict"
            or protocol is None or record.get("frozen_protocol") != protocol
            or record.get("inputs_sha256") != inputs or record.get("config_sha256") != sha256(config_path)
            or record.get("code", {}).get("commit") != state["commit"]
            or len(command) < 2 or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or (root / command[1]).resolve() != Path(invocation[0]).resolve()
            or command[2:] != invocation[1:]):
        raise ValueError("test prediction invocation differs from the frozen running job")
    return inputs


def run_job(config_path, *, root=ROOT):
    root = root.resolve()
    config, inputs, state, protocol = validate(root, config_path)
    output_root = root / "outputs"
    output_root.mkdir(exist_ok=True)
    lock = output_root / ".run.lock"
    # Exclusive creation prevents two owners' wrappers sharing the GPU at once.
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}, handle)
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = output_root / "runs" / f"{stamp}-{config['name']}"
        run_dir.mkdir(parents=True)
        replacements = {"{python}": sys.executable, "{run_dir}": str(run_dir), "{seed}": str(config["seed"])}
        command = []
        for argument in config["command"]:
            for key, value in replacements.items():
                argument = argument.replace(key, value)
            command.append(argument)
        run = {
            "schema_version": 1, "status": "running", "name": config["name"],
            "stage": config["stage"], "split": config["split"], "seed": config["seed"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": config_path.resolve().relative_to(root).as_posix(),
            "config_sha256": sha256(config_path), "config": config,
            "command": command, "code": state, "inputs_sha256": inputs,
            "frozen_protocol": protocol, "environment": environment_snapshot(),
        }
        record = run_dir / "run.json"

        def save():
            record.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

        save()
        started = time.perf_counter()
        exit_code = 1
        try:
            with (run_dir / "console.log").open("w", encoding="utf-8") as log:
                child = subprocess.run(command, cwd=root,
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "RAGTRUTH_RUN_RECORD": str(record)},
                    stdout=log, stderr=subprocess.STDOUT, timeout=config.get("timeout_seconds", 3600))
            exit_code = child.returncode
            run["outputs_sha256"] = {}
            for name in config.get("expected_outputs", []):
                path = contained(run_dir, name)
                if not path.is_file():
                    raise FileNotFoundError(f"Missing expected output: {name}")
                run["outputs_sha256"][name] = sha256(path)
            run["status"] = "completed" if exit_code == 0 else "failed"
        except KeyboardInterrupt:
            run.update(status="interrupted", error="Interrupted by operator")
            exit_code = 130
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            exit_code = exit_code or 1
        finally:
            run.update(exit_code=exit_code, elapsed_seconds=time.perf_counter() - started,
                       finished_utc=datetime.now(timezone.utc).isoformat())
            save()
        print(json.dumps({"status": run["status"], "exit_code": exit_code, "record": str(record)}))
        return exit_code, record
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path, nargs="+", help="Run these jobs sequentially; stop on first failure.")
    parser.add_argument("--check-only", action="store_true", help="Validate config, input hashes and test gate only.")
    args = parser.parse_args()
    try:
        for path in args.configs:
            if args.check_only:
                config, inputs, state, _ = validate(ROOT, path)
                print(json.dumps({"name": config["name"], "status": "ready", "inputs_sha256": inputs,
                                  "code_commit": state["commit"]}))
            else:
                code, _ = run_job(path)
                if code:
                    raise SystemExit(code)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Run preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
