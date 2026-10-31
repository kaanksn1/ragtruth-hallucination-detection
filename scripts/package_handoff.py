"""Build a local, checked technical handoff ZIP; no weights, secrets or uploads."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/kaanksn1/ragtruth-hallucination-detection"
ROOT_FILES = ("README.md", "evaluate.py", "plot_results.py", "transformer.py",
              "requirements-crf.txt", "requirements-transformer.txt", "requirements-evaluation.txt",
              "demo/viewer.html", "demo/viewer.css", "demo/viewer.js",
              "data/splits/split_manifest_v1.csv", "data/splits/split_summary_v1.json")
TREE_TYPES = {"scripts": {".py"}, "ragtruth_data": {".py"}, "ragtruth_eval": {".py"},
              "tests": {".py"}, "configs": {".json"}, "docs": {".md", ".json", ".txt"}}
RESULT_TYPES = {".md", ".json", ".csv", ".png", ".svg"}
FORBIDDEN_PARTS = {"raw", "processed", "__pycache__", "contributions", "slides", "presentations", "checkpoints", "models", "weights"}
FORBIDDEN_FILES = {"weights.json", "model_weights.json"}
DENIED_SUFFIXES = {".safetensors", ".crfsuite", ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle",
                   ".joblib", ".npy", ".npz", ".ppt", ".pptx", ".zip", ".pem", ".key", ".p12", ".pfx"}
SECRET_NAME = re.compile(r"(?:secret|credential|password|private[_-]?key|api[_-]?key|access[_-]?token|contribution|personal[_-]?report|individual[_-]?report)", re.I)
SECRET_CONTENT = re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{30,}|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
MAX_SAMPLE_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_BUNDLE_BYTES = 256 * 1024 * 1024


def contained(root, path):
    root, path = root.resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("handoff path leaves the repository or selects its root")
    return path


def archive_name(name):
    value = PurePosixPath(name)
    if value.is_absolute() or "\\" in name or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError("invalid archive member name")
    return value.as_posix()


def allowed_path(path, root):
    relative = path.relative_to(root)
    return path.suffix.lower() not in DENIED_SUFFIXES and path.name.lower() not in FORBIDDEN_FILES and not any(
        part.startswith(".") or part.lower() in FORBIDDEN_PARTS or SECRET_NAME.search(part) for part in relative.parts)


def read_payload(root, path):
    path = contained(root, path)
    if not allowed_path(path, root):
        raise ValueError("allowlisted input resolves to an excluded file")
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("handoff input is missing, not a file, or too large")
    payload = path.read_bytes()
    if SECRET_CONTENT.search(payload):
        raise ValueError("potential embedded credential in an allowlisted handoff file")
    return payload


def readme():
    return ("# RAGTruth technical handoff\n\n"
        "Open demo.html in a browser. It displays saved VALIDATION examples and predictions; "
        "it does not perform live model inference or guarantee factual correctness.\n\n"
        "results/ contains fixed reports and plots. When final-test results are present, "
        "they are aggregate results of the frozen evaluation; test examples are not used in the demo. "
        "Model settings must not be selected again from those test results.\n\n"
        "Source, tests, configs and documentation are included for reproducibility. "
        "PROJECT_README.md and docs/RUNNING.md explain setup. Run `python -m unittest discover -s tests -v` "
        "after installing the documented dependencies.\n\n"
        "Original raw/processed data, model weights, environments, Git metadata, credentials, "
        "personal contribution reports and presentation slides are not bundled. "
        "Pinned data/model downloads are documented in the project configs. "
        "Small public dataset examples, if included, are covered by docs/DATA_LICENSE.txt.\n\n"
        f"Repository: {REPOSITORY}\n\n"
        "MANIFEST.json records the SHA-256 and byte size of every other archive member.\n").encode("utf-8")


def collect_payloads(root, demo, results_dir):
    root = root.resolve()
    demo, results_dir = contained(root, demo), contained(root, results_dir)
    if demo.suffix.lower() != ".html" or not results_dir.is_dir():
        raise ValueError("provide one saved HTML demo and a results directory")
    payloads, omitted = {"README.md": readme()}, Counter()

    def add(path, name):
        name = archive_name(name)
        if not allowed_path(path, root):
            omitted["excluded_path"] += 1
            return
        if name.casefold() in {member.casefold() for member in payloads}:
            raise ValueError("duplicate or case-colliding archive member")
        payloads[name] = read_payload(root, path)
        if sum(map(len, payloads.values())) > MAX_BUNDLE_BYTES:
            raise ValueError("technical handoff exceeds its bundle size limit")

    add(demo, "demo.html")
    for name in ROOT_FILES:
        path = root / name
        if path.exists():
            add(path, "PROJECT_README.md" if name == "README.md" else name)
    for directory, extensions in TREE_TYPES.items():
        base = root / directory
        if base.exists():
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix.lower() in extensions:
                    add(path, path.relative_to(root).as_posix())
    results_count = 0
    for path in sorted(results_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in RESULT_TYPES and allowed_path(path, root):
            add(path, "results/" + path.relative_to(results_dir).as_posix())
            results_count += 1
    if not results_count:
        raise ValueError("no supported result files were found")
    sample_bytes = 0
    samples = root / "data/samples"
    if samples.exists():
        for path in sorted(samples.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jsonl", ".json"} and allowed_path(path, root):
                canonical = contained(root, path)
                if sample_bytes + canonical.stat().st_size > MAX_SAMPLE_BYTES:
                    omitted["sample_size_limit"] += 1
                    continue
                add(path, path.relative_to(root).as_posix())
                sample_bytes += canonical.stat().st_size
    if "docs/DATA_LICENSE.txt" not in payloads:
        raise ValueError("docs/DATA_LICENSE.txt is required for the dataset examples")
    return payloads, dict(omitted)


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC validation failed")
        names = archive.namelist()
        if len(names) != len(set(name.casefold() for name in names)):
            raise ValueError("duplicate ZIP member")
        for name in names:
            archive_name(name)
        manifest = json.loads(archive.read("MANIFEST.json"))
        files = manifest.get("files", {})
        if set(files) | {"MANIFEST.json"} != set(names) or "MANIFEST.json" in files:
            raise ValueError("ZIP members differ from the checksum manifest")
        for name, expected in files.items():
            payload = archive.read(name)
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError("ZIP payload differs from its checksum manifest")
    return manifest


def build_package(root, demo, results_dir, output):
    root = root.resolve()
    output = contained(root, output)
    if output.suffix.lower() != ".zip" or not any(output.is_relative_to(root / base) for base in ("artifacts", "outputs")):
        raise ValueError("ZIP output must be underneath ignored artifacts/ or outputs/")
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing handoff ZIP")
    payloads, omitted = collect_payloads(root, demo, results_dir)
    manifest = {"schema_version": 1, "kind": "technical_handoff_without_slides",
        "created_utc": datetime.now(timezone.utc).isoformat(), "repository": REPOSITORY,
        "files": {name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                  for name, payload in sorted(payloads.items())}, "omitted_counts": omitted}
    payloads["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="handoff-", dir=output.parent) as directory:
        temporary = Path(directory) / "verified.zip"
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in sorted(payloads.items()):
                archive.writestr(name, payload)
        verify_archive(temporary)
        # Exclusive creation also guards against a second packager racing us.
        with output.open("xb") as destination, temporary.open("rb") as source:
            shutil.copyfileobj(source, destination)
    verify_archive(output)
    return {"output": str(output), "files": len(manifest["files"]) + 1,
            "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/ragtruth_technical_handoff.zip"))
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build_package(ROOT, args.demo, args.results_dir, args.output)))
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Technical handoff failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
