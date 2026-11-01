"""Build the audited V2 archive, retaining repository-relative report paths.

The frozen V1 packager remains byte-identical. This postprocessing wrapper adds
V2 runtime files and keeps results/v2 rather than flattening it into results/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import package_handoff as base

REQUIRED = ("structured_transformer.py", "AGENTS.md",
            "data/splits/split_manifest_v2.csv", "data/splits/split_summary_v2.json",
            "docs/V2_FINAL_PROTOCOL.json", "docs/V2_REPRODUCIBILITY.md",
            "results/v2/FINAL_REPORT.md", "results/v2/AUDIT_TR.md")


def v2_readme():
    return ("# RAGTruth V2 audited technical handoff\n\n"
        "Open demo.html for saved VALIDATION examples; this is not live inference.\n\n"
        "Start with [the audit](results/v2/AUDIT_TR.md) and "
        "[the V2 report](results/v2/FINAL_REPORT.md). "
        "[Reproducibility and limitations](docs/V2_REPRODUCIBILITY.md) explain dependencies, "
        "pinned downloads, owner-specific paths, and the files excluded from this archive.\n\n"
        "V2 uses a new INTERNAL holdout carved from original training sources. "
        "It is not the original official test; their scores cannot be compared directly. "
        "Model selection remains frozen. Original V1 reports in results/ are historical.\n\n"
        "Both root model modules and V2 split contracts are included at their actual runtime paths. "
        "Raw/processed data, model weights, environments, Git metadata, secrets and slides are excluded. "
        "Full training/evaluation replay needs separately acquired data and weights and the Git repository. "
        "The standalone demo needs none of those. Public examples are covered by docs/DATA_LICENSE.txt.\n\n"
        "MANIFEST.json records each other member's SHA-256 and byte size.\n").encode("utf-8")


def build_package(root, demo, output):
    root = root.resolve()
    output = base.contained(root, output)
    if output.suffix.lower() != ".zip" or not any(output.is_relative_to(root / p) for p in ("artifacts", "outputs")):
        raise ValueError("ZIP output must be underneath artifacts/ or outputs/")
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing handoff ZIP")
    # Passing the full results tree preserves results/v2/... and V1 historical links.
    payloads, omitted = base.collect_payloads(root, demo, root / "results")
    payloads["README.md"] = v2_readme()
    for name in REQUIRED:
        payload = base.read_payload(root, root / name)
        if name in payloads and payloads[name] != payload:
            raise ValueError("inconsistent duplicate archive input")
        payloads[name] = payload
    protocol = json.loads(payloads["docs/V2_FINAL_PROTOCOL.json"])
    if (protocol.get("status") != "frozen" or protocol.get("original_official_test_read") is not False
            or protocol.get("study_id") != "ragtruth-v2-fresh-original-train-holdout"
            or not isinstance(protocol.get("inputs_sha256"), dict) or not protocol["inputs_sha256"]):
        raise ValueError("expected a frozen V2 protocol excluding the original official test")
    for name, expected in protocol.get("inputs_sha256", {}).items():
        member = "PROJECT_README.md" if name == "README.md" else name
        if member in payloads and hashlib.sha256(payloads[member]).hexdigest() != expected:
            raise ValueError("bundled input differs from frozen V2 identity: " + name)
    if sum(map(len, payloads.values())) > base.MAX_BUNDLE_BYTES:
        raise ValueError("technical handoff exceeds its bundle size limit")
    manifest = {"schema_version": 1, "kind": "audited_v2_handoff_without_slides",
                "created_utc": datetime.now(timezone.utc).isoformat(), "repository": base.REPOSITORY,
                "files": {name: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
                          for name, payload in sorted(payloads.items())}, "omitted_counts": omitted}
    payloads["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v2-handoff-", dir=output.parent) as directory:
        temporary = Path(directory) / "verified.zip"
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, payload in sorted(payloads.items()):
                archive.writestr(name, payload)
        base.verify_archive(temporary)
        with output.open("xb") as destination, temporary.open("rb") as source:
            shutil.copyfileobj(source, destination)
    base.verify_archive(output)
    return {"output": str(output), "files": len(payloads), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=ROOT / "artifacts/v2-comparison-demo.html")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/ragtruth_v2_audited.zip")
    args = parser.parse_args()
    print(json.dumps(build_package(ROOT, args.demo, args.output)))


if __name__ == "__main__":
    main()
