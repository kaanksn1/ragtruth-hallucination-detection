"""Fetch immutable upstream files and verify hashes; never parse annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(config_path, target, *, check_only=False):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["repository"] != "https://github.com/ParticleMedia/RAGTruth":
        raise ValueError("Unexpected upstream repository.")
    if not re.fullmatch(r"[a-f0-9]{40}", config["revision"]):
        raise ValueError("An immutable 40-character upstream commit is required.")
    names = [Path(name).name for name in config["files"]]
    if set(names) != {"response.jsonl", "source_info.jsonl", "LICENSE"} or len(names) != 3:
        raise ValueError("Expected exactly the two dataset files and upstream LICENSE.")
    if config["files"] != ["dataset/response.jsonl", "dataset/source_info.jsonl", "LICENSE"]:
        raise ValueError("Unexpected upstream file paths.")
    expected = config["sha256"]
    if any(not re.fullmatch(r"[a-f0-9]{64}", expected[name]) for name in names):
        raise ValueError("Each upstream file requires its pinned SHA-256.")
    result = []
    for remote, name in zip(config["files"], names):
        destination = target / name
        if destination.exists():
            actual = sha256(destination)
            if actual != expected[name]:
                raise ValueError(f"Checksum mismatch for {name}; existing file was not overwritten.")
            result.append({"file": name, "status": "verified", "sha256": actual})
            continue
        if check_only:
            raise FileNotFoundError(f"Missing {destination}; run without --check-only to download.")
        target.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{config['revision']}/{remote}"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target, suffix=".partial", delete=False) as handle:
                temporary = Path(handle.name)
                request = Request(url, headers={"User-Agent": "ragtruth-course-project"})
                with urlopen(request, timeout=60) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
            actual = sha256(temporary)
            if actual != expected[name]:
                raise ValueError(f"Downloaded checksum mismatch for {name}.")
            # rename refuses to overwrite on Windows; the normal path is single-writer.
            if destination.exists():
                raise FileExistsError(f"Destination appeared during download: {destination}")
            temporary.rename(destination)
            result.append({"file": name, "status": "downloaded", "sha256": actual})
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/source.json")
    parser.add_argument("--target", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--check-only", action="store_true", help="No network; verify existing bytes only.")
    args = parser.parse_args()
    try:
        result = download(args.config, args.target, check_only=args.check_only)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Data setup failed: {exc}\n")
    print(json.dumps({"files": result}, indent=2))


if __name__ == "__main__":
    main()
