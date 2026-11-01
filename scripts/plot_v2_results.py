"""Label V2 figures with their study cohort, without changing stored metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot_results import plot_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    comparison = json.loads(args.input.read_text(encoding="utf-8"))
    expected = hashlib.sha256((ROOT / "data/splits/split_summary_v2.json").read_bytes()).hexdigest()
    if comparison.get("data", {}).get("sha256", {}).get("split_summary") != expected:
        raise ValueError("figure input must identify this V2 cohort")
    labels = {"validation": "V2 development", "test": "V2 internal holdout"}
    comparison["split"] = labels[comparison["split"]]
    print(json.dumps({"figures": [str(p) for p in plot_comparison(comparison, args.output_dir)]}))


if __name__ == "__main__":
    main()
