"""Export three common comparison figures from evaluate.py JSON (no model runs)."""
import argparse
import json
import os
from pathlib import Path

from ragtruth_data.schema import LABEL_TYPES
from ragtruth_eval import METRIC_VERSION


def plot_comparison(comparison, output_dir):
    if comparison.get("metric_version") != METRIC_VERSION or not comparison.get("models"):
        raise ValueError("expected common evaluator comparison JSON")
    output_dir = Path(output_dir)
    names = ("model_comparison", "type_f1", "task_f1")
    targets = [output_dir / f"{name}.{extension}" for name in names for extension in ("png", "svg")]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite comparison figures")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    models = comparison["models"]
    plt.rcParams.update({"font.size": 10, "figure.dpi": 120, "savefig.dpi": 200,
                         "axes.spines.top": False, "axes.spines.right": False})

    def grouped(filename, labels, values, title):
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.7), 5))
        x = np.arange(len(labels))
        width = min(0.24, 0.8 / len(values))
        for i, (name, scores) in enumerate(values.items()):
            bars = ax.bar(x + (i - (len(values) - 1) / 2) * width, scores, width, label=name)
            ax.bar_label(bars, labels=[f"{score:.3f}" for score in scores], fontsize=8, padding=3)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylabel("F1")
        ax.set_title(title + f" — {comparison['split']} (project metrics)")
        ax.legend(loc="upper right", frameon=False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(output_dir / f"{filename}.{extension}", bbox_inches="tight")
        plt.close(fig)

    grouped("model_comparison", ["Any character", "Typed character macro", "Typed exact span"],
        {name: [e["metrics"]["character"]["any"]["f1"], e["metrics"]["character"]["macro_f1_types"],
                e["metrics"]["span_exact"]["typed"]["f1"]] for name, e in models.items()}, "Model comparison")
    grouped("type_f1", ["Evident\nConflict", "Subtle\nConflict", "Evident\nBaseless Info", "Subtle\nBaseless Info"],
        {name: [e["metrics"]["character"][kind]["f1"] for kind in LABEL_TYPES] for name, e in models.items()},
        "Character F1 by type")
    tasks = sorted({task for e in models.values() for task in e["metrics"]["breakdowns"]["task_type"]})
    grouped("task_f1", tasks, {name: [e["metrics"]["breakdowns"]["task_type"].get(task, {})
                .get("character", {}).get("any", {}).get("f1", 0.0) for task in tasks] for name, e in models.items()},
            "Any character F1 by task")
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        targets = plot_comparison(json.loads(args.input.read_text(encoding="utf-8")), args.output_dir)
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(1, f"Plotting failed: {exc}\n")
    print(json.dumps({"figures": [str(path) for path in targets]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
