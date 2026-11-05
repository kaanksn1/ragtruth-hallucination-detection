"""Build the shared RAGTruth data (2. kişi). Standard library only.

    python scripts/download_data.py          # 1. kişi's verified download (once)
    python scripts/prepare_data.py           # manifest check, train/validation files, report

Writes
    data/processed/{train,validation}.jsonl       shared records (docs/DATA_FORMAT.md), committed delivery
    data/processed/crf_{train,validation}.jsonl   word tokens + O/B/I tags for the CRF, committed delivery
    data/splits/split_manifest_v1.csv             source_id -> split, committed (source of truth)
    data/splits/split_summary_v1.json             counts, seed, input/output hashes, committed
    data/samples/*.jsonl                          a few train/validation examples, committed
    docs/data_report.json, docs/DATA_REPORT.md    train/validation analysis, committed

The official test labels are not read unless --include-test is given; that is
meant for the single final evaluation after the protocol is frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ragtruth_data import analysis, data, splits  # noqa: E402
from ragtruth_data.alignment import word_level_example  # noqa: E402
from ragtruth_data.schema import FORMAT_VERSION  # noqa: E402


def pick_samples(records: list) -> list:
    """Per task: one response without and one with labels; plus overlap cases (train/validation only)."""
    good = sorted((r for r in records if r["quality"] == "good"), key=lambda r: int(r["id"]))
    chosen, ids = [], set()

    def take(pred):
        for r in good:
            if r["id"] not in ids and pred(r) and len(r["source_text"]) < 6000:
                chosen.append(r)
                ids.add(r["id"])
                return

    for t in ("QA", "Summary", "Data2txt"):
        take(lambda r, t=t: r["task_type"] == t and not r["has_hallucination"])
        take(lambda r, t=t: r["task_type"] == t and 1 <= r["n_labels"] <= 3)
    take(lambda r: any(l["overlaps_with"] for l in r["labels"]))
    take(lambda r: any(r["labels"][o]["label_type"] != l["label_type"] for l in r["labels"] for o in l["overlaps_with"]))
    return chosen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/data.json")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    ap.add_argument("--write-manifest", action="store_true",
                    help="(re)write the committed manifest instead of checking against it")
    ap.add_argument("--include-test", action="store_true",
                    help="also write test files with labels (final evaluation only, after the protocol is frozen)")
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    try:
        raw = data.verify_raw(args.raw_dir)
    except (FileNotFoundError, ValueError) as exc:
        ap.exit(1, f"{exc}\n")
    sources, responses = data.load_raw(args.raw_dir)
    print(f"raw files verified ({len(sources)} sources, {len(responses)} responses)")

    split_of, group_of = splits.assign_splits(sources, responses, cfg["validation_fraction"], cfg["seed"])
    print("leakage check:", splits.check_no_leakage(split_of, group_of, responses))
    manifest = ROOT / cfg["manifest"]
    if manifest.exists() and not args.write_manifest:
        committed = splits.read_manifest(manifest)
        if committed != split_of:
            diff = [s for s in split_of if committed.get(s) != split_of[s]]
            ap.exit(1, f"split differs from the committed manifest for {len(diff)} sources (e.g. {diff[:5]}); "
                       "the committed manifest is the source of truth.\n")
        print("split matches", cfg["manifest"])
    else:
        splits.write_manifest(manifest, sources, responses, split_of, group_of)
        print("wrote", cfg["manifest"])

    wanted = {"train", "validation"} | ({"test"} if args.include_test else set())
    records = [data.build_record(r, sources[str(r["source_id"])], split_of[str(r["source_id"])])
               for r in responses if split_of[str(r["source_id"])] in wanted]
    print(f"joined {len(records)} responses; every label offset matches its text")

    out = ROOT / cfg["processed_dir"]
    out_hashes = {}
    for s in sorted(wanted, key=["train", "validation", "test"].index):
        rows = sorted((r for r in records if r["split"] == s), key=lambda r: int(r["id"]))
        p, cp = out / f"{s}.jsonl", out / f"crf_{s}.jsonl"
        data.write_jsonl(p, rows)
        data.write_jsonl(cp, (dict(word_level_example(r), split=s, quality=r["quality"]) for r in rows))
        if s != "test":
            out_hashes[p.name] = data.sha256_file(p)
            out_hashes[cp.name] = data.sha256_file(cp)
        print(f"{s:<10} {len(rows):>6} responses -> {p.relative_to(ROOT)}")

    official_test = [r for r in responses if r["split"] == "test"]
    test_counts = {"sources": len({str(r["source_id"]) for r in official_test}), "responses": len(official_test)}
    src_counts = {s: Counter() for s in ("train", "validation", "test")}
    for sid, s in split_of.items():
        src_counts[s][sources[sid]["task_type"]] += 1
    resp_counts = Counter(split_of[str(r["source_id"])] for r in responses)
    summary = {
        "format_version": FORMAT_VERSION,
        "revision": raw["revision"],
        "raw_sha256": raw["sha256"],
        "seed": cfg["seed"],
        "validation_fraction": cfg["validation_fraction"],
        "stratify_by": cfg["stratify_by"],
        "grouping": "source_id, plus sources with the same normalised source text / prompt / Yelp business / QA question or passages",
        "sources_per_split": {s: dict(sorted(c.items()), total=sum(c.values())) for s, c in src_counts.items()},
        "responses_per_split": {s: resp_counts[s] for s in ("train", "validation", "test")},
        "processed_sha256": out_hashes,
    }
    summ = ROOT / cfg["summary"]
    summ.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print("wrote", cfg["summary"])

    trainval = [r for r in records if r["split"] in ("train", "validation")]
    report = analysis.full_report(trainval, group_of, test_counts)
    (ROOT / cfg["report_json"]).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    (ROOT / cfg["report_md"]).write_text(analysis.render_markdown(report, summary), encoding="utf-8", newline="\n")
    print("wrote", cfg["report_json"], "and", cfg["report_md"])

    samples = pick_samples(trainval)
    sd = ROOT / cfg["samples_dir"]
    data.write_jsonl(sd / "sample_records.jsonl", samples)
    data.write_jsonl(sd / "sample_crf.jsonl", (word_level_example(r) for r in samples))
    print(f"wrote {len(samples)} sample records to {cfg['samples_dir']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
