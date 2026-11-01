# Fresh V2 development study

`scripts/prepare_v2_data.py` reads only the original project's prepared
`train.jsonl`, `validation.jsonl`, and `split_manifest_v1.csv`. It never opens
the original official-test files. Original validation remains development data.

Seed **20261101** selects approximately 10% of original-train source groups,
stratified by task, with no annotation-based selection. Source IDs sharing an
original content group stay together. These groups form a fresh study holdout;
they are not the original official test. Models for this study must start from
the pretrained encoder and train only on the new V2 train file.

Run `python scripts/prepare_v2_data.py --source-root /path/to/original-project`
from this worktree to write development
JSONLs, CRF views, `split_manifest_v2.csv`, and `split_summary_v2.json`. Original
response/source text, offsets, overlapping annotations, and metadata are retained.
New `study_id` and `original_project_split` fields identify this study. Manifest
rows for original official-test sources are excluded. Count-only holdout metadata
comes from the original manifest; no held-out label statistics are produced.

Holdout annotations are not materialized during development. Only after a frozen
`docs/V2_FINAL_PROTOCOL.json` is committed, run
`python scripts/prepare_v2_data.py --source-root /path/to/original-project --include-holdout`.
The default source path assumes the owner's nested worktree layout; other
machines must supply it explicitly. See [reproduction details](V2_REPRODUCIBILITY.md).
The command verifies that
the protocol matches its committed bytes and writes the new study's test JSONLs.
The development summary and manifest remain byte-identical, so model checkpoints
keep the same data provenance. A separate frozen job must hash the eventual
holdout JSONL before evaluation. Identical reruns preserve bytes; conflicting
outputs are refused. Every generated data file uses UTF-8 with LF newlines.
