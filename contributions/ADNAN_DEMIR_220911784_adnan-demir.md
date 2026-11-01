# Adnan Demir — 220911784

GitHub username: adnan-demir
Responsibility: Person 4 — context-aware Transformer, then model selection fixes

> This file was compiled from commit and PR records and must be confirmed by the member.

## Completed work

### 1. ModernBERT detector

- `transformer.py`, `scripts/transformer.py`: multilabel ModernBERT-base detector with four
  independent token outputs, masked BCE on response tokens, window decoding back to
  character offsets, and shared `{id, spans}` predictions.
- `scripts/smoke_transformer.py`, `tests/test_transformer.py`: smoke run and tests.
- `configs/transformer.json`, `configs/jobs/transformer_train.json`,
  `requirements-transformer.txt`, `.github/workflows/transformer.yml`: configuration,
  central training job, pinned dependencies and CI. A follow-up commit fixed the pip cache paths.
- `docs/TRANSFORMER.md`, `docs/TRANSFORMER_PR.md`, `docs/TRANSFORMER_REVIEW.md`: method and review notes.

### 2. Selection and threshold fixes

- `scripts/calibrate_transformer.py`, `scripts/complete_technical_pipeline.py`: ordered
  checkpoint and experiment selection by detection F1 first, added a 1% precision floor, and
  compared default, global, per-type and joint configurations.
- `scripts/baseline_crf.py`: added per-type and joint CRF threshold search with the same
  precision floor. The existing shared-threshold output stayed unchanged.
- `configs/calibration_v2.json`, `configs/calibration_weighted_policy_v2.json`,
  `configs/calibration_full_train_run1.json`: successor configurations. Frozen V1
  configurations stayed byte-identical.
- `PROJECT_HANDOFF.md`, `results/session_2026-11-01/`: handoff and experiment summaries.

These fixes came after the frozen official test. They change no V1 test result. Owner
integration fixes for the initial Transformer are separate commits by Person 1.

## Verification and experiments

- Full baseline run in the handoff: 3 epochs, seed 42, RTX 5070 Ti, 62.4 minutes.
- Data split: full `quality=good` validation split of 2,240 responses; no official test use.
- Output: `results/session_2026-11-01/` and `PROJECT_HANDOFF.md`.

| Model | Previous detection F1 | Updated detection F1 |
|---|---:|---:|
| Transformer | 55.00% | 55.07% |
| CRF | 39.08% | 39.22% |

- Transformer macro F1 rose from 27.86% to 29.23%. Per-type CRF search alone lowered detection
  F1 to 38.84%, while joint search reached 39.22%.
- PR #8 reports 204 tests passed and 6 skipped.
- Limitations: Subtle Conflict F1 stayed 0%, rare-class imbalance remains, and overfitting
  appeared after epoch 2. No model reached the 80% target.

## GitHub evidence

- My commit links:
  - [ModernBERT detector](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/71becb0e41b2c6558199d193450c55c1b751c34d)
  - [CI pip cache fix](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/ea358486f530271b6c921011c6413fe562c81f38)
  - [Detection-first selection](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/4557d678a3d72132d5b9fc0f35c318a7d84126aa)
  - [CRF joint threshold search](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/8f085b4b2cd9ccd644d692c7e109a542e2001801)
  - [Documentation and successor configs](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/92e56b737f1173e8363c6932d07181948fe8fd01)
  - [Handoff and experiment summaries](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/f7ce565588fae4759781288bac9d62f03e57a330)
- My pull request links:
  - [#4 ModernBERT detector](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/4)
  - [#8 Selection and joint threshold fixes](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/8)

## Contribution to the presentation

- The Transformer method is documented in `docs/TRANSFORMER.md`. The repository has no
  separate slide record for this member.
