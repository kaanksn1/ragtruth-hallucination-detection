# Delivery verification — 31 October 2026

- Windows dedicated CRF environment: **193 scenarios, 162 passed / 31 skipped**.
- Windows PyTorch/Transformers environment: **193 scenarios, 190 passed / 3 skipped**.
- The two native CRF scenarios absent from the Transformers environment passed
  in the dedicated CRF environment. The Windows symlink-permission scenario
  passed in WSL as part of **7/7 handoff tests**. No failing scenario remained.
- JavaScript syntax: `node --check demo/viewer.js` passed.
- Original full Transformer validation at threshold 0.5 reproduced byte-identically.
  Both complete CRF validation outputs reproduced byte-identically. The final
  selected calibrated Transformer was replayed over all 2,240 validation responses
  and matched the calibration-cache prediction file byte for byte.
- Weighted training completed all three epochs over 12,702 training and 2,240
  validation responses without record limits. Its saved input/output hashes,
  train-only weight statistics and historical baseline outputs were verified.
- Scientific decisions were committed as `ce4c134` **before** official test
  preparation. Prediction jobs were committed as `6c4c2a2`; evaluation job and
  prediction hashes were committed as `47c3dd5`. All final jobs succeeded.
- Official test predictions were generated once for each of the three frozen
  models, and the frozen evaluator scored the same 2,675 good responses per model.
  No model/threshold changes followed test scoring.
- The final scientific artifact/code/data checksums were checked again after
  report assembly: all 97 declared identities remained unchanged.
- Independent read-only review checked every metric/support/cohort row of the
  generated technical report against the aggregate JSON artifacts. Interpretation
  also records zero Subtle Conflict test F1 and low exact-span performance.
- PNG model/type/task figures were exported with matching SVG files. Main test
  figures and the five-model validation comparison were visually inspected.
- The standalone demo contains all 2,240 good validation examples and three
  frozen model choices, with dataset attribution. It contains no live model.
  Earlier browser interaction/layout verification of the unchanged viewer is
  documented in [DEMO.md](../docs/DEMO.md).

Local test logs and run records remain in ignored `.cache/verification/` and
`outputs/`. The handoff packager validates ZIP CRC and every file's byte count and
SHA-256 against its embedded manifest before returning success. Its manifest
omits only itself. Model weights, raw/processed files and slides are excluded.

One initial completion preflight stopped on Git-normalized CRLF/LF differences
in the declared calibration policy, before model calibration or test access.
The fix accepts only that line-ending representation difference, preserves
the training input bytes, and has a regression test. The successful run is
`technical_completion_v2.json`; the failed preflight record is retained locally.
