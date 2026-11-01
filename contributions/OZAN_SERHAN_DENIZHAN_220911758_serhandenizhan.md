# Ozan Serhan Denizhan — 220911758

GitHub username: serhandenizhan
Responsibility: Person 5 — independent score verification, label audit, slides and presentation

> This file was compiled from commit and PR records and must be confirmed by the member.

## What I completed

- `scripts/independent_rescore.py`: separate scorer that reads gold labels from the raw
  dataset instead of the project scorer and recomputes the frozen V1 test predictions.
  It also adds post-hoc overlap F1, response-level F1 and validation-only boundary trials.
- `scripts/audit_label_quality.py`: annotation quality audit on good-quality train and
  validation records only. It measures span boundaries, per-type span lengths,
  `implicit_true`, `due_to_null` and note-type agreement. No labels were changed.
- `results/independent_rescore.md`, `results/independent_rescore.json`,
  `results/label_quality.md`, `results/label_quality.json`: generated reports.
- Reviewed and merged [PR #8](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/8).
- Prepared the presentation and its speaker notes.

I did not change any model, threshold, split, frozen protocol or other member's files. The
common evaluation first planned for Person 5 was reassigned to Person 1 on 31 October.

## Verification and experiments

- Command: `python scripts/audit_label_quality.py`; `scripts/independent_rescore.py` takes
  the V1 prediction package path as an argument.
- Data split: official test predictions for 2,675 responses; label audit on 14,942
  good-quality train and validation responses; boundary trials on 2,240 validation responses.
- Output: `results/independent_rescore.md` and `results/label_quality.md`.

| Model | Any F1 | Typed macro F1 | Exact typed F1 | Overlap F1 | Response F1 |
|---|---:|---:|---:|---:|---:|
| `crf_response_only` | 0.2074 | 0.1195 | 0.0615 | 0.2616 | 0.5878 |
| `crf_source` | 0.3544 | 0.1976 | 0.0667 | 0.3184 | 0.6624 |
| `transformer` | 0.5206 | 0.2984 | 0.0678 | 0.3806 | 0.7285 |

- The first three metrics reproduce `results/test_comparison.md` to four decimals. Overlap
  and response-level F1 are post-hoc and were not used for model selection.
- 37.67% of audited spans start or end with punctuation, and 74.3% of Subtle Baseless Info
  spans are `implicit_true`.
- On validation, the calibrated thresholds produced about 2.5 times more predicted spans
  than gold spans. Post-processing is not recommended.
- Limitations: annotator agreement could not be measured, because the 43 identical response
  pairs contain no pair where both sides are positive. A local unit test run reported 3
  errors in `test_calibration`, `test_evaluation` and `test_transformer_integrity`; the
  cause was not determined.

## GitHub evidence

- My commit link: [Independent rescoring and label audit](https://github.com/kaanksn1/ragtruth-hallucination-detection/commit/72f909616ac6b4a74987b758e8d6e0ceb1f21da6)
- My pull request link: [#9 Independent rescoring and label quality audit](https://github.com/kaanksn1/ragtruth-hallucination-detection/pull/9) (open)

## Contribution to the presentation

- Prepared the English PPTX with charts and speaker notes, a separate speaker script and
  Turkish per-slide study notes. The PPTX is not stored in the repository.
