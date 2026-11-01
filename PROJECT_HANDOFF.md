# RAGTruth Hallucination Detection — Project Handoff

> **Review update — 2026-11-02:** This is a historical snapshot of a teammate's
> session and machine. Its new-session scores are reported validation results;
> the new predictions/weights are not present in this checkout. The shipped
> calibration JSON contains global thresholds (55.00%), while 55.07% is a separate
> joint experiment report. Original official-test results remain frozen.
> Current commands/modes are in [CALIBRATION.md](docs/CALIBRATION.md).
> This repository has Git; the "no Git" statements below refer to the session's
> delivered working folder. Member confirmation of personal reports remains separate.

**Last updated:** 2026-11-01, end of the model-optimization session described below.
**Audience:** another developer or AI coding assistant continuing this work without repeating completed experiments.

This document is a snapshot of real, measured state. Every number in it either comes from
a file in this repository (cite the path) or from a command run during this session (cited
below). Where a number is a *frozen historical* result (produced by the original team, not
reproducible here because the underlying checkpoints/`.crfsuite` models aren't shipped in
this delivered copy), it is explicitly labeled "historical/frozen." Where a number is from
*this session's* real, independently-trained artifacts, it is labeled "this session, real."
Nothing in this document is fabricated or extrapolated.

---

## 1. Project objective and current status

**Objective:** detect hallucinated spans (4 types: Evident Conflict, Subtle Conflict,
Evident Baseless Info, Subtle Baseless Info) in LLM responses grounded in a source document
(RAGTruth dataset), using character-level union ("any") F1 across all 4 types as the primary
metric, with per-class F1 and 4-class macro F1 as secondary metrics.

**Status:** Three originally-reported problems (incorrect threshold/model selection,
CRF false-positive risk, CRF/Transformer rare-class and overfitting issues) have been
investigated with real evidence and partially fixed. A full, real ModernBERT-base baseline
has been trained from scratch this session and calibrated. CRF has been fully retrained this
session with a new joint-threshold-optimization layer. **No model has reached the 80%
detection-F1 long-term target; these trials establish neither an architectural
ceiling nor that one rare category alone explains the detection-F1 gap (see §8).**

**Best reported validation candidate:** ModernBERT-base, checkpoint `epoch-2`, **joint-threshold
configuration, detection F1 = 55.07%** on the full 2,240-response validation split (this
session, real — see §4).

---

## 2. The three original problems and their fixes

### Problem A — Threshold/checkpoint selection prioritized macro F1 over detection F1
**Root cause:** `scripts/calibrate_transformer.py`'s checkpoint-selection sort key was
`(macro_f1_types, any_f1, ...)` — macro primary. The same pattern existed independently in
`scripts/complete_technical_pipeline.py::choose_experiment` (weighted-vs-unweighted family
choice).
**Fix:** both now sort by `(any_f1, macro_f1_types, ...)` — detection F1 primary, macro F1
only a tie-break. Additionally, each checkpoint's *final* threshold choice used to be
unconditionally the per-type-tuned result; it's now chosen by comparing **default / global /
per-type / joint** configurations and picking whichever has the best detection F1
(`select_best_configuration` in both `scripts/calibrate_transformer.py` and
`scripts/baseline_crf.py`).
**`scripts/build_technical_report.py` was deliberately left unchanged** — it verifies the
already-frozen historical decision (`docs/FINAL_PROTOCOL.json`, `results/FINAL_REPORT.md`),
which legitimately used the old macro-first rule; changing it would break its ability to
reproduce that specific historical, already-tested result.
**Verified:** real measurement on this session's own Transformer run showed per-type tuning
(52.44% any-F1) would have been shipped over the better global config (55.00%) without this
fix — i.e. the bug this fixes is not hypothetical, it reproduced on real data (see §4).

### Problem B — CRF false-positive risk from threshold selection
**Root cause:** no per-type threshold search existed for CRF at all (one shared threshold
for all 4 types, chosen from the aggregate 'any' F1). Once a per-type search was added, it
could still regress the primary detection metric by optimizing each type's own F1 blindly.
**Fix:** `MIN_PRECISION_FLOOR = 0.01` in `scripts/baseline_crf.py` (reused by
`scripts/calibrate_transformer.py`) excludes any threshold with <1% precision unless it
predicts zero characters, blocking a "flood everything, get a technically-nonzero F1" failure
mode. `tune_joint_thresholds` (both files) searches threshold *combinations* to directly
maximize the union metric instead of each type's own F1 independently.
**Verified with real data, twice:** CRF joint search found 39.22% any-F1 vs. 39.08%
(shared)/38.84% (independent per-type) — real, measured, +0.14pp (§3). The feared "~1.84
million FP" scenario from the original bug report **does not occur** anywhere in the real,
selected configurations; it was only reproduced at an extreme, floor-rejected threshold
(0.001 for CRF's Subtle Conflict, giving a 164.69× predicted/gold ratio — confirmed real,
but never selected by any actual configuration).

### Problem C — CRF non-learning / Transformer overfitting
**CRF side — confirmed, real, unresolved.** Subtle Conflict has 0.00% F1 at every tested
threshold/setting, before and after all fixes. Root cause confirmed by direct probing of the
trained model: the maximum P(B/I) ever assigned to a true Subtle Conflict token across all
2,240 validation responses is 0.36%; the model's own training-set recall is real (mean
P=0.232 on train gold tokens) but does not generalize (mean P=0.0004 on validation) — textbook
overfitting on 160 training spans. An aggressive-regularization counter-experiment made this
*worse*, not better, within the same iteration budget (see §7). **Recommendation (per
original task instructions): do not force CRF to handle this category; prioritize the
Transformer, which has non-zero (if weak) signal for it.**
**Transformer side — confirmed real, partially characterized.** Validation loss dips then
rises across 3 epochs (overfitting onset); already mitigated by existing min-val-loss
checkpoint selection (keeps epoch-2, discards epoch-3). **No learning-rate schedule has ever
been used or tested** (confirmed by code audit — `scripts/transformer.py`'s `AdamW` runs at a
constant LR with no scheduler). This is a real, well-motivated, untried lever, not yet
executed.

---

## 3. CRF experiment history (this session, real)

Real full training/validation run: `outputs/crf_local_validation_run1/` (not run through
`run_job.py` — this folder has no git repository, so the wrapper's git-provenance step was
bypassed; `scripts/baseline_crf.py` was invoked directly). Selected setting: `c1=0.1,c2=1.0`,
seed 42. Trained `.crfsuite` model checksums are **byte-identical** to the historical/frozen
team run recorded in `docs/FINAL_PROTOCOL.json` — full reproducibility confirmed.

| Configuration | Detection F1 | Macro F1 | Notes |
|---|---:|---:|---|
| default (threshold 0.5, shared) | 26.63% | 13.11% | |
| global (shared threshold 0.15, existing) | 39.08% | 19.99% | Matches historical/frozen `results/training_crf.json` exactly |
| per-type (independent, 1% floor) | 38.84% | 20.84% | Regresses detection F1 vs. global — the real evidence for Problem A's fix |
| **joint (new, this session)** | **39.22%** | 20.35% | **Selected** — best detection F1 among all four |

Subtle Conflict: 0.00% F1 in all four configurations. One targeted retrain experiment
(`c1=0, c2=0.001, minfreq=1`, single type, ~74s) tested whether far less regularization lets
the model learn at all — it made training-set recall *worse* (mean P=0.0004 vs. 0.232 for the
current setting), evidence that 150 L-BFGS iterations don't converge with that many free
features, not that the current regularization is the bottleneck. **No further CRF
intervention is recommended** for Subtle Conflict specifically.

Full detail: `docs/BASELINE_CRF.md` (dated note, 2026-11-01), `outputs/crf_local_validation_run1/crf_summary.json`
(`selected`, `selected_per_type`, `selected_joint`, `best_configuration` keys),
`outputs/crf_local_validation_run1/four_way_comparison.json`.

---

## 4. Transformer experiment history (this session, real)

### Pilot (functional check only — not a performance baseline)
`outputs/transformer_pilot_cpu/`: 8 train / 4 validation responses, 1 epoch, CPU. Real
pretrained `answerdotai/ModernBERT-base` weights (pinned revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`), downloaded to `~/.cache/huggingface` (573MB).
Used only to prove the fixed calibration pipeline behaves correctly end-to-end on real
predictions (it did — picked `global` over `per_type` correctly despite `per_type` having
~24x higher macro F1). **The F1 numbers from this run (~5%) are meaningless as performance
and must never be cited as a baseline.**

### Full baseline training (real, GPU, this session)
`outputs/transformer_full_train_run1/`: full real split (12,702 train / 2,240 validation good
responses), 3 epochs, `configs/transformer.json` unchanged, seed 42, RTX 5070 Ti.

| Epoch | Train loss | Val loss | Train time | Val time |
|---|---:|---:|---:|---:|
| 1 | 0.05851 | 0.04743 | 1152.4s | 57.4s |
| 2 | 0.03926 | **0.04365 (best)** | 1125.6s | 62.8s |
| 3 | 0.02947 | 0.04822 | 1224.5s | 60.1s |

Total: 62.4 minutes. No errors, no OOM (peak 3.1GB/16.3GB VRAM). `checkpoints/epoch-2`
correctly retained as best (min validation loss); `epoch-3` checkpoint was never saved
(didn't improve).

### Calibration + Experiment 1 (joint threshold optimization, real, this session)
`outputs/transformer_calibration_run1/`: both checkpoints (`epoch-1`, `epoch-2`) calibrated
on the full validation split via `scripts/calibrate_transformer.py --config
configs/calibration_full_train_run1.json --device cuda`. Cached probabilities:
`outputs/transformer_calibration_run1/probabilities/candidate-{1,2}.jsonl.gz`.

epoch-2 selected over epoch-1 (55.00% vs 50.98% any-F1 under `global`, consistent with the
loss-based checkpoint choice). Then Experiment 1 (`tune_joint_thresholds`, new this session)
was run **reusing the existing cached probabilities — no new GPU inference, no retraining**
(script: see §9; result file:
`outputs/transformer_calibration_run1/experiment1_joint_threshold_comparison.json`):

| Configuration | Detection P | Detection R | **Detection F1** | Macro F1 | Thresholds |
|---|---:|---:|---:|---:|---|
| default (0.5 everywhere) | 72.61% | 33.50% | 45.85% | 23.24% | all 0.5 |
| global (shared, previously selected) | 54.39% | 55.62% | 55.00% | 27.86% | all 0.2 |
| per-type (independent, 1% floor) | 48.73% | 56.77% | 52.44% | 30.65% | EC 0.2, SC 0.6, EBI 0.3, SBI 0.05 |
| **joint (new, this session)** | 54.17% | 55.99% | **55.07%** | 29.23% | EC 0.2, SC 0.6, EBI 0.2, SBI 0.15 |

**Joint wins — real, measured +0.07pp over the 55.00% baseline.** Modest, not fabricated or
rounded up. Per requirement, if joint had *not* beaten 55.00%, global would have been
retained; it did, by 0.07pp, so joint is the new selected configuration. 9,317 threshold
combinations were evaluated in 101.5 seconds (CPU, no GPU).

**Per-class breakdown, selected (joint) configuration:**

| Type | P | R | F1 | TP | FP | FN | Predicted | Gold | Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Evident Conflict | 56.41% | 45.76% | 50.53% | 11,507 | 8,893 | 13,639 | 20,400 | 25,146 | 0.81x |
| Subtle Conflict | 0.00% | 0.00% | 0.00% | 0 | 0 | 830 | 0 | 830 | 0.00x |
| Evident Baseless Info | 42.49% | 60.93% | 50.07% | 35,194 | 47,629 | 22,568 | 82,823 | 57,762 | 1.43x |
| Subtle Baseless Info | 28.22% | 11.50% | 16.34% | 2,162 | 5,500 | 16,646 | 7,662 | 18,808 | 0.41x |

No category exceeds the 10x predicted/gold flag in any of the four configurations (max
observed anywhere: per-type's Subtle Baseless Info at 3.67x).

### Historical/frozen comparison (team's original run, NOT reproduced here — different hardware/seed-independent run, cited for context only)
`results/FINAL_REPORT.md`: calibrated any-F1 54.05%, macro F1 32.01%, on RX 7800 XT. This
session's independently-trained model (55.07%) is close to but not directly comparable to
that number (different training run, same config/seed/data — expected variance, not a
controlled single-seed guarantee of superiority).

---

## 5. Reported validation candidate and F1

**ModernBERT-base**, checkpoint `outputs/transformer_full_train_run1/checkpoints/epoch-2`,
**joint threshold configuration** (thresholds: Evident Conflict 0.2, Subtle Conflict 0.6
[never fires], Evident Baseless Info 0.2, Subtle Baseless Info 0.15).

**Reported detection F1 = 55.07%** (validation, this session's joint experiment).
The committed normal calibration artifact still contains global 0.2 thresholds
and reports 55.00%; the joint candidate was not exported there. New-session
predictions are needed for independent rescoring. Neither value is a new
official-test result or proof of generalization beyond this reused dev set.

---

## 6. GPU environment and dependencies

- Hardware: NVIDIA RTX 5070 Ti, 16GB VRAM, driver 617.42, compute capability (12,0) — Blackwell.
- **Dedicated venv**: `.venv-gpu/` at repo root (Python 3.12.10 — NOT the default `py` 3.14,
  which has no CUDA torch wheels at all). Created via `py -3.12 -m venv .venv-gpu`.
- Installed: `torch==2.11.0+cu128` (from `https://download.pytorch.org/whl/cu128`),
  `transformers==4.57.6`, `tokenizers==0.22.2`, `safetensors`, `numpy`,
  `python-crfsuite==0.9.12` — matches `requirements-transformer.txt`/`requirements-crf.txt`
  exactly except for the CUDA-specific torch build.
- **Verified empirically** (not assumed): `sm_120` is in the installed wheel's compiled arch
  list; a real GPU matmul executed correctly with no compatibility warnings.
- Separately, the default global Python 3.14 interpreter has CPU-only `torch==2.11.0+cpu`,
  `transformers==4.57.6`, `tokenizers==0.22.2`, `python-crfsuite==0.9.12` installed (used for
  the CRF run, the CPU pilot, and all test-suite runs — CRF doesn't need CUDA).
- All commands below assume Windows PowerShell/Git-Bash; use `./.venv-gpu/Scripts/python.exe`
  for anything GPU-related, plain `py` (3.14, CPU) for CRF/tests.

---

## 7. Relevant code, configuration, and artifact paths

### Code changed this session
- `scripts/baseline_crf.py` — `MIN_PRECISION_FLOOR`, `tune_type_thresholds`,
  `tune_joint_thresholds`, `select_best_configuration`, wired into `cmd_train` as
  `selected_per_type`/`selected_joint`/`best_configuration` (additive; original `selected`
  path untouched).
- `scripts/calibrate_transformer.py` — `select_checkpoint` (detection-F1-first), floor-aware
  `tune_thresholds`, `tune_global_threshold`, `tune_joint_thresholds`,
  `select_best_configuration`, wired into `main()`'s per-candidate loop (default/global/
  per_type/joint comparison; ships whichever wins).
- `scripts/complete_technical_pipeline.py` — `choose_experiment` now detection-F1-first.
- `docs/BASELINE_CRF.md`, `docs/CALIBRATION.md` — dated addenda only; original historical
  content above them is untouched (hash-verified).
- Tests: `tests/test_baseline_crf.py`, `tests/test_calibration.py`,
  `tests/test_completion_pipeline.py` — regression tests for every fix above, including
  hand-verified cases proving joint search mathematically cannot do worse than independent
  per-type search (it searches a superset that includes the independent combo).

### Deliberately NOT changed
- `scripts/build_technical_report.py` — verifies the frozen historical decision; still
  encodes the old macro-first rule because that's literally what produced
  `results/FINAL_REPORT.md`. Do not "fix" this unless producing a *new* frozen decision.
- `configs/calibration.json`, `configs/calibration_weighted_policy.json`, `configs/crf.json`
  — byte-identical to their SHA-256 in `docs/FINAL_PROTOCOL.json`'s provenance trail (verified
  every session). **Never edit these in place** — their hash is load-bearing history.
- `docs/FINAL_PROTOCOL.json`, `results/FINAL_REPORT.md`, `results/*.json` — frozen, untouched,
  hash-verified every session.

### New configs (successors to the frozen ones, for new runs)
- `configs/calibration_v2.json`, `configs/calibration_weighted_policy_v2.json` — corrected
  `selection_metric` field, not yet used for a real run.
- `configs/calibration_full_train_run1.json` — **the one actually used** for this session's
  real Transformer calibration (points at `outputs/transformer_full_train_run1`'s two
  checkpoints).

### Real artifacts produced this session (none overwrite anything)
- `outputs/crf_local_validation_run1/` — real CRF training/validation run (models, summary,
  report, `four_way_comparison.json`).
- `outputs/transformer_pilot_cpu/` — functional-only tiny pilot (not a performance number).
- `outputs/transformer_full_train_run1/` — real full 3-epoch baseline (both checkpoints,
  `transformer_summary.json`).
- `outputs/transformer_calibration_run1/` — real calibration of both checkpoints
  (`calibration.json`, `calibration_report.json`, `probabilities/*.jsonl.gz`,
  `experiment1_joint_threshold_comparison.json`).
- `.venv-gpu/` — the GPU-enabled Python environment (not source-controlled; recreate via §9
  if missing).

**All of the `outputs/` paths above are local-only** (gitignored, consistent with this
repo's existing convention — `outputs/` is for large run artifacts: checkpoints, `.crfsuite`
models, probability caches). They are **not** part of this PR and will not exist for anyone
who clones the repo; regenerate them via §9 if needed. Small, text-only summaries extracted
from those runs **are** committed, under `results/session_2026-11-01/`: `crf_summary.json`,
`crf_report.md`, `crf_four_way_comparison.json`, `transformer_summary.json`,
`transformer_calibration.json`, `transformer_calibration_report.json`,
`transformer_experiment1_joint_threshold_comparison.json` (176KB total). These are the
evidence behind every number in this document and do not require re-running anything to read.

---

## 8. Known limitations

- **Subtle Conflict remained unresolved in these trials.** Reported probability
  probing shows essentially zero score separation between gold and non-gold characters
  (mean 0.0013 vs 0.0008) — not a calibration problem. Qualitative inspection of real gold
  spans (`"for four days"`, `"The 12-episode miniseries"`, `"appear social and casual"`)
  suggests numeric/semantic relations deserve further study. This single recipe does
  not prove that alternative losses or architectures cannot learn the class. Its 830
  dev gold characters are only about 0.81% of the any-character gold; failure on this
  type alone cannot explain detection F1 near 55% or prove an 80% ceiling.
- **Subtle Baseless Info has real, recoverable signal** (confidence separation ~10x,
  per-type-only optimum F1 21.61% vs the shipped 16.34%) — this is the most promising lever
  for further gains, likely via loss weighting or a smarter joint-search variant that doesn't
  purely optimize the aggregate union metric.
- **No learning-rate schedule has ever been tested.** Constant 2e-5 throughout; the
  overfitting onset at epoch 3 is a textbook case for LR decay, untried.
- **Positive-weighting (sqrt-capped, all 4 types) was already tried historically and failed**
  (51.54% vs 54.05% unweighted any-F1, historical/frozen `results/FINAL_REPORT.md`) — don't
  repeat the blanket version; a type-specific variant (Subtle Baseless Info only) is untested
  and better-justified by the confidence-distribution evidence above.
- **The reported 55.07%/55.00%/39.22% numbers are validation-selected, not independent
  estimates.** Checkpoint choice, configuration choice, and threshold choice all reuse the
  same 2,240 validation responses. These are selection-set measurements. The project
  already saw its official-test results before this session, so they are not a new
  blind holdout. Full V1 training also includes the separate V2 study's carved
  holdout, which cannot serve as unseen evaluation for this full-V1 model.
- **The 1% precision floor is reasonable but not perfectly tight.** Measured: a 0.5% floor
  would let Subtle Conflict recover a trivial amount of signal (F1 0.84% instead of 0%, at a
  1.88x ratio) — not worth it on its own, but worth knowing it's a real, measured sensitivity,
  not an arbitrary cutoff.
- **This folder has no git repository.** `scripts/run_job.py`'s wrapper (which records
  git commit/branch/working-tree state for provenance) cannot run here; all this session's
  real runs invoked `scripts/baseline_crf.py`/`scripts/transformer.py`/
  `scripts/calibrate_transformer.py` directly instead. If a git repo is ever initialized here
  (e.g. by cloning the real GitHub remote mentioned in `README.md`), the central-run wrapper
  should work normally.

---

## 9. Instructions for reproducing these results

```powershell
# 0) One-time setup (if not already done)
py -3.12 -m venv .venv-gpu
./.venv-gpu/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
./.venv-gpu/Scripts/python.exe -m pip install "transformers==4.57.6" "tokenizers==0.22.2" "safetensors>=0.4.3,<1" numpy
py -m pip install python-crfsuite==0.9.12 "transformers==4.57.6" "tokenizers==0.22.2" "safetensors>=0.4.3,<1"

# 1) Real data (deterministic; reproduced byte-identical to frozen hashes this session)
py scripts/download_data.py
py scripts/prepare_data.py

# 2) CRF (CPU, ~25 min) -- run_job.py needs git; this repo has none, so invoke directly
py scripts/baseline_crf.py train --config configs/crf.json --output outputs/<new-crf-run> --seed 42

# 3) Transformer full training (GPU, ~62 min on RTX 5070 Ti; longer on slower/CPU hardware)
./.venv-gpu/Scripts/python.exe scripts/transformer.py train --config configs/transformer.json --seed 42 --device cuda --output outputs/<new-transformer-run>

# 4) Calibration against the new checkpoints (GPU for speed, CPU also works)
#    First write a config pointing at the new run's checkpoints/epoch-1 and epoch-2
#    (see configs/calibration_full_train_run1.json for the exact shape), then:
./.venv-gpu/Scripts/python.exe scripts/calibrate_transformer.py --config configs/<your-config>.json --output outputs/<new-calibration-run> --device cuda

# 5) Tests (either interpreter; HF deps needed for full coverage)
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
py -m unittest discover -s tests -v
# Expect: 204 passed, 3 skipped (1 Windows symlink limitation, 2 need a cached real
# ModernBERT tokenizer that is intentionally never auto-downloaded)
```

To re-run **Experiment 1 (joint threshold search) without any new inference**, reuse an
existing probability cache directly — see the pattern in
`outputs/transformer_calibration_run1/experiment1_joint_threshold_comparison.json`'s
generating logic: load records via `scripts.transformer.read_records`, load the cache via
`scripts.calibrate_transformer.read_cache`, then call `tune_joint_thresholds` /
`select_best_configuration` directly. Takes ~100 seconds, no GPU.

---

## 10. Recommended next experiments (NOT executed — proposals only)

In priority order, each changes one factor from the current 55.07% baseline:

1. **Learning-rate scheduling** (GPU, ~65 min, ~3.2GB VRAM). Add cosine or linear decay to
   `scripts/transformer.py`'s `AdamW` (currently constant LR, never tested). Directly targets
   the observed epoch-3 overfitting onset. Rollback: discard if detection F1 < 55.07% or any
   class exceeds the 10x predicted/gold ratio.
2. **Subtle-Baseless-Info-only positive weighting** (GPU, ~65 min, ~3.2GB VRAM). A single
   capped weight (e.g. sqrt(102.5)≈10, capped ≤15) applied to *only* this type's loss term,
   leaving the other three untouched — narrower and better-justified than the historical
   blanket 4-type weighting that already failed. Rollback: discard if any-F1 < 55.07%, macro
   F1 drops, or Subtle Baseless Info's ratio exceeds 10x.
3. **A joint-search variant with a per-class recall floor.** The current joint search purely
   maximizes the union metric, which is why it picked a worse-for-SBI-alone threshold (0.15)
   than per-type's own optimum (0.05, F1 21.61%). A constrained version (e.g. "maximize union
   F1 subject to every type's own recall ≥ X%") might recover more of that 21.61% without
   fully reverting to per-type's worse aggregate. Zero GPU cost, pure post-hoc computation on
   the existing cache — cheaper than either experiment above and worth trying first.
4. Do **not** revisit CRF's Subtle Conflict with more regularization tuning — already tested
   and shown not to be the bottleneck (§3, §8).
5. Do **not** repeat blanket 4-type positive weighting — already tried historically, already
   failed on the primary metric.

None of these are guaranteed to reach 80% detection F1. The observed low rare-class
signal does not establish a hard ceiling for this architecture or the primary any-F1.
