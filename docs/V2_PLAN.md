# V2 — bounded overnight development plan

The user authorized a new improvement round on 1 November 2026. The original
official-test report remains archived and is not used for new scoring. This
worktree is separate from the completed V1 delivery.

## Hypotheses and references

V1 development errors show fragmented spans and excessive false positives in
subtle types. These observations motivate two bounded interventions:

1. A focal-loss variant reduces the contribution of easily classified targets.
   This adapts the loss idea from [Lin et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html);
   that paper studies object detection, not RAGTruth, so it is motivation rather
   than evidence that this project will improve. Fix gamma=2 and alpha=0.75.
2. Word-pooled Transformer emissions and four independent BIO CRFs learn and
   decode contiguous spans while preserving cross-type overlap. Structured
   sequence prediction is motivated by [Lample et al., NAACL 2016](https://aclanthology.org/N16-1030/),
   which is an NER study, not a matching hallucination benchmark. Better word
   boundaries do not guarantee better semantic decisions.

## Fixed experiment budget

- Three fresh models: plain four-sigmoid BCE control, four-sigmoid focal variant,
  and word-pooled four-BIO-CRF variant. No additional family after these results.
- Each starts from `answerdotai/ModernBERT-base`, revision
  `8949b909ec900327062f0ebf497f51aef5e6f0c8`. V1 task weights are prohibited.
- Same V2 data, seed 42, max_length 2048, best-source-window training, inference
  over all windows, FP32, batch 1/accumulation 8, gradient checkpointing,
  encoder learning rate 2e-5, and three epochs. Save every epoch checkpoint.
- Structured-layer feasibility is checked with a small train-only GPU pilot.
  A failed memory/throughput preflight may cancel that family, with the reason
  recorded before full runs. No arbitrary extra experiment replaces it.
  A three-epoch runtime extrapolated from the fixed 32-train/16-dev pilot above
  five hours, or peak allocated GPU memory above 14 GiB, cancels the structured
  family. These are rough resource bounds, not performance estimates.
- V2 split: source-content groups, seed 20261101, task-stratified 10% of original
  training groups reserved for a new internal holdout; original validation is dev.
  Original official-test sources never enter any V2 data file.
- No class-based sampling or weights are imported from V1. No relabeling or
  merging of the four raw types is allowed.
- Plain/focal decoding uses the identical 15-value per-type threshold grid:
  `[0.005,0.01,0.02,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]`.
  Select each threshold by dev per-type character F1, ties higher threshold.
- Structured decoding uses the fixed per-type O-emission bias grid
  `[-8,-6,-4,-2,0,2,4,6,8]` and constrained Viterbi; select by dev per-type character F1,
  ties larger O bias. This controls false alarms without modifying weights.
  This coarse range is declared before any real pilot/training/dev scoring;
  the known rare training prior motivates covering larger logit margins.
- Within each family select checkpoint by dev four-type character macro F1,
  then any-character F1, then earlier epoch. Primary family follows the same
  rule, exact family ties prefer the BCE control, then focal, then structured.
- Freeze each family's selected weights and decoder, plus metric/data/code
  hashes in committed `docs/V2_FINAL_PROTOCOL.json`. Only then materialize the
  new internal holdout. Score each fixed family once on that cohort. No
  per-class holdout feedback drives further changes.

Metrics remain the original project's any-character F1, fixed-four-type macro
F1 and exact typed-span F1. New holdout scores compare only fresh V2 models
trained on the V2 training partition. V1 scores cannot be compared numerically
as though measured on the same unseen cohort. Existing V1 models trained on
the carved holdout and therefore cannot serve as its valid control.

Reports must show failed/negative trials as well as improvements, including any
remaining zero-F1 type, and record actual runtime, hardware and pilot limits.
