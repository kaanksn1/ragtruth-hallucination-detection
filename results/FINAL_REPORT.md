# RAGTruth technical completion report

Scope: code, experiments, evaluation, technical error analysis and the offline demo. Slides/presentation are separate. Original data, CRF and Transformer implementation authorship is preserved; subsequent integration/completion work belongs to the repository owner, without fabricated personal contributions.

## Task, data and integrity

Given supplied context and an LLM response, locate annotated response spans and identify four raw hallucination types. Unsupported content is not necessarily false in the world. Only source/response text enters the model; IDs, generator/task metadata and annotation comments are excluded. `implicit_true` and `due_to_null` annotations remain in the targets.

References: pinned [RAGTruth source](https://github.com/ParticleMedia/RAGTruth/tree/c103204b9ce28d6bbad859304bf30de72b8ed8fe), [MIT license](https://github.com/ParticleMedia/RAGTruth/blob/c103204b9ce28d6bbad859304bf30de72b8ed8fe/LICENSE) and [ACL paper](https://aclanthology.org/2024.acl-long.585/). Raw inputs are checksum-verified. This project does not reproduce the paper's generative detector or claim scorer/SOTA equivalence.

| Cohort | Raw responses | Good responses scored/used | Quality exclusions |
|---|---:|---:|---|
| train | 12,828 | 12,702 | {'truncated': 25, 'incorrect_refusal': 101} |
| validation | 2,262 | 2,240 | {'incorrect_refusal': 19, 'truncated': 3} |
| Reserved test | 2,700 | 2,675 | {'incorrect_refusal': 24, 'truncated': 1} |

The fixed manifest has **2,138/377 train/validation source IDs** and **2,136/377 equal-content groups**. Source-ID, group-ID and response-ID overlaps are zero; exact and normalized source-text overlaps are zero. Shared answer boilerplate occurs 83 times in train and 20 in validation; all shared gold labels empty: **true**. The answer 'Unable to answer based on given passages.' is retained across different contexts; we do not claim zero answer overlap or silently remove rows.

| Type | Train spans | Validation spans | Positive train word tokens (%) |
|---|---:|---:|---:|
| Evident Conflict | 4017 | 670 | 1.4955 |
| Subtle Conflict | 160 | 25 | 0.0696 |
| Evident Baseless Info | 4654 | 813 | 3.0555 |
| Subtle Baseless Info | 2006 | 348 | 0.9661 |

## Models and bounded development

**CRF:** four independent O/B/I linear-chain CRFs, with word/shape/neighbour features plus lexical source presence/novelty. Three regularization settings and Viterbi/posterior decoders are selected by validation any-character F1. The response-only control keeps the selected setting/decoder. Choice: `c1_0.1_c2_1`, `posterior_0.15`; runtime 25.0 minutes. At the 150-iteration L-BFGS cap, 12/12 grid taggers and 4/4 control taggers stopped at the cap; convergence is not asserted.

**Transformer:** pretrained ModernBERT-base plus dropout/four independent sigmoid logits. Multi-hot targets retain cross-type overlaps; masked BCE supervises only answer tokens intersected with attention_mask. Both training families use the same source-group split, seed 42, three epochs, FP32/SDPA, batch 1, accumulation 8, gradient checkpointing, lr 2e-5 and 2048-token cap. About 99.1% of development responses fit one window. Training selects the lexical best source window per answer chunk without gold; inference evaluates all windows, takes per-type minimum scores, then averages overlapping answer chunks.

The unweighted baseline originally chose epoch 2 by validation BCE and used threshold 0.5. Its bounded calibration compares saved epochs 1/2 with 11 thresholds per type. One additional train-only weighting trial uses `min(50,max(1,sqrt(negative/positive)))` over response-token counts. Its 15-threshold grid evaluates only saved improving checkpoints (one to three). Each type maximizes validation character F1, ties prefer the higher threshold; checkpoint ties prefer higher any F1, then earlier declared candidate. These are independent per-type searches, not exhaustive joint grids.

| Type | Actual positive-loss weight |
|---|---:|
| Evident Conflict | 8.1815 |
| Subtle Conflict | 38.2192 |
| Evident Baseless Info | 5.6957 |
| Subtle Baseless Info | 10.2438 |

| Validation development step | Any character F1 (%) | Typed macro F1 (%) |
|---|---:|---:|
| Unweighted, original 0.5 | 44.642 | 22.444 |
| Unweighted, calibrated | 54.053 | 32.007 |
| Weighted trial, calibrated | 51.545 | 29.423 |

The frozen primary `transformer` is **unweighted**, selected by validation typed macro F1, then any F1, with exact family ties retaining the unweighted baseline. The weighted trial has one saved checkpoint candidate. No weighted improvement is assumed from the training loss alone. Primary checkpoint and thresholds:

`outputs/runs/20261031T110122793234Z-transformer-train/checkpoints/epoch-2`

| Type | Frozen threshold |
|---|---:|
| Evident Conflict | 0.15 |
| Subtle Conflict | 0.01 |
| Evident Baseless Info | 0.3 |
| Subtle Baseless Info | 0.05 |

## Common metrics and results

Metric version: `ragtruth-project-eval-v1`. Character micro P/R/F1 counts original Unicode codepoints including whitespace; any uses the union across types, while typed macro averages all four fixed types. Exact typed spans require original boundaries and type. Identical duplicates count once, but overlapping/touching raw spans are not merged for exact matching. Empty denominators yield zero. All table metrics below are percentages, not token accuracy; [full definitions](../docs/EVALUATION.md).

### Validation (used for selection)

Development scores include decoder/checkpoint/family selection and are not independent performance estimates.

| Model | Any P (%) | Any R (%) | Any char F1 (%) | Typed macro F1 (%) | Exact typed span F1 (%) |
|---|---:|---:|---:|---:|---:|
| crf_source | 36.52 | 42.02 | 39.08 | 19.99 | 6.41 |
| crf_response_only | 21.46 | 25.01 | 23.10 | 13.13 | 5.28 |
| transformer | 50.76 | 57.80 | 54.05 | 32.01 | 7.00 |
| transformer_default | 72.69 | 32.21 | 44.64 | 22.44 | 11.92 |
| weighted_trial | 44.07 | 62.07 | 51.54 | 29.42 | 4.65 |

### Reserved official test

The model set is fixed by [FINAL_PROTOCOL.json](../docs/FINAL_PROTOCOL.json). Final-test jobs require committed scientific decisions and exact model/code/data hashes before prediction/evaluation. These already-scored aggregates record frozen-job gold provenance; this report neither opens test records nor performs post-test model selection.

| Model | Any P (%) | Any R (%) | Any char F1 (%) | Typed macro F1 (%) | Exact typed span F1 (%) |
|---|---:|---:|---:|---:|---:|
| crf_source | 32.60 | 38.83 | 35.44 | 19.76 | 6.67 |
| crf_response_only | 18.32 | 23.90 | 20.74 | 11.95 | 6.15 |
| transformer | 48.28 | 56.49 | 52.06 | 29.84 | 6.78 |

| Model | Evident Conflict F1 (%) | Subtle Conflict F1 (%) | Evident Baseless F1 (%) | Subtle Baseless F1 (%) |
|---|---:|---:|---:|---:|
| crf_source | 27.15 | 0.00 | 33.18 | 18.71 |
| crf_response_only | 27.68 | 0.00 | 14.62 | 5.48 |
| transformer | 47.46 | 0.00 | 54.50 | 17.40 |

## Interpretation, limitations and reproduction

Observed test results: Transformer any-character F1 is 52.06%, compared with 35.44% for source-aware CRF and 20.74% for response-only CRF. All three have zero Subtle Conflict F1; this type appears in 15 test responses. Transformer Subtle Baseless Info F1 (17.40%) is below source-aware CRF (18.71%). Exact typed-span F1 remains 6.78% for Transformer. On validation, threshold selection improves character macro F1 but reduces exact typed-span F1 from 11.92% to 7.00%; the declared selection objective was character macro F1. [Turkish interpretation](BULGULAR_TR.md) explains these tradeoffs.

Subtle Conflict has little support; the original 0.5 decoder produced no validation prediction for it. Threshold calibration changes the precision/recall operating point, not reasoning ability. Rare-type reliability remains limited. Weighted/unweighted trials also have different declared threshold grids; their validation gap is not attributed solely to loss weighting. CRF versus Transformer differences combine architecture and input features; only the two CRF variants directly control source features.

Best-window training can miss evidence; minimum-over-window inference can suppress real errors. Word/subword edges, whitespace bridging and same-type unions limit exact-boundary recovery. A single fixed seed and bounded trials do not support multi-seed or statistical confidence claims. Validation error examples follow declared categories and are not a representative random sample.

[RUNNING.md](../docs/RUNNING.md), [CALIBRATION.md](../docs/CALIBRATION.md) and [DEMO.md](../docs/DEMO.md) document execution. The offline demo shows saved validation predictions beside separately labeled human annotations, without live inference or factuality guarantees. Raw data, full predictions and weights stay ignored; aggregate reports, scripts and fixed manifests are versioned.

Artifacts: [validation error examples](error_analysis.md), [test comparison](test_comparison.md), [test plots](figures/test/), and [delivery guide](../docs/TECHNICAL_HANDOFF.md). Tables/calibration/support sections were assembled from fixed aggregates; the observed-results interpretation was added after scoring without changing any model decision.

```powershell
python -m unittest discover -s tests -v
python scripts/run_job.py configs/jobs/crf_validation.json
python scripts/run_job.py configs/jobs/transformer_train.json
python scripts/run_job.py configs/jobs/transformer_weighted_train.json
```

Exact frozen inference commands are archived under `configs/final/`; test results must not drive retuning. The report builder consumes only the supplied scored summaries, calibration metadata and frozen decision. It refuses wrong splits/model sets and existing output files.

Frozen decision SHA-256: `19c1c169f6563793fc7dae47856babc9915f98b8850df57cee4917efa39422f3`.
