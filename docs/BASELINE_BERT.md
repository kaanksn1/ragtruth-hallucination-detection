# Supplementary classical BERT baseline — 5 November 2026

This is a **post-feedback addition**, not a model present in the original
submission. It does not change the frozen V1 scientific decision or its test
scores. The separate original BERT encoder requested in the feedback is
`google-bert/bert-base-uncased`, pinned to
`86b5e0934494bd15c9632b12f734a8a67f723594`.

Model source and pretrained-model license (Apache-2.0):
<https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594>.
The project MIT license does not replace the pretrained model's license.
Pretrained and finetuned model weights are not stored in Git.

## Fixed settings and scope

`configs/supplementary/bert.json` reuses the tested shared four-sigmoid
token-classification implementation and original grouped train/validation
split (12,702 / 2,240 eligible responses). Inputs contain response and source;
annotation metadata never becomes input. Independent outputs retain overlapping
types. Token predictions map to original Unicode character offsets.

This resource-bounded baseline uses **one full training epoch**, FP32, seed42,
AdamW at2e-5, effective batch8, maximum512 tokens. Training selects the best
input-only source window per response chunk; validation aggregates all windows
with the same `min` rule as the original Transformer pipeline. Long answers and
sources are windowed, not silently dropped. A one-epoch baseline is not an
equally tuned or compute-matched comparison against the original three-epoch,
2,048-token ModernBERT experiment. Report these differences when comparing.

The base threshold is0.5. A separately recorded validation-only per-type grid
may be used with the historical macro-first calibration rule. There is no
official test inference/evaluation or tuning in this supplementary run.

```bash
python scripts/transformer.py train --config configs/supplementary/bert.json --output outputs/supplementary_bert_20261105 --device cuda
```

Dependency versions are those already pinned in `requirements-transformer.txt`;
install the appropriate existing CPU/CUDA/ROCm Torch build first.
The training command writes the actual configuration, input/code/checkpoint
hashes, train and validation losses, environment, runtime and complete validation
predictions. `scripts/calibrate_transformer.py` saves checkpoint-bound
calibration and its producer identity when run with a separately declared config.
