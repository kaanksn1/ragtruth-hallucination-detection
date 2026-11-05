# Supplementary RNN and CNN token baselines — 5 November 2026

These baselines were implemented after grading feedback. They do not replace
the archived V1 study or imply that these experiments existed at first delivery.

Both take **only the response text**, using the shared regex word/punctuation
tokenizer and its original Unicode character offsets. The source, generator,
task category, IDs and annotation metadata never enter the encoder. Vocabulary
casefolding changes token lookup only; original text/offsets are untouched.
Vocabulary counts come only from the fixed V1 training set (minimum frequency
2; at most 20,000 entries including PAD/UNK). Validation words do not expand it.

- **RNN:** 128-dimensional learned word embeddings, one bidirectional vanilla
  tanh RNN (64 hidden units per direction), dropout 0.1, tokenwise linear head.
- **CNN:** the same embeddings, parallel 1D convolutions of widths 3 and 5
  (64 channels each), ReLU, dropout 0.1, tokenwise linear head. No sentence pooling.

Four sigmoid outputs and masked unweighted binary cross-entropy preserve
cross-type overlap. A word is positive for every annotation type sharing at
least one character with it. Padding does not contribute to the loss.
Every answer is split into nonoverlapping chunks of at most 256 words; **no
answer words are truncated**. Encoders reset at chunk boundaries, limiting
context. Predictions are reassembled before adjacent positive word runs are
converted to spans, including intervening original whitespace. Same-type
touching/overlapping annotations cannot always be represented as distinct
boundaries, so exact span performance is reported separately from character F1.

Default protocol: seed 42; three epochs; batch 32; Adam learning rate 0.001;
gradient clipping 1.0. All epochs are saved. Per epoch, one **global** threshold
from `[0.01, 0.05, 0.1, 0.2, 0.3, 0.5]` is selected by validation typed character
macro F1, then any-character F1; exact ties retain the earlier grid threshold
and epoch. The selected checkpoint is reloaded and its validation scores must
replay before a final prediction file/summary is written. These settings are
declared before running supplementary experiments and do not use test results.

Run from repository root, with the original prepared data and an existing
PyTorch CPU/CUDA/ROCm environment:

```bash
python scripts/baseline_neural.py --model rnn --output-dir outputs/supplement_20261105/rnn --device auto
python scripts/baseline_neural.py --model cnn --output-dir outputs/supplement_20261105/cnn --device auto
```

Outputs use a new directory and cannot overwrite earlier runs. `model.json`
contains the selected checkpoint hash, vocabulary, input/data identities,
hyperparameters, implementation hashes and environment. `epoch-N.json` records
all declared validation threshold candidates; `summary.json` records selected
metrics and time; `validation_predictions.jsonl` uses the common ID/span schema.
Model weights are local run artifacts; the repository's supplementary results
must be generated from actual completed runs, not inferred from implementation.

The runner reads only train/validation. It has no official-test option. This
supplement is a **validation benchmark** on the same cohort as archived V1
validation results. Comparing these response-only encoders with a source-aware
Transformer combines input and architecture differences; it is not a controlled
architecture-only ablation. Use the response-only CRF as a matching-input control.
