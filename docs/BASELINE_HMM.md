# Supplementary supervised HMM baseline

Added after the original evaluation, in response to grading feedback. It is a
new **validation-only supplementary benchmark**, not part of the original
frozen official-test study and not evidence of pre-submission work.

`scripts/baseline_hmm.py` fits four independent supervised three-state (`O/B/I`)
Hidden Markov Models. Each type has its own emission, transition, start and
end probabilities; cross-type overlapping spans are retained. Exact response
word offsets come from the existing shared tokenizer. This is a **response-only**
control; it cannot directly verify claims against a source. Its input therefore
differs from the source-aware CRF and ModernBERT, so score differences must not
be attributed to architecture alone.

Vocabulary and all probability estimates use only the original source-grouped
train split, filtered to `quality=good`. Words seen fewer than twice become
fixed unknown-word categories (number/uppercase/title/punctuation/other).
Lower-casing affects observations only; response text and output offsets stay
unchanged. Additive smoothing is fixed in advance at `alpha=0.1`. Viterbi uses
BIO structural zeros (no starting `I`, no `O→I`) and explicit end probabilities.
No hyperparameter grid, validation vocabulary adaptation or official-test
access is performed. Annotation metadata and source text are never features.

From the repository root, with Python 3.12 (standard library only):

```powershell
python scripts/baseline_hmm.py train --output outputs/supplementary_hmm
python scripts/baseline_hmm.py predict --model outputs/supplementary_hmm/model.json --output outputs/supplementary_hmm_replay
```

The output directory must be new. `model.json` contains the vocabulary, count
matrices and fixed settings; `validation_predictions.jsonl` contains every good
validation response in the common prediction schema. `summary.json` records
common project metrics, complete split/quality identities, input/model/code
hashes, runtime and the supplementary status. Replay uses the serialized model
without refitting. Scores must be reported only after a real full-data run.

Verification: `python -m unittest discover -s tests -p test_baseline_hmm.py -v`.
