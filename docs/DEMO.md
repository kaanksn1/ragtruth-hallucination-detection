# Offline prediction viewer

The demo is one self-contained HTML file. It shows **saved validation outputs**,
not live inference or newly generated answers. It displays source-relative model
predictions separately from human dataset annotations; a highlight is not a claim
of universal factual truth. There is no server, external JavaScript, CDN or network
request. Open the generated HTML file in a browser.

The standalone file embeds RAGTruth's original Particle Media MIT notice in
the attribution footer; the same notice is kept in `docs/DATA_LICENSE.txt`.

## Build

From this repository's root, using any Python 3.12 environment:

```powershell
python scripts/build_demo.py --predictions Transformer=outputs/runs/20261031T110122793234Z-transformer-train/validation_predictions.jsonl --output artifacts/demo.html
```

Add another saved model with a distinct display name:

```powershell
python scripts/build_demo.py --predictions Transformer=outputs/transformer/validation_predictions.jsonl --predictions CRF=outputs/crf/validation_predictions.jsonl --output artifacts/comparison-demo.html
```

The second command uses illustrative paths; supply the actual CRF/Transformer
run paths. `--records` defaults to `data/processed/validation.jsonl`. All rows
must belong to validation. Only `quality=good` responses are displayed, matching
the training/evaluation policy. Official test or mixed-split files are rejected,
including test rows that would otherwise be excluded by a quality/display filter.

Prediction IDs must exactly match every good validation response, even with
`--max-examples 100` (an optional **display-only** limit). Missing/extra/duplicate
IDs, invalid character offsets and unknown types stop the build. Existing output
files are never overwritten. Artifacts stay under ignored `artifacts/` or `outputs/`.

## Inspect

- Select a model, task and response, or search IDs/source/answer text.
- Filter to examples with/without predicted spans or exact typed span mismatches.
- Read the original source and compare predicted highlights with the separately
  labeled human annotations. Hide annotations to inspect predictions alone.
- Four colors represent the four raw annotation types. Cross-type overlaps use
  stripes; hovering shows types and `[start,end)` character offsets.
- Navigation and filters change only the view. They do not recompute predictions
  or report a metric for the complete dataset.

The selected example/model is stored in the URL fragment, for example
`demo.html#id=123&model=Transformer`. This is a same-file bookmark, not a
network request. Invalid/unavailable IDs fall back to the first matching example.

The exact-match filter compares sets of `(start,end,label_type)` triples;
identical duplicate annotations do not change that filter. Original span counts
remain visible and overlapping spans are preserved. A partial overlap may receive
character-level credit while still appearing in the mismatch filter.

Unicode segmentation happens in Python using the dataset's codepoint offsets;
emoji do not shift positions through JavaScript UTF-16 indexing. Dataset text and
model names are rendered with text nodes. Embedded JSON escapes script-closing
markup, HTML delimiters and Unicode line separators. Annotation metadata/comments
are not embedded in the artifact. Input file SHA-256 values are embedded for traceability.

Verification: `python -m unittest discover -s tests -p test_demo.py -v`.

## Local verification — 31 October 2026

Seven demo tests passed, including malformed/missing IDs, Unicode overlap
segmentation, script-closing text and overwrite protection. JavaScript syntax
passed `node --check demo/viewer.js`.

The actual 2,240-response Transformer artifact was checked in an isolated
headless Edge browser. Navigation, task/search filters, no-results state,
human-annotation toggle, positive highlights, striped cross-type overlaps and
same-file bookmarks passed without JavaScript exceptions. At a real 430-pixel
viewport the document's scroll width was 430 pixels (no horizontal overflow).
Desktop and mobile screenshots were visually inspected.

Local QA evidence is ignored by Git: `artifacts/demo-browser-qa.json`,
`artifacts/demo-browser-wide.png`, `artifacts/demo-browser-mobile.png` and
`artifacts/validation-demo-transformer-v2.html`. The comparison artifact should
be rebuilt from the final selected prediction files before delivery.
