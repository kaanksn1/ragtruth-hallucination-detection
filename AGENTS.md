# Project 2 — RAGTruth scope and ownership

The user selected RAGTruth and assigned a five-person workflow on 2026-10-29.
All full training and model-prediction runs, including the CPU CRF baseline,
must run on the repository owner's PC. Person 5 presents.

- Person 1 (owner): GPU setup/preflight and pilot execution, central run queue,
  reproducibility logs, integration, common CLI, and a small local demo.
- Person 2: data analysis, grouped splits, shared data/offset contract,
  character/token alignment, and window-to-response mapping.
- Person 3: CRF baseline implementation/configuration/analysis; optional
  response-only experiment configuration and analysis using person 4's code.
- Person 4: context-aware small Transformer implementation, overlap-capable
  targets/outputs, prediction decoding, and own model analysis.
- Person 5: common evaluation, 2–3 comparison plots, slide assembly, presentation.

Each member owns their code/tests, configuration, own findings/error examples,
and 1–2 slide drafts with speaker notes. Running another member's training
does not transfer authorship of their model to the owner. Each contributes
through a separate branch and PR. Do not fabricate personal contribution reports.

When assisting the owner, implement the owner's scope and minimum shared
infrastructure; keep teammates' implementation/experiments pending until they
contribute them or the user explicitly reassigns work. Read-only inspection and
reviews are allowed. Full details are in `docs/TEAM.md`.

Preserve source IDs, original response text/character offsets, four annotation
types, overlapping spans, and annotation metadata. Annotation-only fields must
not become model input. Keep each source group on one side of train/validation.
Use validation for decisions. The final scientific decision is frozen in
`docs/FINAL_PROTOCOL.json`, with exact job/protocol pairs under `configs/final/`.
Official test evaluation has completed through that frozen pipeline. Do not
retune models, thresholds, features, splits or selection criteria against test
results. `docs/PROTOCOL_DRAFT.md` remains an archived design draft.

Do not alter BANKING77 data, protocols, teammate implementations, or existing
personal contribution reports as part of this new project.

For changes to this project's integrity-inspection code, run:
`python -m unittest discover -s tests -v` from the standalone RAGTruth repository
root. Documentation-only task assignments do not require retraining models.

## Explicit bug-fix assignment — 2026-10-30

The user assigned the owner to fix person 2's `labels_multi` batch conversion
bug, including regression tests and its data-contract documentation. This does
not reassign the rest of person 2's data work or person 4's model implementation.

## Explicit CRF review/fix assignment — 2026-10-31

The user assigned review, fixes and necessary additions to person 3's new CRF
delivery, including its tests, run contracts and integration documentation.
Personal analysis/slides remain with person 3; person 4's implementation and
person 5's common evaluation remain with their owners.

## Explicit Transformer review/fix assignment — 2026-10-31

The user assigned review and bug fixes for person 4's pushed Transformer
delivery, including regression tests and central-run integration. This does
not reassign person 4's model analysis/slides or person 5's common evaluation.

## Explicit completion assignment — 2026-10-31

The user explicitly assigned ALL remaining technical work except slides and
presentation to the owner/assistant. This includes full central model runs,
validation calibration and justified model fixes, common evaluation/comparison
plots, technical error/data analysis, the local demo, reproducibility reports,
and committed final-test protocols followed by the reserved official test.
Do not write personal contribution reports or attribute this completion work
to teammates. Existing implementation authorship remains unchanged. Slides
and the presentation itself remain outside this assignment.
