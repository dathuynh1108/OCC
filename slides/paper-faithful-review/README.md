# OCC / Normality Bubble Diffusion — sources and rerun handoff

## Files
- `review.pdf`: complete 26-slide English deck.
- `review.tex` + `assets/`: editable Beamer / Boadilla / 16:9 sources.
- `handoff/CODEX_REPRODUCE.md`: full task to give Codex on the local GPU computer.
- `handoff/source_lock_reviewed.json`: author repositories and reviewed commits.
- `handoff/SOURCE_AUDIT_NOTES.md`: differences confirmed before preparing the prompt.
- `results/` and `configs/full.json`: unchanged **historical v2** result/config snapshots.

## Compile
Run from this folder:

```sh
pdflatex -interaction=nonstopmode -halt-on-error review.tex
pdflatex -interaction=nonstopmode -halt-on-error review.tex
```

No external fonts or image downloads are needed. `assets/` must remain beside the
TEX file. The source references and GitHub links in the PDF are clickable.

## What changed
Two pages before the existing method slides explain the native feature extractors
and the exact PatchCore author-code extraction path. Original-repository GitHub
links now appear next to each method and in a new final source/rerun page.

The old result tables, NBD formulae, component-scaling placement and the Deep SVDD
collapse page are retained. Baseline-only results still precede the NBD section.

**No benchmark was rerun.** The 1536-D v2 MVTec results have not been relabeled as
1024-D PatchCore-author-code results. Native-paper runs are pending execution.
The Codex task requires a source audit and numerical parity before full training,
and separates author-code/paper discrepancies from controlled adaptations.

## Give this to Codex
Copy `handoff/` into the local `dathuynh1108/OCC` checkout, then say:

> Read handoff/CODEX_REPRODUCE.md and its source lock. Audit the pinned author code,
> implement the runners and verification, then execute the native-source and
> controlled-NBD tracks on my local NVIDIA GPU. Preserve mvtec-full-v2. Do not
> substitute frozen heads for native CNN methods or silently change protocols.
> Finish with measured results and a slide-update handoff, not only a plan.

The handoff is a task specification, **not a claim that the new runners already
exist or have passed tests**. It requires Codex to write and test them locally.
