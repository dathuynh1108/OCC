# Lean NBD supervisor deck

`review.tex` is the editable LaTeX deck. It retains the theory figures and equations from the supplied Overleaf reference, groups Kernel SVDD and Deep SVDD together, and reports the completed RTX 5070 study with the two patch scores `A+D` and `B+D` only.

The verified fresh result is in `../../results/lean-nbd-5070-20260915/`: 15 MVTec AD classes × 3 seeds, 45 completed checkpoint-replayed runs. `class_delta_report.csv` records every improved and reduced class.

Build with `pdflatex review.tex` twice in this directory. The local host has Poppler for PDF review but no LaTeX compiler installed, so this package deliberately does not include a stale PDF built from an earlier deck.
