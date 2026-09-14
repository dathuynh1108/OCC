# Updated OCC reproduction handoff

The supplied task specification is preserved in docs/reproduction/REQUEST.md. This file records the measured delivery; it does not replace the original evidence.

NBD B+A+D+F: **78.66 ± 1.99% image AUROC**, **90.99 ± 0.77% AP** on all 15 MVTec categories × 3 seeds.

- [Complete slide-update handoff](results/reproduction-2026-09-14/SLIDE_UPDATE_HANDOFF.md)
- [Measured report and protocol differences](results/reproduction-2026-09-14/REPRODUCTION_REPORT.md)
- [Paper versus measured native results](results/reproduction-2026-09-14/paper_vs_measured_summary.csv)
- [Updated slide PDF](slides/paper-faithful-review/review.pdf) / [Overleaf source](slides/paper-faithful-review/review.tex)
- [Exact rerun commands](docs/reproduction/RERUN.md)

The user selected fewer epochs and about two hours remaining: native Deep AE5/SVDD12, native DROCC5, common heads15. These are source-audited shortened replications. Historical Theano numerical parity and Tax2004's exact Iris table remain unverified/unreproduced. Native datasets and the common MVTec comparison are separate. All weak scores and the declared final NBD remain visible.

All selected matrix rows completed.

Fitted-state backup and restore: [instructions](docs/reproduction/BACKUP_RESTORE.md).
The [GitHub Release](https://github.com/dathuynh1108/OCC/releases/tag/reproduction-2026-09-14)
contains the final Overleaf ZIP and ten incremental checkpoint archives.
