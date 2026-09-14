# Slide update handoff — measured 14 September 2026

Use the updated `slides/paper-faithful-review/review.tex` and `review.pdf`.
The slide text is intentionally short; this report and CSVs hold the detail.

## Main message

NBD B+A+D+F: **78.66 ± 1.99% AUROC**, **90.99 ± 0.77% AP** on the common MVTec benchmark.
Keep NBD as the declared model even when an ablation/baseline scores higher.

| Method | Image AUROC (%) | AP (%) | FPR (%) | TPR (%) |
| --- | --- | --- | --- | --- |
| PatchScore: same centers | 95.66 ± 0.21 | 98.41 ± 0.10 | 11.40 | 79.97 |
| PatchScore: byte budget | 98.96 ± 0.15 | 99.70 ± 0.05 | 13.71 | 89.30 |
| RBF SVDD (feature adaptation) | 75.56 ± 0.28 | 88.92 ± 0.19 | 7.27 | 41.69 |
| Deep SVDD head (15 epochs) | 87.03 ± 0.54 | 94.12 ± 0.33 | 4.92 | 58.05 |
| DROCC head (15 epochs) | 49.46 ± 2.55 | 72.88 ± 1.44 | 4.46 | 8.43 |
| B | 79.91 ± 0.63 | 91.99 ± 0.88 | 12.78 | 49.63 |
| B+A | 80.22 ± 0.56 | 92.20 ± 0.68 | 12.29 | 51.55 |
| B+A+D | 77.74 ± 1.21 | 90.65 ± 0.22 | 11.49 | 46.96 |
| B+A+F | 81.76 ± 0.66 | 92.75 ± 0.77 | 10.85 | 51.04 |
| NBD: B+A+D+F | 78.66 ± 1.99 | 90.99 ± 0.77 | 11.16 | 46.73 |
| PatchCore: same FIT, top 1% | 98.96 ± 0.15 | 99.71 ± 0.04 | 13.35 | 89.28 |

## What changed in the deck

- Use the new1024-D author descriptor, preserve the author's startup BN behavior.
- Show one common15-category ×3-seed MVTec table with11 explicitly named variants.
- Put MNIST/CIFAR native results only on their own method reproduction slides.
- Label AE 5/SVDD 12, nativeDROCC5 and sharedhead15 epochs visibly; do not say full
  historical-paper reproduction. Keep original full commands in the rerun guide.
- Keep best-test-selected DROCC distinct from fixed-final results.
- Explain DROCC negatives briefly: Gaussian start, gradient ascent, radius projection,
  anomaly training label. Native CIFAR gamma 1 gives a sphere; feature adaptation differs.
- Show exact coverage; Tax2004 original numeric target stays unreproduced.
- Keep older v2 numbers as historical evidence in their original directory.

All selected matrix rows completed.

## Editing rules

Use concise English captions and tables, one main point per slide. Do not paste
this report as slide prose. Preserve source citations and uncertainty; do not
replace missing cells, low scores or final NBD with a stronger ablation.

Sources of numbers: common_summary.csv, common_category_summary.csv,
common_paired_deltas.csv, native_summary.csv, native_per_class_per_seed.csv,
paper_vs_measured_summary.csv and paper_vs_measured_per_class.csv.
Independent check: independent_metric_audit.json. Exact selected configuration:
run_plan_budgeted.json. Full detail: REPRODUCTION_REPORT.md and SOURCE_AUDIT.md.

Delivery: code, tables, raw predictions and final PDF/TeX are on GitHub `main`.
The `reproduction-2026-09-14` release contains the final Overleaf ZIP and incremental
checkpoint archives. See `docs/reproduction/BACKUP_RESTORE.md` for the hybrid backup
scope and `execution_closure.json` for verified instance deletion.
