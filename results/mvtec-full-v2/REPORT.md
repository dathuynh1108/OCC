# Final NBD benchmark report

**Status: complete, with all 45 category/seed runs and 450 metric rows verified.**

NBD image AUROC: **78.79 ± 2.37%**.
This is a controlled shared-CNN-feature comparison on full MVTec AD, not a
reproduction of the official PatchCore, Deep SVDD or DROCC benchmark tables.
The previous WDBC/Wine/Digits pilot is separate historical evidence.

## Results

| Method | Image AUROC (%) | AP (%) | Test FPR (%) | Test TPR (%) |
|---|---:|---:|---:|---:|
| PatchScore_same_centers | 95.86 ± 0.23 | 98.67 ± 0.05 | 11.36 | 79.28 |
| PatchScore_byte_budget | 98.56 ± 0.21 | 99.59 ± 0.06 | 13.23 | 87.04 |
| RBF_SVDD | 78.20 ± 0.26 | 90.46 ± 0.17 | 6.86 | 45.26 |
| DeepSVDD_head | 81.85 ± 0.52 | 91.58 ± 0.32 | 6.88 | 49.56 |
| DROCC_head | 54.45 ± 5.43 | 75.73 ± 3.29 | 3.51 | 8.38 |
| Bubble_B | 82.70 ± 1.81 | 93.19 ± 0.85 | 11.78 | 54.04 |
| Bubble_BA | 82.80 ± 1.62 | 93.26 ± 0.75 | 11.07 | 53.80 |
| Bubble_BAD | 79.72 ± 2.46 | 91.53 ± 1.29 | 10.31 | 47.72 |
| Bubble_BAF | 82.82 ± 1.88 | 93.27 ± 0.91 | 10.89 | 53.20 |
| NBD | 78.79 ± 2.37 | 91.06 ± 1.44 | 10.81 | 47.27 |

For each seed, first average image AUROC equally over all 15 categories.
Then report the mean and sample SD (ddof=1) of the three seed macro means.
AP uses anomaly=1. Test FPR/TPR are macro means at alpha=0.05 normal-only thresholds.
All ten variants are reported; the NBD variant is fixed B+A+D+F, not the best ablation.

## Paired comparisons

- NBD_minus_same_centers: -17.07 ± 2.22 AUROC percentage points.
- NBD_minus_byte_budget: -19.77 ± 2.29 AUROC percentage points.

These are paired by category and seed. See `paired_deltas.csv` for all 45 pairs.

## Data, training and scope

- MVTec AD: 3629 original training normals; 467 normal + 1258 anomalous test images; all 15 categories.
- Every training image belongs to exactly one of fit, score calibration, threshold calibration. Seeds: 0, 1, 2. Full split paths and file hashes are saved.
- Frozen Wide ResNet-50-2 IMAGENET1K_V1, layer2+layer3, 1536-D, 28x28 patch grid, resize 256/crop 224. No global PCA. All methods use identical cached descriptors within each run.
- Deep SVDD: bias-free 1536→64→16 head, 50 complete AE epochs, 100 complete SVDD epochs, fixed nonzero center. Output variance is diagnostic, not a score-selection criterion.
- DROCC: 1536→64→1 head, 100 complete epochs, 10 normal-only warmup epochs, 50 projected ascent steps per adversarial batch. Every fit patch is used in every epoch. Neural head batch size is 8192, a declared shared-feature adaptation.
- RBF SVDD: actual dual QP on a declared 2048-point normal coreset. This cap is an adaptation; do not describe it as full-kernel training on every patch.
- NBD: 128 local-PCA bubbles of rank 16, 128-neighbor fits, bidirectional energy graph, projector discrepancy, all nonconstant diffusion modes, times 1/3/5. Full config is locked before any test prediction.
- Same-center control uses the exact NBD centers. Byte-budget control uses at most NBD geometric/graph array bytes; shared backbone and component calibration are excluded from that particular comparison.
- Top-1% mean image aggregation is shared by all variants. Do not retain the old slide's max-PatchScore equation for this table.
- Thresholds use held-out normal images only. Small calibration sets can require +inf at alpha=0.05 (notably toothbrush); retain that fact and the measured FPR/TPR. No test-selected threshold.
- Saved patch maps are not pixel-AUROC/AUPRO measurements. Do not present an evaluated segmentation claim.

## Reproducibility and verification

GPU: NVIDIA GeForce RTX 3090 Ti; Python 3.12.14; PyTorch 2.11.0+cu128; torchvision 0.26.0+cu128; CUDA 12.8.

Config SHA-256: `ae3ae4520f0a2ef57099601ecc0e4712b11396fd87eb3195033ae8394f3804b2`.
Protocol identity (config/source/data/backbone): `61f47ee191cbc2ac0422fd21cc88517b9965bfc2028fc632faabab4a9083b9c0`.
Backbone checkpoint SHA-256: `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`.

Each completed run includes model/optimizer/RNG checkpoints, epoch histories,
split paths, raw patch scores, image predictions, thresholds, QP diagnostics,
Deep SVDD variance and SHA-256 manifests. Every calibration/threshold/test patch
was rescored from reloaded checkpoints. AUROCs were independently recomputed
from saved prediction CSV files. See `verification.json` and per-run records.

The full float32 descriptor cache lives in RAM one category at a time to respect
GPU-host disk limits. Its hash, ordered input paths and extraction code are saved;
it can be regenerated from original images and the pinned backbone checkpoint.

## Files for slides

`summary.csv`, `category_summary.csv`, `per_category_per_seed.csv`, `split_counts.csv`,
`paired_deltas.csv`, `results_table.tex`, `plots/`, `SLIDE_UPDATE_HANDOFF.md`.

Primary sources and precise algorithm adaptations are documented in the repository README.

## Delivery verification

The separate replay regenerated all 15 full CNN caches and checked all 135 calibration/threshold/test groups. Maximum absolute score error was **0.0**. The independent standard-library CSV audit passed all 51,750 prediction rows, 450 metric rows and ten summary rows. The verified local backup preceded GPU shutdown; the provider reports `cur_state=stopped` and `actual_status=exited`. See the corresponding JSON evidence files.

For slide interpretation, read the complete `SLIDE_UPDATE_HANDOFF.md`, including the negative paired deltas, ablations, measured FPR versus the alpha target and weak head-training diagnostics. Prefer `plots/category_auroc_full_scale.png` to show scores below 50% without color saturation. The original plot and original generated report remain in the local output archive.
