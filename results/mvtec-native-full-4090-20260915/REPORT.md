# MVTec AD: full image-model adaptation

All 15 MVTec AD categories, 3,629 normal training images and 1,725 test images; seed 0.
Deep SVDD and DROCC are author-image-model adaptations to MVTec, not published MVTec benchmarks from those papers.

## Locked full schedules

- Deep SVDD: unchanged MVTec preprocessing (per-image L1 GCN, normal-train scalar min/max), CIFAR CNN/AE, AE 350 epochs with milestone 250, then 150 SVDD epochs with milestone 50; both objectives; fixed-final metric.
- DROCC: unchanged author-CIFAR normalization and MVTec adaptation parameters (Adam 0.001, batch 128, radius 0.2, gamma 2, mu 1, effective 50 ascent steps, projection every 10, no CE-only warmup); 100 epochs; source LR thresholds at 40% and 80%; fixed-final metric is primary.
- RBF SVDD has no epoch budget and is reused from the verified identical light export. PatchCore and NBD predictions are reused unchanged.

## Full versus light (category-macro image metrics, %)

| Method | Light AUROC | Full AUROC | Delta pp | Light AP | Full AP | Delta pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSVDD one-class | 57.61 | 79.43 | +21.82 | 80.43 | 91.03 | +10.60 |
| DeepSVDD soft-boundary | 57.96 | 76.60 | +18.64 | 80.42 | 89.82 | +9.40 |
| DROCC fixed-final | 46.47 | 43.77 | -2.70 | 74.11 | 73.13 | -0.98 |

## DROCC selection export (full schedule)

| Export | AUROC | AP |
| --- | ---: | ---: |
| Fixed final epoch (comparison) | 43.77 | 73.13 |
| Source test-selected diagnostic | 71.56 | 86.80 |
| Diagnostic minus fixed-final (pp) | +27.79 | +13.67 |

## Per-category full results (AUROC / AP, %)

| Category | Deep one-class | Deep soft-boundary | DROCC fixed-final | DROCC test-selected diagnostic | Deep s | DROCC s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 95.16 / 98.52 | 92.38 / 97.40 | 36.59 / 73.26 | 73.10 / 91.73 | 16.3 | 44.7 |
| cable | 83.58 / 89.23 | 79.74 / 86.85 | 62.86 / 78.79 | 71.29 / 82.77 | 15.5 | 44.2 |
| capsule | 58.88 / 88.17 | 50.82 / 86.22 | 30.16 / 74.06 | 48.82 / 80.79 | 15.1 | 41.5 |
| carpet | 67.26 / 88.38 | 62.12 / 86.57 | 17.62 / 64.94 | 71.87 / 88.80 | 15.3 | 69.0 |
| grid | 75.10 / 88.54 | 70.59 / 85.71 | 36.34 / 70.32 | 87.55 / 90.81 | 15.7 | 66.8 |
| hazelnut | 85.75 / 91.80 | 82.71 / 89.77 | 56.43 / 73.12 | 73.00 / 85.10 | 16.8 | 79.2 |
| leather | 98.47 / 99.52 | 98.34 / 99.48 | 50.82 / 81.89 | 83.29 / 94.97 | 15.5 | 46.0 |
| metal_nut | 69.50 / 90.34 | 64.76 / 90.03 | 53.86 / 84.35 | 74.68 / 92.12 | 15.3 | 44.9 |
| pill | 74.09 / 93.83 | 70.19 / 93.17 | 38.00 / 78.17 | 62.58 / 88.58 | 15.7 | 68.1 |
| screw | 51.30 / 75.89 | 49.68 / 73.16 | 23.10 / 59.73 | 52.80 / 77.80 | 16.0 | 67.0 |
| tile | 70.31 / 87.80 | 75.79 / 90.03 | 44.52 / 75.63 | 69.37 / 83.06 | 15.4 | 41.2 |
| toothbrush | 98.89 / 99.61 | 97.78 / 99.30 | 79.72 / 92.37 | 82.22 / 92.16 | 10.6 | 24.1 |
| transistor | 82.04 / 81.56 | 77.04 / 78.54 | 44.38 / 42.48 | 73.17 / 70.73 | 15.2 | 45.0 |
| wood | 93.77 / 97.84 | 94.04 / 97.89 | 38.33 / 71.38 | 86.23 / 95.75 | 15.5 | 43.3 |
| zipper | 87.39 / 94.40 | 83.04 / 93.26 | 43.80 / 76.45 | 63.42 / 86.78 | 15.4 | 39.7 |

## Runtime and convergence checks

- Deep SVDD wall time: 229.4 s total, 15.3 s/category (AE shared by both objectives).
- DROCC wall time: 764.6 s total, 51.0 s/category.
- Every 350-epoch AE, 150-epoch one-class/soft-boundary phase, and 100-epoch DROCC history has the required length and finite tracked loss, gradient, LR and radius values.
- Every saved final and source-selected DROCC checkpoint was reloaded for score replay before its result was admitted. `comparison_verification.json` independently recomputes AUROC/AP and checks all test IDs and labels against the PatchCore reference.
- The first fresh Deep SVDD fixture attempt is retained under `evidence/deep_fixture`; it stopped on a relative-output-path logging error before benchmark training. The runner was corrected to resolve the explicit fixture path, then the separate `evidence/deep_fixture_v2` fixture passed. No metric was reused from the failed fixture.

## Provenance

- Full protocol SHA256: `796cc7e796a0fba4cc227b6a6bf7fb1f6dcfc37f7d752344ce4b7853ae7ad7b6`
- Prediction/metric verification SHA256: `492ed7b2f8fe1c5618df75debf58da3bb50d5451eca1d1092f936fe87123a316`
- Input cache manifest SHA256: `9c469576ab86e93c7104530d08d4aef3862360d28d0c2154d69dc6911edc5829`
- Source lock SHA256: `809666ff44691270b772d3a0228e584e98c4289265bb04a88ffe1715b03c1d6e`
