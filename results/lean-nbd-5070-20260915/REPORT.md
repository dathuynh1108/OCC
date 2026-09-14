# Lean NBD full MVTec report

Verified full coverage: 45 category-seed runs (15 classes × 3 seeds).

## Final score candidates

| Score | Macro image AUROC (%) | AP (%) |
|---|---:|---:|
| PatchScore_same_centers | 95.86 ± 0.23 | 98.67 ± 0.05 |
| PatchScore_byte_budget | 98.57 ± 0.21 | 99.59 ± 0.06 |
| RBF_SVDD | 78.20 ± 0.26 | 90.46 ± 0.17 |
| DeepSVDD_head | 81.15 ± 0.18 | 91.24 ± 0.13 |
| DROCC_head | 57.24 ± 3.52 | 76.95 ± 1.83 |
| Bubble_A | 82.47 ± 1.75 | 93.03 ± 0.86 |
| Bubble_B | 82.71 ± 1.81 | 93.20 ± 0.85 |
| Bubble_A+D | 76.16 ± 2.72 | 89.55 ± 1.60 |
| Bubble_B+D | 75.64 ± 2.64 | 89.18 ± 1.59 |

## Per-class change after adding D

| Class | Final | Compared with | Delta AUROC (pp) | Status |
|---|---|---|---:|---|
| bottle | Bubble_A+D | Bubble_A | -7.67 ± 1.60 | reduced |
| cable | Bubble_A+D | Bubble_A | -12.49 ± 2.59 | reduced |
| capsule | Bubble_A+D | Bubble_A | -10.04 ± 3.62 | reduced |
| carpet | Bubble_A+D | Bubble_A | -1.99 ± 4.57 | reduced |
| grid | Bubble_A+D | Bubble_A | -13.92 ± 0.60 | reduced |
| hazelnut | Bubble_A+D | Bubble_A | -9.71 ± 2.03 | reduced |
| leather | Bubble_A+D | Bubble_A | -5.14 ± 1.99 | reduced |
| metal_nut | Bubble_A+D | Bubble_A | -4.43 ± 2.74 | reduced |
| pill | Bubble_A+D | Bubble_A | +2.37 ± 3.00 | improved |
| screw | Bubble_A+D | Bubble_A | -4.00 ± 1.19 | reduced |
| tile | Bubble_A+D | Bubble_A | +0.47 ± 0.28 | improved |
| toothbrush | Bubble_A+D | Bubble_A | -3.89 ± 1.92 | reduced |
| transistor | Bubble_A+D | Bubble_A | -4.13 ± 1.27 | reduced |
| wood | Bubble_A+D | Bubble_A | -18.33 ± 16.14 | reduced |
| zipper | Bubble_A+D | Bubble_A | -1.69 ± 0.98 | reduced |
| bottle | Bubble_B+D | Bubble_B | -8.02 ± 0.92 | reduced |
| cable | Bubble_B+D | Bubble_B | -13.77 ± 3.00 | reduced |
| capsule | Bubble_B+D | Bubble_B | -10.53 ± 3.33 | reduced |
| carpet | Bubble_B+D | Bubble_B | -3.09 ± 4.67 | reduced |
| grid | Bubble_B+D | Bubble_B | -14.17 ± 0.75 | reduced |
| hazelnut | Bubble_B+D | Bubble_B | -9.54 ± 1.09 | reduced |
| leather | Bubble_B+D | Bubble_B | -7.78 ± 5.43 | reduced |
| metal_nut | Bubble_B+D | Bubble_B | -4.77 ± 2.13 | reduced |
| pill | Bubble_B+D | Bubble_B | +1.45 ± 2.00 | improved |
| screw | Bubble_B+D | Bubble_B | -4.09 ± 1.59 | reduced |
| tile | Bubble_B+D | Bubble_B | +0.63 ± 0.40 | improved |
| toothbrush | Bubble_B+D | Bubble_B | -4.26 ± 2.43 | reduced |
| transistor | Bubble_B+D | Bubble_B | -5.76 ± 1.44 | reduced |
| wood | Bubble_B+D | Bubble_B | -20.13 ± 16.57 | reduced |
| zipper | Bubble_B+D | Bubble_B | -2.12 ± 1.66 | reduced |

D is the diffusion component of the graph, whose edge construction includes frame geometry. No direct F score or F fusion is reported.
All metrics use saved predictions after checkpoint replay; top-1% patch mean is the common image score.
