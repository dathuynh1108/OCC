# MVTec image-method comparison: full schedule

This is a new full-schedule identity for the author-image-model adaptation to
MVTec AD. It does not replace or mutate `mvtec-native-light-4090-20260915`.

## Fixed inputs and evaluation

- All 15 official MVTec AD categories; 3,629 normal training images and 1,725
  test images; seed 0.
- The 32x32 cache is byte-identified by
  `artifacts/reproduction/mvtec-images32/manifest.json`; category IDs, labels,
  source lock and AUROC/AP implementation are unchanged from the light run.
- No test-set parameter search. Every admitted checkpoint is reloaded for score
  replay. DROCC keeps its source test-selected checkpoint only as a diagnostic;
  the comparison row uses the fixed final epoch.

## Full schedules

- **Deep SVDD:** unchanged MVTec preprocessing (per-image L1 GCN, then a scalar
  normal-train min/max), author CIFAR CNN/AE, batch 200, Adam 1e-4, gamma 0.1,
  AE 350 epochs/milestone 250, then SVDD 150 epochs/milestone 50. The AE is
  shared by one-class and soft-boundary objectives; the center and soft-boundary
  behavior remain the source runner's behavior.
- **DROCC:** unchanged author CIFAR normalization and MVTec adaptation settings:
  Adam 0.001, batch 128, radius 0.2, gamma 2, mu 1, ascent step 0.001,
  effective 50 ascent steps, projection every 10, no CE-only warmup. Run for
  100 epochs with the source 40%/80% LR thresholds.
- **RBF SVDD:** has no epoch schedule. Its verified identical light export is
  reused rather than refit. PatchCore and the NBD results are also reused
  unchanged.

The result directory is named after the actual accelerator used. This is an
adaptation of the authors' image models to MVTec, not a published MVTec benchmark
from the Deep SVDD or DROCC papers.
