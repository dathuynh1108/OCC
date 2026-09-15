# MVTec image-method comparison: light configuration (diagnostic)

This experiment applies the authors' image models to MVTec AD. Deep SVDD and
DROCC did not supply this MVTec recipe in the selected releases. This is a
same-dataset comparison, not a claim to reproduce a published MVTec table.

## Locked before execution

- All 3,629 official normal training images and all 1,725 test images in 15
  categories; seed 0. Image IDs, labels and original SHA256 match the existing
  PatchCore/NBD dataset. No subsets of test images or categories.
- RGB, resize short side 256, center crop 224, then bilinear resize to 32×32.
  The first crop matches the existing MVTec evaluation field; 32×32 is the
  fixed input of the authors' CIFAR image CNNs. Cached uint8 pixels are hashed.
- Deep SVDD imports the unchanged author CIFAR CNN and autoencoder. Per-image
  source L1 GCN, then scalar min/max fitted only to the category's normal train
  images. CIFAR's dataset-specific min/max constants are not reused on MVTec.
  AE5/SVDD12; both objectives; batch200; author Adam, losses, LR ordering, fixed
  center and soft-boundary radius updates. One AE is shared between objectives.
- DROCC imports the unchanged author CIFAR CNN and adversarial search. Author
  CIFAR normalization, Adam lr0.001, batch128, five epochs, no CE-only warmup.
  Uses source CLI defaults radius0.2, gamma2, mu1 and ascent step0.001; the
  effective source search has 50 iterations, projecting every10. CIFAR Table11's
  animal-class parameters are not arbitrarily mapped onto industrial categories.
  Both the source best-test-AUC checkpoint and fixed-final checkpoint are exported.
  This light diagnostic is not used in the final benchmark comparison: a
  test-selected checkpoint cannot be the reported comparison row.
- RBF SVDD uses the constant-diagonal OC-SVM equivalence, nu0.01/0.1, source pixel
  scaling and train-only PCA95. Uses the author's non-grid-search branch:
  gamma = 1 / maximum squared pairwise train distance (`src/svm.py:188-192`).
  No test holdout is removed and no gamma is tuned on test labels.
- FP32, no AMP/TF32. Final table uses category-macro image AUROC/AP for seed0
  for every method, including already measured PatchCore and NBD.

## Evidence and operation

`prepare_mvtec_images.py` verifies every original image hash and builds the input
cache. `mvtec_image_baselines.py` imports the existing source-audited native
training loops; it does not use `nbdbench.heads` or WR50 patch descriptors.
The saved protocol and per-category configs bind the input cache and source lock.
Each model is reloaded and its predictions checked before a result is marked
complete. A category completion manifest hashes predictions, results and models.

```sh
python -m reproduction.prepare_mvtec_images
export OCC_RUN_PLAN=run_plan_budgeted.json
python -m reproduction.verify_deep
python -m reproduction.verify_drocc
python -m reproduction.verify_shallow
python -m reproduction.mvtec_image_baselines --method deep
python -m reproduction.mvtec_image_baselines --method drocc
python -m reproduction.mvtec_image_baselines --method shallow
```

Run source-parity checks in a fresh fixture directory. Existing completed model
runs are verified before being reused. No benchmark results are asserted here
until the actual exports and complete category coverage pass validation.
