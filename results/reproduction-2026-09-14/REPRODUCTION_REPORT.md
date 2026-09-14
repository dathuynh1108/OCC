# OCC source-audited reproduction and common MVTec benchmark

The common MVTec experiment completed 15 categories × 3 seeds × 11 variants.
**Final declared NBD (B+A+D+F): 78.66 ± 1.99% image AUROC; 90.99 ± 0.77% AP.**
Native-paper datasets are separate method evaluations, not a cross-dataset ranking.

The user replaced the original long schedule with about two hours remaining.
All scores below are real measured exports. Reduced-epoch results are **not full-schedule historical-paper reproductions**.

## Common MVTec results

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

AUROC/AP: mean of category-macro seed means; sample SD across three seeds.
FPR/TPR: category-and-seed macro means at normal-only calibration thresholds.
Poor results and every ablation remain visible. The best ablation does not replace NBD.

- NBD minus PatchScore: same centers: -16.99 ± 1.80 AUROC percentage points.
- NBD minus PatchScore: byte budget: -20.29 ± 1.90 AUROC percentage points.
- NBD minus PatchCore: same FIT, top 1%: -20.30 ± 1.86 AUROC percentage points.

Shared protocol: official MVTec AD, 3,629 normal train images and all 1,725 test
images (467 normal, 1,258 anomaly). Exact historical v2 image split IDs: 60% FIT,
20% NBD component calibration, remainder image threshold calibration. Shared
WR50-2 ImageNet V1, author 3×3 Unfold/alignment/pooling into 784×1,024 descriptors;
author startup BN probe retained, then frozen/eval. No v2 feature cache reused.
All common rows use mean of top 1% patch scores; thresholds use alpha=.05,
k=ceil((n+1)*.95), infinity when k>n, strict score>threshold. No threshold tuning on test.

Deep head: AE 5+SVDD 15. DROCC feature head:15 epochs (10 warmup + 5 adversarial),50
ascent steps. Native-method names must not be applied to these feature heads.
NBD math, graph, calibration,128 bubbles/rank16 and byte-budget controls are unchanged
from91223d6; only author descriptor construction and the user-approved epoch budgets
change. This is a post-result protocol correction, not evidence of a causal improvement.

## Native method results

| Target | Dataset | Variant | Paper AUROC (%) | Measured AUROC (%) | Repeats |
| --- | --- | --- | --- | --- | --- |
| DROCC | cifar10 | fixed_final_epoch | 74.23 | 64.71 ± 2.18 | 3 |
| DROCC | cifar10 | test_selected | 74.23 | 65.99 ± 2.00 | 3 |
| DeepSVDD | cifar10 | one-class | 64.81 | 61.09 ± 0.62 | 10 |
| DeepSVDD | cifar10 | soft-boundary | 63.28 | 61.52 ± 0.63 | 10 |
| DeepSVDD | mnist | one-class | 94.80 | 92.69 ± 0.74 | 10 |
| DeepSVDD | mnist | soft-boundary | 93.53 | 92.73 ± 0.78 | 10 |
| Gaussian_OCSVM_equiv_SVDD | cifar10 | nu_0.01 | 64.78 | 60.56 ± 0.20 | 10 |
| Gaussian_OCSVM_equiv_SVDD | cifar10 | nu_0.1 | 64.78 | 60.67 ± 0.26 | 10 |
| Gaussian_OCSVM_equiv_SVDD | mnist | nu_0.01 | 91.29 | 95.51 ± 0.05 | 10 |
| Gaussian_OCSVM_equiv_SVDD | mnist | nu_0.1 | 91.29 | 95.51 ± 0.04 | 10 |
| PatchCore | mvtec | author_code | 99.00 | 99.02 ± 0.14 | 3 |

Paper columns are reference values, not matched-protocol deltas. For Ruff/Goyal,
they are macros calculated from the published Table 1 class means; no seed SD is
inferred from class SDs. Ruff's kernel and soft-boundary paper rows select the
better nu, whereas the measured rows retain the declared separate configurations.
PatchCore references the paper's 10% row (99.0%), whose equation differs from
the released image-score code. Source URLs and exact printed values are in
[paper_reference.json](../../docs/reproduction/paper_reference.json).
Per-class reference/measured values are in paper_vs_measured_per_class.csv.
All selected matrix rows completed.

- PatchCore: author-code WR50/V1, full normal training,10% approximate greedy coreset,
  FP32 GPU FAISS squared-L2 1NN, image maximum. Pixel AUROC is in native_summary.csv;
  AUPRO not evaluated. Released source omits paper Eq.7 reweighting, so that exact
  paper target is not claimed. Three declared repeat seeds0/1/2.
- Deep SVDD: unchanged author PyTorch LeNet modules, MNIST/CIFAR10, all normal classes,
  seeds1..10, AE 5+SVDD 12, source 10-epoch soft-boundary warmup. Final-epoch selection.
  Original LR milestones remain outside this short run. Theano historical runtime
  was not reproduced; modern author-port parity is verified only on this runtime.
- DROCC: unchanged author CIFAR CNN and adversarial function, five epochs, 50 ascent
  steps, projection every10, class-specific Table 11 parameters and gamma 1. Both
  final and best-test-selected outputs retained. **Best-test-selected is optimistic
  and uses test labels**, as the released runner does; it is not a clean final-test estimate.
- Gaussian OC-SVM/equivalent RBF SVDD: modern libsvm/source-protocol adaptation,
  full selected-class training except source MNIST batch-multiple rule, train-only
  PCA 95%, gamma2**[-10..-1], both nu=.01/.1. Gamma uses1000 labeled test samples;
  evaluated9000 exclude that holdout. Nu rows remain separate. This is not an
  executed MATLAB dd_tools or Tax--Duin2004 numeric reproduction.
- Tax--Duin Table2 remains unreproduced: exact Iris folds, sigma search and numerical
  settings were not recovered. No arbitrary substitute score is reported.

## Inference scope and stored state

| Method | Median ms | p95 ms | State MiB | NBD calibration MiB |
| --- | --- | --- | --- | --- |
| Shared author WR50 descriptor | 30.91 | 32.93 | 263.03 | 0.00 |
| PatchScore: same centers | 1.30 | 1.32 | 0.50 | 0.00 |
| PatchScore: byte budget | 1.47 | 1.49 | 9.01 | 0.00 |
| RBF SVDD (feature adaptation) | 10.82 | 11.05 | 16.02 | 0.00 |
| Deep SVDD head (15 epochs) | 1.41 | 1.43 | 0.25 | 0.00 |
| DROCC head (15 epochs) | 1.50 | 1.52 | 0.25 | 0.00 |
| NBD: B+A+D+F | 7.47 | 8.77 | 9.13 | 0.98 |
| PatchCore: same FIT, top 1% | 1.91 | 1.92 | 38.28 | 0.00 |

Measured after training, on one bottle image / seed 0, batch size 1 (784 patches),
30 repetitions after 3 warmups, CUDA synchronized. This is an illustrative actual
checkpoint measurement, not a dataset-wide latency claim. Shared descriptor time
excludes disk/PIL; method times exclude the CNN and include CPU descriptor input,
transfers and image pooling. NBD includes the current implementation's ECDF sorting.
State counts numerical arrays only; calibration is separate, not peak GPU memory.
Full scope: inference_scope.json. Geometry budgets across all category/seed runs:
common_resources.csv. The common backbone is counted once, separately.

## Verification, provenance and rerun

8,262,100 exported image scores were checked independently
with Python standard-library tied-rank AUROC and threshold-group AP. Common FPR/TPR
were recalculated with the declared strict threshold. IDs, labels, uniqueness,
counts, prediction hashes and checkpoint replay gates were checked. See
independent_metric_audit.json and coverage.csv for exact evidence.

Pinned source hashes: source_lock.json. Original schedule: run_plan.json.
Budgeted schedule: run_plan_budgeted.json. Source analysis:
docs/reproduction/SOURCE_AUDIT.md. Numeric fixtures cover upstream Deep/DROCC
losses, gradients, parameters, BN and resume; PatchCore descriptors, NN/maps and
cached/direct coreset/RNG parity; shallow extracted source splits/PCA and independent
QP equivalence. Every admitted trained result replays its persisted checkpoint.

A shallow replay gate caught a float32 PCA batch-shape difference (9,000 versus 10,000
rows, maximum score error about2.04e-6). Replay was corrected to transform the same
full10000 rows before selecting the9000 evaluated IDs, restoring exact agreement;
no metric tolerance, trained estimator, gamma selection or dataset was changed.
Failed attempts are retained in runtime logs and are not admitted as results.

Training ran on one rented RTX3090 with FP32, AMP/TF32 disabled. Native GPU jobs and
CPU libsvm jobs overlapped, so training wall times are not comparable standalone
latencies. Geometric/coreset bytes exclude the shared backbone, calibration and
allocator overhead. Any separate inference timing is labeled by measured scope.

The independent audit also checks exact CSV float64 parsing. Pandas default parsing
can merge nearly tied kernel scores at about1e-16. Exports now use round-trip
parsing, retain previous metrics for traceability, and are checked against saved
estimator scores. No gamma selection or trained model was changed.

Code, CSVs, histories, manifests, figures and updated Beamer source/PDF are in Git.
Large checkpoints, score maps, official datasets and V1 weights are tracked by
backup_manifest.json. Verified snapshot coverage spans 5,577 local files and
422 additional files in ten GitHub Release ZIPs. See backup_offbox_verified.json,
cloud_backup_manifest.json and docs/reproduction/BACKUP_RESTORE.md. This is a hybrid
backup; backup_verified.json is reserved for a future complete local restore.
execution_closure.json records the final provider lifecycle state.
Historical results/mvtec-full-v2 are preserved unchanged and must not be mixed
with this author-encoder/reduced-epoch experiment.
