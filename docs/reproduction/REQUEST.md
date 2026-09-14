# Codex task: source-faithful reproduction, then a separate NBD comparison

Work in the local checkout of https://github.com/dathuynh1108/OCC.
Implement the missing runners and verification, and run them on the local NVIDIA GPU.
Do not stop at a plan or replace experiments with synthetic benchmarks. Read this entire task first.
Never rent cloud GPUs, incur charges, delete existing outputs, or push changes without permission.

## 1. Preserve what has actually been measured

The reviewed OCC snapshot is `91223d63adc3289786bbdeff16d5feeb59aeb3d7`.
`results/mvtec-full-v2/` is a completed **shared-frozen-CNN adaptation**, not an
original-paper reproduction. Preserve it, including its poor NBD, DeepSVDD-head
and DROCC-head results. Do not rewrite scores, labels, splits, or provenance.

The old extractor is in `nbdbench/data.py`: frozen Wide ResNet-50-2 V1,
layer2/layer3, spatial average pooling, bilinear alignment and concatenation to
1536 dimensions. Its PatchScore uses local-mean centers and top-1% image pooling.
This is NOT the author PatchCore feature/scoring implementation. Its neural
methods train MLP heads on those cached vectors, not the original image CNNs.

Create a new branch without disturbing uncommitted changes. Use new output paths.
If HEAD has changed, record the actual HEAD and its diff from the reviewed snapshot.

## 2. Sources: pin these repositories, not an arbitrary reimplementation

| Method | Repository | Reviewed commit |
|---|---|---|
| PatchCore | https://github.com/amazon-science/patchcore-inspection | `fcaa92f124fb1ad74a7acf56726decd4b27cbcad` |
| SVDD author reference toolbox | https://github.com/DMJTax/dd_tools | `efaaf04efae1f8be78906836a5d31547b48be7af` |
| Deep SVDD original paper implementation | https://github.com/lukasruff/Deep-SVDD | `e20f18c8d0ad9dc01cad09fdf311bd861351a9ad` |
| Deep SVDD author PyTorch implementation | https://github.com/lukasruff/Deep-SVDD-PyTorch | `1901612d595e23675fb75c4ebb563dd0ffebc21e` |
| DROCC | https://github.com/microsoft/EdgeML | `81025fce8ba28707eabe72e11bf3987a8d745608` |

Papers and supplementary material:
- PatchCore: https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html
- Accessible author-hosted PatchCore PDF: https://cdn.amazon.science/ec/c4/8f8fad644e45a044262ca9fb95c1/towards-total-recall-in-industrial-anomaly-detection.pdf
- SVDD: https://doi.org/10.1023/B:MACH.0000008084.60811.49
- Deep SVDD: https://proceedings.mlr.press/v80/ruff18a.html
- Deep SVDD PDF: https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf
- DROCC: https://proceedings.mlr.press/v119/goyal20c.html
- DROCC supplement: https://proceedings.mlr.press/v119/goyal20c/goyal20c-supp.pdf

Keep licenses and authorship. Use separate environments/containers for legacy
frameworks. Do not import identically named upstream modules into one shared
namespace. Pin packages and record the actual checkpoint bytes, not only a model
name. `dd_tools` is a later author-maintained MATLAB reference, not an archived
2004 experimental release; do not call its current defaults the paper settings.

## 3. Required audit before expensive runs

Write `docs/reproduction/SOURCE_AUDIT.md`, `source_lock.json`,
`reproduction_matrix.csv` and a machine-readable `run_plan.json` first. Every
numeric setting must cite a paper page/table or pinned source file/line. Follow
runtime argument plumbing; parser defaults alone are not evidence.

For each target record: original experiment/table, dataset/version, normal class,
train/validation/test identities, preprocessing and its fit population, network,
initialization/checkpoint, trainable parameters, loss, optimizer, regularization,
epochs/LR schedule, batch size, actual optimizer steps, model-selection policy,
feature aggregation, memory sampling, score direction, threshold, metrics and
number of repeated runs. Distinguish paper wording, released-code behavior, and
any deliberate compatibility or evaluation changes.

Do not call a result exact reproduction merely because it uses an author repo.
If source and paper conflict, report both; select a clearly named target instead
of silently combining favorable choices. Missing information must stay marked
`unresolved`. Do not invent a hyperparameter or claim an unrun table reproduced.

## 4. Run separate tracks; never combine their results into one ranking

### A. Native-source reproduction / replication

Use each method's own published data representation and original benchmark.
Do not force a shared ImageNet backbone on Deep SVDD or DROCC.

**PatchCore: full MVTec AD, author-code baseline.**

1. Use the pinned author's `src/patchcore/patchcore.py`, `common.py`, `sampler.py`,
   `backbones.py`, `datasets/mvtec.py` and the Quick Guide/sample scripts.
2. Explicit baseline target: Quick Guide single WideResNet50, layer2+layer3,
   resize 256 / crop 224, patch size 3, stride 1, pretrain embedding dimension
   1024, target embedding dimension 1024, 1 nearest neighbor,
   `approx_greedy_coreset` percentage 0.1. The ratio 0.1 is 10%, not 0.1%.
   Other paper/ensemble/1% configurations get separate names and source targets.
3. Preserve the **entire** extractor: 3x3 `Unfold`, per-layer patch-grid alignment,
   `MeanMapper` adaptive pooling to 1024, stack, then `Aggregator` to 1024.
   At 224 input size the final descriptors are 28x28x1024. This is neither global
   image pooling nor our old 28x28x1536 average-and-concatenate extractor.
4. The old `pretrained=True` Wide ResNet-50-2 checkpoint maps to ImageNet-1K V1
   only after verifying upstream loader/version/checkpoint identity. Do not use
   a mutable `DEFAULT`, V2, or random fallback. Freeze/eval the encoder.
5. Coreset entries must be selected normal descriptors, not neighborhood means.
   Preserve upstream projection and sampling. Its random projection accelerates
   coreset selection; it is NOT a new PCA-projected inference feature space.
6. Upstream source uses FAISS L2 distances (squared L2 in IndexFlatL2) and a maximum
   over patch scores. The paper Eq. (7) additionally specifies neighbor-based image
   reweighting. The inspected release does not implement that reweighting in its
   normal `_predict` path. Report this discrepancy. Name the first run
   `PatchCore_author_code_WR50_10pct`; do not claim exact Eq. (7) reproduction.
   A paper-equation variant is separate and may run only after all unspecified
   neighbor settings are resolved from source/supplement, not guessed.
7. Use all official training normals for this native author-code AUROC target and
   the complete official test split. Do not apply our 60/20/20 split to it and
   still call it the unmodified original training protocol. Run all 15 categories.
   Seeds 0/1/2 are our declared replication repeats unless the chosen original
   table requires another set/count; never label this count as a paper fact.
8. Preserve patch-map upsampling/smoothing for native localization evaluation.
   Report pixel AUROC only if computed from the original aligned masks. AUPRO
   requires a separately verified evaluator; otherwise mark not evaluated.
   A saved colored heatmap is not a localization metric.

**Deep SVDD: original image encoder, not a frozen-CNN head.**

1. Native target: MNIST and CIFAR-10 one-class experiments, all ten normal classes,
   original test sets, both one-class and soft-boundary objectives kept separate.
   Paper Table 1 reports ten seeds; use the original experiment scripts' seed
   list, or explicitly record a new ten-seed list if exact seeds cannot be found.
2. The paper-run repository is Theano/Lasagne (`lukasruff/Deep-SVDD`); the author's
   PyTorch repo is a later implementation. Prefer the original experiment runner
   in an isolated compatible environment. If porting is necessary, use the author
   PyTorch modules but document and test parity instead of silently switching to
   their shorter example schedule. Label unverified parity as an author-code
   replication, not a matched historical numeric reproduction.
3. Native LeNet-type CNNs operate on images and are trainable end-to-end. MNIST:
   conv channels 8/4, embedding 32. CIFAR-10: conv channels 32/64/128, embedding
   128. Resolve spatial padding, pooling, leaky-ReLU slope, BatchNorm affine
   settings/statistics and weight initialization from the selected source.
   Do not attach the old 1536->64->16 MLP or an ImageNet WRN in this track.
4. Reconstruct images with a DCAE, copy the pretrained encoder, discard its
   decoder, initialize a fixed nonzero center using the documented normal
   forward pass, then train Deep SVDD. Keep no-bias constraints and the original
   activations. AE is initialization, not an anti-collapse guarantee. Do not add
   a reconstruction penalty during SVDD unless explicitly running another named
   method. Preserve the original soft-boundary R/nu update protocol separately.
5. Audit schedule discrepancies: the paper describes DCAE 250+100 epochs and
   Deep SVDD 150+100, with 1e-4 then 1e-5; author PyTorch examples use different
   phase lengths (e.g. 150 SVDD epochs and dataset-specific AE durations).
   Original scripts can load AE weights via `in_name` while passing `--pretrain 0`.
   Verify the loaded encoder provenance; `pretrain 0` here is not proof of random
   initialization. Never splice a paper epoch total into a different scheduler.
6. Audit original global contrast normalization/min-max preprocessing rather than
   substitute ImageNet normalization. Trace any precomputed bounds and what
   data they were computed on. Report source behavior, not a cleaner invented one.
7. Save epoch losses, feature variances before/after, center and gradients. Report
   collapse diagnostics without selecting the run/epoch by final test AUROC.

**SVDD: kernel data description; no native CNN backbone.**

1. Use Tax's `DMJTax/dd_tools` as the author reference (`svdd.m`, `svdd_optrbf.m`,
   `ksvdd.m`, documented examples). Respect MATLAB/PRTools requirements; do not
   assume Octave compatibility or buy a license. If unavailable, report that
   limitation and use an independently verified QP implementation with its own
   label, not an allegedly executed MATLAB reference.
2. Select an explicitly identifiable original Tax--Duin experiment from the paper
   and reproduce its actual data, preprocessing, kernel, parameters and metric.
   First write the target in the audit. If the historical data/settings cannot be
   recovered, mark that numeric reproduction blocked. Running a dd_tools example
   verifies a reference implementation, not the entire 2004 paper table.
3. Also reproduce the shallow Gaussian OC-SVM/SVDD baseline used in Ruff et al.'s
   image Table 1 as a separately named target. Their paper uses PCA retaining
   95% variance, an inverse-length-scale grid and a 10% labeled test holdout for
   tuning, and reports the better of two nu settings. This is NOT our old median
   bandwidth / 2048-CNN-patch coreset experiment. Make all supervised/transductive
   choices visible; do not claim a normal-only deployment protocol for that row.
4. Check the actual SVDD dual, box constraints, sum(alpha)=1, radius convention,
   decision orientation, primal/dual feasibility and KKT residuals. The author's
   Gaussian kernel is exp(-d^2/sigma^2); avoid a hidden factor of two in gamma.
   RBF OC-SVM and SVDD have a known constant-diagonal-kernel connection; an
   equivalent implementation needs a derivation and numeric score/radius checks.
   Never relabel an arbitrary OneClassSVM fit as an independently solved SVDD QP.
5. A 2048-point or other support cap changes the experiment; it must not silently
   replace full native training to make a run feasible.

**DROCC: author CIFAR image pipeline and original class-specific settings.**

1. Start with the full ten-class CIFAR-10 one-vs-rest experiment (main paper Table
   1), using `examples/pytorch/DROCC/main_cifar.py`, its data processor and
   `pytorch/edgeml_pytorch/trainer/drocc_trainer.py`. No WRN feature cache here.
2. Use the trainable CIFAR10_LeNet defined by the author: conv channels 32/64/128,
   representation 128, then the source binary head. Negative search is on the
   normalized image input tensor. Preserve the preprocessing, BN behavior,
   optimizer and schedule. Do not inject AE pretraining that this path does not use.
3. Read supplement Table 11 visually. The per-class (radius, mu, optimizer, LR,
   ascent step) settings are:

   airplane:   (8,  1.0, Adam, 0.001, 0.001)
   automobile: (8,  0.5, SGD,  0.001, 0.001)
   bird:       (40, 0.5, Adam, 0.001, 0.001)
   cat:        (28, 1.0, SGD,  0.001, 0.001)
   deer:       (32, 1.0, SGD,  0.001, 0.001)
   dog:        (24, 0.5, SGD,  0.01,  0.001)
   frog:       (36, 1.0, SGD,  0.001, 0.01)
   horse:      (32, 0.5, SGD,  0.001, 0.001)
   ship:       (28, 0.5, SGD,  0.001, 0.001)
   truck:      (16, 0.5, SGD,  0.001, 0.001)

   Cross-check these transcribed settings yourself against the PDF before lock.
   The image experiments specify gamma=1. Do not use our feature-space gamma=2,
   median-5NN radius, 8192 batch or 10-epoch warm-up and call it original DROCC.
4. Trace effective arguments: reviewed `main_cifar.py` passes `only_ce_epochs=0`,
   not its parser warm-up default; `--ascent_num_steps` is parsed but not forwarded,
   so the trainer default is 50. Projection occurs every 10 ascent steps, not
   automatically every step. Preserve or explicitly separate altered behavior.
5. Important integrity issue: that script passes `test_loader` as `val_loader`;
   the trainer saves the model with the best score on that loader across epochs.
   Thus blindly calling the author script does not create an untouched-test run.
   Keep a historical/source-behavior diagnostic labeled `test_selected`, and a
   separate preregistered terminal-epoch result labeled `fixed_final_epoch`.
   One training trajectory can export both checkpoints without retraining.
   Only the latter belongs in an untouched-test comparison. Do not hide either
   policy, claim they are identical, or optimize new hyperparameters on the test.
6. Determine seed count and any missing training details from original sources.
   If exact seeds/count remain unavailable, use an explicitly declared replication
   plan (0/1/2) and say it is our plan; do not invent a paper seed count.
7. Tabular DROCC and ImageNet-10 are different source experiments. Their datasets,
   metrics and architecture choices must remain separate. ImageNet-10 uses
   MobileNetV2 in the paper, not this LeNet. Audit coverage and availability; do not
   claim all DROCC experiments reproduced if only CIFAR is completed. Do not
   substitute DROCC-LF or a random synthetic image dataset for a missing experiment.

### B. Controlled NBD comparison with the author PatchCore encoder

After feature parity passes, rerun a NEW shared-feature experiment on all 15
MVTec categories and seeds 0/1/2. Its purpose is to isolate NBD scoring, NOT to
reproduce every native paper on MVTec.

- Reuse the old image split manifests (60% fit / 20% NBD component calibration /
  remaining image-threshold calibration). Use the exact author PatchCore feature
  extractor, 1024-D, for every controlled method. Generate new caches and hashes.
  Never reuse 1536-D cache files or old predictions under a 1024-D config.
- Keep the existing NBD model, parameters and B/A/D/F ablations unchanged except
  for the input dimension implied by the new encoder. Do not tune them on already
  inspected MVTec results. This rerun is a post-result protocol correction, not a
  pristine discovery holdout or proof that feature extraction caused old failures.
- Same-center PatchScore must use the exact fitted NBD local-mean centers; label
  it a control, not PatchCore. Byte-budget PatchScore must use actual selected
  normal patches and report its actual geometric-array budget/exclusions.
- Add an author-PatchCore-memory control on the same FIT subset. Its modified split
  and our shared image aggregation mean it is a controlled adaptation, not Track A.
- Retain explicit `SVDD_shared_features`, `DeepSVDD_head_shared_features`, and
  `DROCC_head_shared_features` labels if running those rows. They may reuse the
  existing head settings after dimension adjustment but are never native-source
  reproductions. Keep them out of the native-paper results table.
- Keep top-1% image pooling for ALL primary controlled rows to isolate feature
  changes from aggregation changes. Report native PatchCore max only in Track A;
  any controlled max-pooling ablation must be predeclared for every relevant row.
- Reserved component-normal images are used only by NBD's component tail scaling;
  ordinary baselines need not consume them. All methods choose an image threshold
  using the separate threshold-normal subset. No test-derived thresholds.
- Preserve the finite-sample rule k=ceil((n+1)(1-alpha)), alpha=0.05;
  threshold=s_(k) if k<=n and +infinity otherwise; anomaly iff score>threshold.
  Retain infinite thresholds, measured FPR/TPR, and all weak results. AUROC does
  not require this threshold. Patch-level scaling is not an anomaly probability.
- Report memory/latency with common backbone costs and method-specific state
  separately. Fair byte matching is not a statement about equal peak GPU memory.

## 5. Verification gates, runtime and outputs

First run unit tests and tiny numerical fixtures; label them tests, not benchmark
results. For the author PatchCore adapter compare upstream outputs on exactly the
same preprocessed real images/checkpoint: layer tensors, extracted descriptors,
fixed-memory NN distances, image score, and score maps. Record max/mean absolute
error and tolerances. An independent shape check (1024 dims) is not enough.
Compare head architecture, loss, gradients and one deterministic optimizer step
against upstream modules when porting. Validate score polarity without test tuning.

Run a predeclared small smoke job for execution only, then the complete locked
run matrix. Do not select categories/configs based on smoke performance. Estimate
GPU time and RAM/disk first. If resource limits prevent completion, checkpoint,
resume and report exact completed/failed coverage; do not truncate epochs, swap
architectures or silently drop categories. Gradient accumulation is not automatically
batch-equivalent when BatchNorm is present; log and label any necessary change.

Store: repo/weights/data SHA-256, environment and CUDA/GPU details, exact commands,
resolved config, split IDs, original labels, predictions, thresholds, checkpoints,
optimizer/RNG state, learning curves, failure logs, completed-run manifest and
wall time. Recompute AUROC/AP/confusion counts independently from saved predictions.
Replay scores from checkpoints. Save named diagnostics instead of auto-restarting
bad-looking runs until they improve. Full MVTec has 15 categories; report all.

Create at least:
- `docs/reproduction/SOURCE_AUDIT.md` and `source_lock.json`;
- `docs/reproduction/reproduction_matrix.csv` with target table and coverage;
- locked configs and runnable native/controlled scripts;
- `results/native/<method>/<target>/` with separate protocol labels;
- `results/mvtec-author-encoder-controlled-v3/` for Track B;
- `REPRODUCTION_REPORT.md`, per-class/per-category CSVs, seed-macro summaries,
  paper-reported vs measured columns, verification JSON and commands to resume;
- `SLIDE_UPDATE_HANDOFF.md` with valid GitHub source permalinks and evidence links.

The native table must not average MNIST, CIFAR and MVTec into one method ranking.
Keep exact-paper reproduction, released-code replication, changed evaluation and
shared-feature adaptation as different statuses. Reference results are not targets
that the code must be adjusted to hit. Report differences without causal claims.

## 6. Slide changes after new runs

Keep English, concise text, the current navy/teal Beamer style and figures.
Main order: native method explanation + its GitHub link -> clearly labeled
baseline experiment/results -> NBD model -> NBD comparisons/ablations.
Never introduce NBD component calibration as a baseline method requirement.

Link directly to author GitHub repos on the relevant method slides. For Deep
SVDD, label original Theano vs author PyTorch clearly. For SVDD, label dd_tools
as an author reference toolbox. Keep the collapse/no-bias/fixed-center/AE slide.
Show the exact encoder path and tensor shapes, and distinguish ImageNet weights
from AE initialization and training from scratch. Do not replace a measured
1536-D slide/table label with 1024-D until the new run actually produced that table.

Historical MVTec v2 results remain visible as historical shared-feature evidence.
New native results appear only after complete verification; unfinished tables are
marked pending with no invented numbers. Keep NBD B+A+D+F as the declared method,
not the best ablation. Add clickable links to the new result folder only after
it exists in Git; do not fabricate repository paths.

At completion provide source differences found, actual coverage, measured numbers,
failed/pending tasks, verification results, file paths, exact commands and a short
slide-update summary. Do not end with only an implementation plan.
