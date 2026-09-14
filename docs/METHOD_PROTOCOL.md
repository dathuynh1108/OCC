# What was reproduced and what was adapted

This experiment evaluates algorithms on a common image representation. The
implementation is new; the previous project source and pilot configuration were
not available. The supplied slides define NBD, and the supplied handoff identifies
MVTec AD and Wide ResNet-50-2. All settings were fixed before test predictions.

| Family | Preserved principle | Declared adaptation for this experiment |
|---|---|---|
| PatchScore | Minimum squared Euclidean distance to normal memory | Project baseline, not full PatchCore; NBD-center and geometric-byte-budget controls; shared top-1% mean aggregation |
| RBF SVDD | Solve the constrained SVDD dual QP; kernel-space center and radius from free support vectors | Fit on 2048 farthest-first normal patch descriptors; normal-only median bandwidth; not a dense kernel on every fit patch |
| Deep SVDD | Bias-free trainable encoder, fixed nonzero center, squared distance loss, weight decay, AE initialization | Frozen ImageNet CNN plus 1536→64→16 MLP; 50 AE and 100 one-class epochs; not the original end-to-end MNIST/CIFAR CNN protocol |
| DROCC | Normal BCE plus adversarial-negative BCE; normalized ascent and annulus projection | Frozen descriptors and 1536→64→1 MLP; 100 epochs, 10 warmup, 50 ascent steps, projection every step; radius from fit-normal 5-NN distances |
| NBD | Local PCA bubbles, bidirectional local energy graph, diffusion distance, frame consistency, held-out normal tail scaling | Previously unspecified hyperparameters fixed in `configs/full.json`; 128 bubbles, rank 16, 128 neighbors, 8 attachment candidates, 12 graph neighbors, diffusion times 1/3/5 |

The primary source URLs and inspected code revisions are in
[`source_provenance.json`](source_provenance.json). In particular, the DROCC source
uses different domain-specific radii, epoch schedules and architectures; those
paper-table numbers are not directly comparable with this shared-descriptor run.

## Dataset and label isolation

The full original MVTec AD archive must match the published SHA-256 used by the
Anomalib downloader. Its original category image counts and anomaly masks are
checked before extracting features. Every training normal image is allocated to
exactly one of fit, score calibration and image-threshold calibration. No test
image or label participates in fitting, neighbor/radius estimation, bandwidth
selection, component calibration, epoch selection or threshold selection.

The 1536-dimensional descriptor is the concatenation of layer2 (512) and layer3
(1024), with local 3x3 average pooling and bilinear spatial alignment. All methods
consume the same float32 arrays. No per-method or test-derived standardization,
PCA, feature learning or resampling is applied. Every fit patch trains both heads
in every epoch, including the final partial batch.

## NBD details needed for the slides

Local PCA uses the sample covariance denominator n−1. The stored center is the
neighborhood mean, not the farthest-first anchor. Tangential variance and residual
variance are regularized by the declared positive epsilon. The graph uses the
union of directed normal-center neighbor edges and therefore has symmetric
weights. Graph scales are positive medians over normal-only candidate edges.
Self-loop weight is 1; the optional thickness-mismatch term is disabled.

Diffusion coordinates retain all M−1 nonconstant modes, ordered by absolute
eigenvalue. The constant eigenfunction is removed explicitly, including when
there are multiple stationary modes because the graph is disconnected. This
preserves the transition-distribution distance; a regression test checks both
connected and disconnected cases. Query diffusion dispersion is evaluated as
weighted variance, equivalent to the pairwise formula in the slides.

Each of B/A/D/F is transformed by the exact held-out-normal empirical upper tail,
including ties, with add-one smoothing. The selected NBD score is B+A+D+F with
frame weight 1. Reporting a better ablation as if it were the selected NBD score
would be a protocol change. Every variant receives the same top-1% mean image
aggregation. The reported score is not an anomaly probability.

## Training and accounting details

DROCC's two-layer ReLU head permits an exact analytic normalized input gradient.
For a generated negative, BCEWithLogits(f(x),0)=softplus(f(x)); its positive scalar
derivative cancels on normalization. The implemented direction is tested against
autograd. This removes unnecessary backward graph construction without reducing
the number of ascent steps or changing the objective.

The geometric byte-budget comparison counts all stored NBD center, frame,
variance, graph and diffusion arrays. It excludes the shared backbone and score
calibration arrays; that exclusion is part of the comparison definition, not a
claim that those arrays consume zero storage. Per-run diagnostics expose actual
array byte counts and the resulting PatchScore memory size.

Image thresholds use the order statistic ceil((n+1)(1−alpha)), with the score
strictly greater than the threshold classified as anomalous. If that order
exceeds n, the threshold is positive infinity. At alpha=0.05 this affects small
held-out sets such as toothbrush; no test label is used to force a finite cutoff.

## Interpretation rules

Keep poor scores, collapse-like training diagnostics, infinite thresholds and
negative paired deltas in the report. Do not import the previous WDBC/Wine/Digits
numbers into the new table. Do not describe a CPU fixture or random-network test
as a completed benchmark. The measured primary endpoint is image AUROC; saved
patch maps alone do not establish pixel AUROC or AUPRO.

The frame-distance implementation evaluates the actual stored projector
Frobenius norm in float64, then stores the result in float32 with an exact zero
diagonal. This prevents float32 orthogonality roundoff from creating nonzero
self-distances before empirical tail scaling. The discarded full-v1 execution
and its normal-only regression evidence are described in `VALIDATION.md`.
Median computations follow PyTorch's lower central order statistic for an even
number of values, consistently across normal graph scales, kernel bandwidth and
DROCC radius estimation.
