# Source audit and execution contract

No new benchmark results are asserted by this audit. Reviewed OCC base:
`91223d63adc3289786bbdeff16d5feeb59aeb3d7`. The supplied Overleaf archive is an
editorial input; its tables remain historical v2 until new runs are verified.
The user explicitly authorized the newly rented Vast GPU, pushing code/results,
local backup and stopping that GPU when finished. Those direct requests take
precedence over the generic local-only/no-push wording in the attached task.

Pinned author repositories and file hashes are in `source_lock.json`. Every
citation below is relative to the identified pinned repository. Missing historic
facts remain unresolved. Seeds, runtime choices and evaluation additions declared
here are replication choices, not paper facts.

## PatchCore: released author code, WR50, 10% memory

Target `PatchCore_author_code_WR50_10pct`, not exact paper Eq.7 reproduction.
The Quick Guide (`patchcore/README.md:23-34`) supplies WR50 layer2+3, resize256,
center crop224, patchsize3, embedding1024/1024, 1NN, approximate greedy ratio0.1.
`patchcore.py:91-145,282-308` unfolds 3x3 windows, aligns layer3 patch grids
bilinearly to layer2 with align_corners=False. `common.py:145-183` uses per-layer
MeanMapper and stacked Aggregator adaptive pooling, yielding 784x1024 descriptors.
The old OCC 1536D extractor is not this operation.

Use pinned ImageNet-1K V1 WR50-2 bytes, freeze all parameters and evaluate BN.
No optimizer, no trained parameters, no AE. `backbones.py` uses pretrained=True;
explicit V1 loading avoids the mutable modern DEFAULT. Weight SHA is locked.
Input normalization and original masks: `datasets/mvtec.py:41-109`; PIL Resize
and CenterCrop are retained for both images and masks. Binary pixel labels follow
the author's evaluator (`metrics.py`), with their exact mask handling audited in
the parity fixture. Image/mask interpolation is not silently replaced.

`sampler.py:31-75,118-171` projects temporarily to128 dimensions using a random
bias-free Linear, starts from10 random points and selects floor(N*0.1) original
1024D descriptors. This is 10%, not 0.1%. Selection order and RNG are preserved.
`common.py:14-61,296-350` uses FAISS squared L2; `patchcore.py:203-228,313-322`
uses image max; `common.py:186-208` bilinear maps then Gaussian sigma4. The paper
Eq.7 adds neighborhood image reweighting absent here. Its unspecified reproduction
settings remain unresolved; no guessed Eq.7 run is admitted.

All 15 MVTec AD categories, full official normal train/full test; seeds0,1,2 are
our declared repeats. No threshold is required for AUROC/AP. Pixel AUROC is only
reported from actual aligned masks; AUPRO remains not evaluated. One image at a
time is the declared extraction/inference batch to bound Unfold intermediates;
this is safe for a frozen eval-mode network and tested against multi-image output.
FAISS CPU/GPU numerical/backend choice must be recorded before full execution.

## Deep SVDD: native image networks, author PyTorch release

Historic numeric reproduction is distinct from released-code replication.
The original Theano project pins Theano0.8.2/Lasagne0.2.dev1/numpy1.12.1 in
`requirements.txt`; the available GPU runtime is Python3.12/CUDA12.8 on Ampere.
A compatibility probe is required and recorded before choosing the fallback.
No MATLAB or legacy Python environment was found during the initial runtime probe.

The selected executable fallback, if legacy runtime cannot execute, is named
`DeepSVDD_author_torch_readme`, explicitly a later author-code replication.
It uses the complete README examples, not a silently shortened paper run:
`deep_svdd_torch/README.md:86,105`: MNIST AE150/drop50/weight_decay5e-4;
CIFAR AE350/drop250/weight_decay5e-7; both SVDD150/drop50/weight_decay5e-7;
Adam lr1e-4 then1e-5, batch200. Paper Sec4.2 instead describes AE250+100,
SVDD150+100 and leak0.1. Those are not claimed reproduced by the README target.
The original experiment scripts also differ: original MNIST SVDD script150/drop50,
CIFAR loads AE via in_name with pretrain0. That flag is not random initialization.

Native CNNs are imported unchanged from `src/networks/mnist_LeNet.py` and
`cifar10_LeNet.py`: channels8/4+embedding32 (MNIST),32/64/128+embedding128
(CIFAR), kernel5/padding2/maxpool2, no convolution/linear biases, BN eps1e-4,
affine=False. Their F.leaky_relu default is0.01, unlike paper0.1. CIFAR AE
convolutions/transposed convolutions use Xavier with author gain, other layers
use author Torch initializers. Parameter counts are measured from these modules.
No ImageNet network or MLP head is attached. Decoder construction is source-owned.

`src/datasets/{mnist,cifar10}.py:20-47` uses per-image L1 GCN and per-class
precomputed min/max, claiming train-normal origin. `preprocessing.py:15-37`
divides by mean absolute deviation. Their numeric provenance is checked against
real train normals; no new fit on test is allowed. Original Theano loader instead
randomly truncates MNIST normals to a batch multiple (`datasets/mnist.py:112-116`)
and rescales using that subset; its GCN divides by sum absolute values, whose
constant scale cancels through min/max. This is an additional historical difference.
Modern torchvision removed CIFAR train_data/train_labels aliases: an explicit
adapter maps them to data/targets while retaining the author's transform/index.

`src/deepSVDD.py:102-113` copies matching AE encoder state keys. AE SSE sums pixels
then means examples (`ae_trainer.py:49-61`). SVDD uses fixed center mean in eval
mode with epsilon0.1 adjustment only for nonzero small components; objective is
mean squared distance or R^2 + mean(max(0,dist-R^2))/nu. Source soft-boundary nu0.1,
warmup10, radius quantile each minibatch afterwards (`deepSVDD_trainer.py:31,
55-94,156-182`). This differs from paper block-coordinate radius updates. Both
objectives run separately on MNIST and CIFAR10, all10 normal classes, seeds1..10
as in original `experiments/mnist_svdd_exp_seeds.sh:5`. No best-test selection.
Full test10000, full selected-class train with final partial batch per Torch loader.

Both source trainers call scheduler.step before minibatches. Effective LR per
epoch, framework-version shift, initialization/state transfer, losses, gradients,
and one optimizer step must pass recorded fixtures before running. Runtime
compatibility is not a claim of numeric parity with original Theano/2018 Torch.
Checkpoints contain model, optimizer, scheduler, RNG, center/radius and full history.
Log before/after feature variance, center, gradients; never restart a low-AUC run.

## SVDD: reference QP and separately named Ruff shallow experiment

`dd_tools/svdd_optrbf.m:34-72` builds K=exp(-d²/sigma²), maximizes
sum(alpha_i*K_ii)-alpha^T K alpha, sum(alpha)=1, 0<=alpha<=C, adds a
positive-definiteness correction and uses qld/quadprog. `:85-102` radius offset
is the mean -2*K*alpha over free support vectors, while `svdd.m:105` adds
1+alpha^T K alpha. This later MATLAB toolbox is not an archived 2004 experiment.
MATLAB/PRTools execution and original Tax--Duin data/settings remain unresolved;
no license purchase and no assumed Octave equivalence. A blocked historical
numeric target is reported honestly if the original experiment cannot be recovered.

Ruff paper Sec4.1/Table1 shallow baseline is separately targeted: PCA95%, full
normal training, gamma=2**[-10..-1], nu0.01 and0.1, 10% labeled test tuning.
Original `src/svm.py:139-176` removes the holdout from test, searches gamma per nu
and keeps strict greater AUC; `:251-267` negates decision_function for anomaly.
Original `scripts/mnist_ocsvm.sh` does not enable GCN; parser defaults gcn0.
`datasets/preprocessing.py:50-66` scales from train extrema; PCA fits train only.
Original MNIST subset/order RNG plumbing must be reproduced and indices exported.
Reported best-nu policy is explicit; both nu rows retained. Ten seeds1..10.

For constant-diagonal RBF, minimizing alpha^T K alpha under these constraints
also solves normalized OC-SVM. If f=sum(alpha*K)-rho, squared-distance excess
is -2*f with R²=1-2*rho+alpha^T K alpha. sklearn coefficients/intercept must be
normalized by nu*N. Verify equality, box constraints, sum, free-SV radius and
KKT against an independent small QP. Label this implementation an equivalent
Gaussian OC-SVM solver, never an independently executed MATLAB QP. Any full-fit
truncation is forbidden. Native shallow AUROC uses held-out9000, not10000.

## DROCC: CIFAR author source and transparent epoch selection

Selected source target `DROCC_author_CIFAR_Table11`. All10 classes, full5000
normal train and10000 test. Three declared repeats0,1,2 (original repeat count
unresolved). Supplement Table11 was visually checked from its original PDF.
Per-class optimizer/lr/ascent/radius/mu values are in run_plan.json; gamma1.
`main_cifar.py:14-43`: native CNN32/64/128, embedding128, head128->64->1,
all linear/conv bias-free, BN eps1e-4 affineFalse, leaky0.01. No ImageNet/AE.
`process_cifar.py:61-80`: ToTensor then mean[.4914,.4822,.4465],
std[.247,.243,.261]; targetnormal1; class-subset train, fulltest.

`main_cifar.py:73-93` uses batch128,100epochs, SGD momentum0 or Adam with no
weight decay; --reg is not passed. Epoch LR uses thresholds40% and80%
(`:45-69`). `only_ce_epochs=0` is explicitly passed. Parser ascent100 is not
forwarded: trainer default50 is effective. `drocc_trainer.py:158-214` initializes
Gaussian perturbation, ascends BCE with input-gradient unit normalization,
projects every10 steps into image-space [r,gamma*r], with BN still in train mode.
Zero-gradient behavior must not be silently changed. Loss is positive BCE +mu
adversarial BCE. Upstream loss/gradient/one-step fixtures are mandatory.

The runner passes its testloader into trainer validation. `drocc_trainer.py:
107-114` keeps best test AUC and overwrites finalmodel. Save both best-test-selected
and fixed-final-epoch checkpoints/predictions from the same trajectory; label
first diagnostic historical selection and second changed evaluation. Normal
logit scores become negative logits for anomaly1, without changing AUC.
No final-test tuning beyond this explicitly labeled historical source behavior.
Other tabular/ImageNet10 targets are coverage-audit-only, not reproduced claims.

## Controlled MVTec author-encoder comparison v3

All15 categories x seeds0/1/2, reuse byte-identical v2 image split IDs. Author
1024D extractor only; no reuse of v2 feature values. Old NBD formulas and settings
from OCC base configs/full.json, models.py and math.py stay frozen, with dim1024.
This is a post-result correction, not a pristine holdout or proven causal fix.
All controlled methods use top1%mean pooling (8 of784 patches); native PatchCore
uses max. NBD component calibration alone consumes its reserved calibration split.
Thresholds use separate image calibration with alpha.05, k=ceil((n+1)*.95),
infinity if k>n, strict greater comparison. No test threshold optimization.

Include full NBD B+A+D+F, all old ablations, PatchScore with exactly NBD means,
byte-matched selected normal patches, and author coreset memory on same FIT images
with same pooling. Retain old shallow/head adaptations with explicit shared-feature
labels where required by the task; they never enter native tables. Method memory
and shared frozen backbone bytes/latency are recorded separately; matching bytes
excludes allocator/framework overhead and is not matching peakGPU memory.

## Verification and resources

Before expensive runs: lock matrix/config/source; real-image feature/layer/NN/map
parity; native loss/gradient/optimizer checks; checkpoint/resume and score replay;
small predeclared execution smokes without score-based config selection.
Use FP32, no AMP/TF32 or gradient accumulation. Record actual batches/epochs/LR,
CUDA versions, wall time, peak memory and effective parameters.
Initial planning estimate: tens of GPU-hours for400 deep runs +30 adversarial
runs +45 native and45 controlled MVTec fits; refine from fixed smoke timings,
not test performance. Available RTX3090 has24GiB VRAM, host125GiB RAM and32GB
disk. Stream one category/cache at a time and copy completed checkpoint archives
to local storage with checksum verification before any storage recycling.
MVTec archive5.26GB plus extracted data, model states and cached descriptors are
budgeted explicitly by the runner. Never truncate epochs/categories due to budget.
Independent metrics use exported sample IDs, original/anomaly labels, scores and
selection policy. Every completed row has a checkpoint replay and coverage entry.
Unfinished/failed rows remain visible. Preserve v2 bytes and poor results.

### Additional startup behavior found by tracing the actual PatchCore runner

`backbones.py:50-51` returns a fresh train-mode torchvision model. Neither
`bin/run_patchcore.py:293-314` nor `PatchCore.load` switches it to eval before
`common.py:270-273` probes dimensions with one all-ones tensor. The probe can
update BN running statistics once. Preserve this author startup behavior, then
freeze/eval for all real-image extraction. Record raw V1 weight hash and effective
post-probe backbone-state hash separately. The parity gate must compare to this
actual released initialization, not to an independently pre-eval'd model. This is
a newly found source detail, not an alternative chosen based on test scores.

Original Theano dependency probe: its exact core lock cannot resolve because
Lasagne==0.2.dev1 is unavailable from the package index. The complete lock also has
no compatible cvxopt1.1.9 wheel in the current Python3.12 environment. This does
not establish that an expert could never reconstruct a legacy environment; it
establishes that the historic locked runner has not been executed here. The
explicit author-PyTorch fallback is used, with no claim of historical numeric parity.

Deep fixture results: six architecture/objective cases compare gradients and all
parameters/BN buffers against unchanged upstream trainers on the same GPU runtime.
All pass tolerance1e-7; six two-epoch interrupted/resume checks also pass.
Full real-data MNIST smoke runs one AE epoch and one epoch of each objective and
replays all10000 test scores exactly. These are execution tests, not benchmark rows.
Initial MNIST timing: about0.22 seconds per SVDD epoch on5923 images,30 batches.
This is a planning observation only; full native/controlled timing is still measured.

DROCC full-data smoke:5000 normal images,40 optimizer batches,50 input ascent
steps per batch; one epoch took11.53 seconds while a Deep SVDD process also ran.
All10000 scores replayed with maxerror0 for both selection exports. Native30x100
DROCC epochs alone therefore represent about9.6 observed GPU-process hours before
I/O/overlap effects; this estimate is not a standalone inference latency result.
CIFAR Deep smoke also passed both objectives on all10000 test images, replay0.
No smoke AUROC was used to choose a class, configuration, epoch count or rerun.

User's clarified presentation scope: the single common benchmark is MVTec AD,
showing NBD and explicitly named shared-feature baselines on identical splits.
Native dataset results appear with each method's reproduce slides. They are not
merged into a cross-dataset model ranking.

### Identifiable Tax--Duin historical target and remaining recovery gap

The author-uploaded2004 article identifies Table2 (printed pp59-60), Gaussian
SVDD on the three Iris classes, as a concrete historical target. It describes
10-fold evaluation and choosing kernel width to produce approximately10% support
vectors. The exact folds/seeds, width-search procedure/tolerance and complete
numeric optimizer settings are not recovered in the current maintained toolbox.
Those missing choices affect a tiny dataset substantially; we do not invent them
and label an arbitrary Iris run a reproduced Table2 result. Historical numeric
reproduction remains blocked; the independently checked QP and Ruff image-table
Gaussian solver are separate targets.
Source: https://www.researchgate.net/publication/226109293_Support_Vector_Data_Description
(author-uploaded full text), Section3.2/Table2; bibliographic confirmation:
https://research.tudelft.nl/en/publications/support-vector-data-description/.

### User-authorized budget amendment (14 September 2026)

The user subsequently asked to reduce epochs, rejected the ~10-hour schedule,
and selected **about two hours remaining**. This direct request supersedes the
full-epoch/no-truncation wording above and in the attached request. The original
`run_plan.json` and its existing output remain preserved. New executions use
`OCC_RUN_PLAN=run_plan_budgeted.json` and separate target/output identities.

Native Deep SVDD: AE5 + SVDD12; original learning-rate milestones and ten-epoch
soft-boundary warmup retained, so radius updates occur in the final two epochs.
Native DROCC:5 epochs, still50 ascent steps/projection every10, source learning-rate
fractions applied to5 epochs. Controlled heads: AE5, Deep15, DROCC15 (ten warmup
plus five adversarial epochs). All selected datasets/classes/categories/seeds
remain in the manifest. Any rows unfinished when the wall-clock budget expires
are explicitly incomplete. No low scores trigger reruns or hyperparameter changes.
These are reduced-epoch method replications, not full-schedule paper reproductions.

PatchCore has no optimizer epochs and retains its complete10% coreset procedure.
A separate real-image fixture now compares direct upstream `fit(DataLoader)` with
cached fitting: selected indices, selected descriptors and post-selection NumPy,
Torch and CUDA RNG states match exactly. The native cached implementation therefore
preserves the source sampling path despite avoiding repeated CNN extraction.

Shallow fixtures execute the extracted original MNIST/CIFAR split and test-holdout
blocks with only explicit Python2 range/integer-division modernization. Six split
and RNG cases pass, source PCA matches exactly, and an independent OSQP Gaussian
SVDD dual matches normalized libsvm coefficients within1e-6. This confirms the
modern solver/protocol adaptation; it does not execute MATLAB or reproduce Tax2004.

All20 author precomputed GCN min/max pairs have now been checked against every
original normal training image in MNIST/CIFAR, using float64 GCN for the provenance
calculation. All match within1e-4; the actual training transforms still import the
unchanged source constants. File hashes/counts are recorded in native_data_audit.json.

Independent CSV rank metrics exposed a separate float64 serialization issue in
shallow results: pandas' default parser can change values around1e-16 and merge
nearly tied kernel scores, shifting AUROC around1e-7. The exporter now requests
round-trip parsing and asserts exact equality with original score arrays. Older
completed result metadata is corrected only after a saved-estimator replay; old
metrics remain embedded in each result JSON. Predictions, trained model, gamma
selection and metric tolerances are unchanged. This is an output-parsing repair.
