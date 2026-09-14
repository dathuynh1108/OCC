# Active paper-source reproduction

Status: source audit and runner implementation; no new benchmark scores yet.
Base: 91223d63adc3289786bbdeff16d5feeb59aeb3d7. Branch codex/source-faithful-reproduction.
Historical results/mvtec-full-v2 must remain byte-identical.
User authorized the newly rented Vast GPU, code/results push, local backup, stop GPU after verified backup.
New supplied Overleaf source extracted under slides/paper-faithful-review; input zip SHA256 341b55b0709dfcc9b9e4f2af6e28484bd87ccce9038472da5f5801d0c2dbfb27.
All five upstream commits checked out under ignored vendor/. PatchCore LFS model downloads were canceled; source restored from exact commit without pretrained result files.
GPU 50996199 RTX3090 24GB, Python3.12.14 / Torch2.11.0+cu128 / torchvision0.26.0; no MATLAB/conda/micromamba observed. CUDA matmul passed.
Dataset transfer started; no benchmark running. Download original PatchCore PDF in progress.
Remaining: finish audit/source lock/run plan; numeric parity + smoke; run native & controlled full matrix; replay/independent metrics; slide update/compile; code+all evidence local and git; verify backup; STOP (never destroy) GPU.

## Latest progress

User clarified: one shared MVTec comparison table for NBD/all baselines; native
MNIST/CIFAR results belong only to each model's reproduce slides. Accepted.

Commit25eff28 locks audit/plan/Overleaf inputs/Deep SVDD runners. Not pushed yet.
Source plan SHA39019ebc0e3a8b57e8c52263c051b22778372a6119f51e6c84449b579649316d.
920 planned matrix rows:45 PatchCore,400 Deep SVDD,400 shallow (two nu),30 DROCC,
45 controlled (each contains all methods). No matrix rows dropped.

CURRENT GPU WORK: supervisor program occ-deep-native is running the complete
Deep matrix (MNIST then CIFAR). Source module loss/trainer state code frozen at
commit25eff28. Configs use complete author Torch README150/150MNIST and350/150CIFAR,
NOT paper schedule; this explicit later-author target is documented. Original exact
Lasagne0.2.dev1 cannot resolve; historical Theano numeric parity not claimed.

Deep fixture6 cases passed all parameters/BN/gradients <=1e-7;6 resume cases pass.
MNIST full-data smoke1AE+1eachobjective passed,10000test scores replay maxerror0.
Deep complete full runs beginning to appear under results/native/... on GPU.
MVTec5.26GB archive verified/extracted on GPU; disk11GB used/22GB free at that point.
CIFAR original Toronto download throttled. Byte-identical mirror download verified
MD5c58f30108f718f92721af3b95e74349a against torchvision official source; SHA256
6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce.
URLhttps://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz.
Partial Toronto downloads preserved *.partial; their download-only jobs stopped.
New CIFAR smoke then DROCC smoke are running via SSH exec session (see tools).

New uncommitted modules patchcore.py/verify_patchcore.py and drocc.py/verify_drocc.py.
PatchCore fixture on real bottle train+anomaly images PASSED: layers, independent
Unfold/MeanMapper/Aggregator, fixed memory FAISS/image/maps, batch1vs2, cache exact.
NEW critical source fact: upstream backbone is TRAIN during initial all-ones shape
probe (common.py270-273), updating BN once. Preserve this, THEN freeze/eval. Both
raw V1 and effective post-probe state hashes exported. Do NOT pre-eval before load.
Native cached fit must preserve RNG: upstream has train DataLoader iterator before
coreset, TEST iterator after coreset. Current extract_category extracts both first;
full runner must restore RNG from end of train or recreate only train iterator
seed before coreset. No full PatchCore runner implemented yet.

DROCC fixture PASSED Adam+SGD, full50 ascent steps:52 BCE values, gradients, weights,
BN buffers all maxerror0; two-epoch resume maxerror0. Native model/adversarialfn
import unchanged. source test-selected and final outputs fromsame trajectory.

Remaining immediately: full PatchCore+controlled runner with old v2 run_one reuse
and exact historical split check; DROCC full matrix launch after smoke; native
shallow source protocol/QP parity+full fits; build all reporting/metric audit,
slidemerging+compile (pdflatex not found locally); local backup+gitpush; stop GPU.
Dataset/native source file hashes/environment/paperprovenance still need export.
Do not mark done based on checkpoint text; only full verified coverage counts.

## Superseding live checkpoint: user-selected two-hour budget

At 09:41 UTC on14Sep2026 the user selected about2hours remaining after explicitly
requesting fewer epochs. Full supervisors occ-deep-native and occ-drocc-native
were stopped; all completed scores and last checkpoints retained. The above
full-schedule status is historical. run_plan_budgeted.json now drives separate
AE5/SVDD12 Deep targets, E5 nativeDROCC, and AE5/Deep15/DROCC15 sharedMVTecheads.
Original fullplan unchanged. MVTec15categories x3seeds, Deep2datasets x10classes
x10seeds x2objectives, DROCC10classes x3seeds stillplanned. Shallow400rows runs
seed-first acrossallclasses so balanced repeats can be reported if time expires.
Active remote supervisors: occ-mvtec-budgeted, occ-drocc-budgeted,
occ-deep-budgeted, occ-shallow-budgeted. Deadline11:41UTC includesdelivery;
reserve timeforbackup/metricverification/slides/gitpush/STOP50996199.
Shallow6source-split/RNGcases+PCA+independentQPfixture passed. PatchCoredirect-fit
versus cachedfit selection/RNGfixture passedexactly. Tectonicinstalledlocaland
warmingitsTeXbundle. No finalaggregates yet. Initial completed resultsrsyncedlocal,
but finaloffboxmanifestnotverified. Do not stopGPUbeforebackupverification.

## Lifecycle correction and code delivery

User now explicitly requested DELETE the new Vast instance after verified local
backup, complete concise slides and Git push. This supersedes earlier STOP-only.
Delete only50996199; never delete before final checksum/readback. Code commit
cc16d38 is pushed on codex/source-faithful-reproduction. Final results/main push
and deletion remain outstanding. New scripts measure_inference.py and
backup_manifest.py are not yet committed; inference replay fixture passed on
actual bottle/seed0 models. Timing itself must wait for isolated GPU.

Shallow strict replay exposed float32 PCA batch-shape variation (~2.04e-6):
replaying9000rows differed from original10000rows. Corrected replay to transform
all10000beforeselectingtest9000; exact score agreement0. No model/hyperparameter
or metric tolerance changed. Failed attempts logged; reruns now pass. FourCPU
processes run the same source protocol. Two stale failures.json entries may be
fromthepre-fixattempt; resolveagainstcompletedresults/matrix_exit ratherthan
claimingcurrentfailureorremovinghistory.
