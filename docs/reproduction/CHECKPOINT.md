# Current reproduction checkpoint

Updated 2026-09-14 10:30 UTC. Work remains active; this is not a completion report.

## Scope and decisions

- Preserve reviewed commit 91223d63adc3289786bbdeff16d5feeb59aeb3d7 and every historical v2 result.
- Native author methods use their image architectures and datasets. One separate shared MVTec comparison uses the author WR50-2/V1 1024-D descriptor and unchanged NBD math.
- User selected about two hours remaining at 09:41 UTC, explicitly allowing fewer epochs. Deadline about 11:41 UTC includes backup, slides and Git delivery.
- Measured short plan: native Deep AE5/SVDD12; native DROCC5; common AE5/Deep15/DROCC15. All classes, categories, seeds and full declared data remain selected. Original long plans and interrupted checkpoints remain separate.
- User explicitly authorized Git push and finally DELETE Vast instance 50996199, only after verified local backup, complete concise slides and successful Git delivery. This supersedes all previous STOP-only wording.

## Current runtime

RTX3090 / Python3.12 / Torch2.11.0+cu128 / torchvision0.26.0.
Source fixtures passed: Deep loss/gradient/state/resume; DROCC 50-step adversarial loss/gradient/state/resume; PatchCore descriptors, NN/maps and direct/cached coreset RNG; shallow source split/PCA and independent QP equivalence.
Data audit verified all author GCN bounds against the original training datasets.

Latest completed rows: Deep244/400, DROCC60/60, shallow165/400, native PatchCore21/45, common20/45 base groups and20/45 PatchCore groups. DROCC matrix_exit reports no failures. Others are still running.

## Corrections and provenance

- Preserve the author PatchCore startup all-ones BN shape probe, then freeze/eval.
- Shallow PCA checkpoint replay must transform all10000 test images before selecting9000 evaluated IDs, matching original GEMM shape. Fixed without changing fitted models or tolerances.
- CSV parsing must round-trip float64 values; default pandas parsing changed near-ties at about1e-16. Earlier metric JSONs were repaired only after exact saved-estimator score replay; old metric values remain recorded. No remaining old-format exports at this checkpoint.
- Stale shallow failures.json records previous failed attempts. Resolve against final matrix_exit and completed results, preserve failure history.

## Delivery status and remaining work

Branch codex/source-faithful-reproduction pushed through5731253. Uncommitted report/audit improvements and partial results remain local. Progressive result/checkpoint/data backups are copied; final manifest is NOT yet verified.

Finish matrix; run isolated inference timing; sync all final evidence; create/verify off-box SHA256 manifest. Independently audit complete predictions and metrics; generate report, reference comparisons, concise Beamer slides and Overleaf zip; compile and visually inspect PDF. Push completed branch and default main, verify Git readback. Then delete only the authorized instance and push closure evidence.

The user withdrew the unrelated Luna/cost question; do not investigate it here.
