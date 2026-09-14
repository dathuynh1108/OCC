# Validation evidence

GPU preflight on 2026-09-14: NVIDIA GeForce RTX 3090 Ti, 24564 MiB, driver 595.58.03; Python 3.12.14; torch 2.11.0+cu128; torchvision 0.26.0+cu128. A real CUDA matrix multiplication completed with finite values.

`CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python -m pytest -q`: **16 passed**, 2 OSQP PendingDeprecationWarnings, 4.88 s, on the training GPU and Python environment. No CUDA tests were skipped.

Coverage includes disjoint/exhaustive image splits; original dataset counts; exact normal tails including ties; finite-sample image thresholds; diffusion versus transition-distribution distance including disconnected graphs; tangent-frame invariance; bubble energies versus explicit projectors; true SVDD dual constraints; DROCC analytic gradient versus autograd and annulus bounds; deep-head bias/center constraints; complete epoch patch coverage; checkpoint/optimizer/RNG resume; and a complete synthetic-fixture score/prediction/replay workflow. Synthetic fixtures are tests, not benchmark results.

The first test run found an incorrect removal of an arbitrary stationary eigenvector for disconnected graphs. The corrected implementation removes the known constant eigenfunction explicitly before diagonalizing the complement. This was corrected and retested before the full benchmark began.

`ruff check nbdbench tests`, `bash -n scripts/*.sh`, `shellcheck scripts/*.sh`, `shfmt -d scripts/*.sh` all passed before launch.

Full-run status and final checkpoint replay evidence are reported only by generated benchmark verification files.

## Normal-only numeric correction before the final run

The initial full-v1 execution was invalidated after inspecting only fitted normal
geometry: 116 of 128 stored frame self-distances in bottle/seed-0 were nonzero,
up to 2.47955322265625e-05. A separate synthetic float32 regression then failed
on the same mathematical invariant. No test AUROC was used to choose this fix.

The corrected routine computes the Frobenius distance of the actual stored
projectors using float64 contractions and an exact zero diagonal. This avoids
substituting ideal integer ranks for slightly non-orthogonal float32 matrices.
All backbones, input data, splits, seeds, budgets and training hyperparameters
remain unchanged. Full-v1 artifacts and source are retained; **full-v2 starts
all 45 runs from scratch** and is the only candidate for the final report.

After this correction, the identical CUDA test command passed **17 tests**, zero
skips, two OSQP PendingDeprecationWarnings, in 4.93 s. Ruff and shell checks passed.
A separate normal-training-only numerical preflight also passed for toothbrush
and transistor, including local PCA rank 16 and positive DROCC radii. These checks
are not counted as benchmark fits or scores.
