# Rerunning the source-audited experiments

Start at `results/reproduction-2026-09-14/REPRODUCTION_REPORT.md` and
`SLIDE_UPDATE_HANDOFF.md`. The supplied slide source is preserved as
`slides/paper-faithful-review/review.input.tex`; the updated deck is `review.tex`.

## Environment and inputs

Use Linux, Python3.12 and a CUDA12.8-capable NVIDIA environment. The measured host
was one RTX3090. Install the matching CUDA Torch build before the other packages:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-reproduction.txt
python -m reproduction.fetch_sources
```

Exact installed versions are also retained in the runtime evidence. Source
fetching pins all five upstream commits and skips PatchCore's result-model LFS
files. The runners verify source file hashes; upstream licenses remain in vendor/.

Place the official MVTec archive in `data/mvtec_anomaly_detection.tar.xz`, verify
SHA256 `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`,
then extract into `data/mvtec_ad/` (category directories directly below it).
The official dataset entry is https://www.mvtec.com/company/research/datasets/mvtec-ad.
Preserve the original dataset license; images are not redistributed in Git.

Place official torchvision ImageNet1K V1 weights at
`artifacts/backbone/wide_resnet50_2-95faca4d.pth`. Verified SHA256:
`95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`.
Download source: https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth.
Do not substitute DEFAULT/V2.

MNIST/CIFAR native runners download via torchvision into `data/native/`. If the
Toronto CIFAR endpoint is throttled, the measured byte-identical mirror was
https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz.
Only admit it after checking official archive MD5 `c58f30108f718f92721af3b95e74349a`
and SHA256 `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`.
Save as `data/native/cifar-10-python.tar.gz`; torchvision validates/extracts it.

## Verification gates

```bash
python -m reproduction.verify_deep
python -m reproduction.verify_drocc
python -m reproduction.verify_patchcore
python -m reproduction.verify_patchcore_selection
python -m reproduction.verify_shallow
python -m reproduction.deep_svdd --dataset mnist --normal-class 0 --seed 1 --smoke
python -m reproduction.deep_svdd --dataset cifar10 --normal-class 0 --seed 1 --smoke
python -m reproduction.drocc --normal-class 0 --seed 0 --smoke
python -m reproduction.mvtec --smoke
```

Smoke outputs are separate and never enter the benchmark table. Some parity
fixtures intentionally refuse overwriting an existing fixture directory: retain
or move that evidence before a genuinely new validation execution.

## Measured budgeted configuration

```bash
export OCC_RUN_PLAN=run_plan_budgeted.json
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m reproduction.mvtec
python -m reproduction.run_deep_matrix
python -m reproduction.run_drocc_matrix
python -m reproduction.run_shallow_matrix
```

The commands are sequential here for simple reruns. The measured run overlapped
GPU tracks and used four isolated CPU libsvm processes; source-level math and
batch sizes stayed unchanged. Full matrix size:45 native PatchCore,400 Deep
objective results,60 DROCC selection exports,400 shallow nu results and495
common MVTec metric rows. Check `coverage.csv` before calling any target complete.

Native Deep AE5+SVDD12; native DROCC5; common AE5/Deep15/DROCC15. Source milestones,
soft-boundary warmup and ascent iterations remain explicit in the plan. These
short schedules were selected by the user for runtime, not tuned to test scores.

For the original long schedules, unset `OCC_RUN_PLAN` and use the same commands.
The immutable `run_plan.json` selects separate full-schedule Deep/DROCC/common
output identities. PatchCore and shallow are unchanged non-epoch targets and
may reuse a completed identical result. Expect substantially longer training.
Do not change a config in place under existing output: use a new target/output
identity or an independently preserved checkout/output directory.

## Export and slides

After the final result files are stable and copied locally:

```bash
python reproduction/summarize.py --require-common-complete
python reproduction/render_delivery.py
cd slides/paper-faithful-review
tectonic --keep-logs review.tex
```

The summarizer independently recomputes image AUROC/AP/FPR/TPR, enforces all common
category/seed/method groups and emits explicit native coverage. Balanced native
seed repeats are summarized; partial repeats remain in the per-class table.
Large `.pt`, `.npz`, `.joblib`, datasets and original weights stay in the verified
local backup, outside Git. Predictions, losses, configs and SHA256 manifests let
another reader audit results or retrain from the repository alone.

The user requested deleting the rented instance after local backup verification,
complete slides and Git push. Instance lifecycle commands must target the actual
rented instance; no credentials or machine-specific SSH endpoints are stored here.

Additional integrity gates and isolated timing:

```bash
python -m reproduction.audit_native_data
python -m reproduction.measure_inference --validate-only
# After GPU training ends, measure the declared single-image/category scope:
python -m reproduction.measure_inference
```

`verify_csv_roundtrip.py` repairs historical exports from the initial default
pandas CSV parser, retaining old metrics and replaying the stored estimator.
New shallow exports already use exact round-trip parsing; no repair is needed.
The independent summarizer will fail rather than silently accept changed ties.
