# Run and audit the experiment again

The Git repository contains the implementation, fixed configuration, tests and
small final evidence files. The local delivery also retains the original dataset
archive, complete per-run artifacts, executed source and backbone weights. Those
large files are ignored by Git; cloning the repository alone does not download
the trained models or dataset.
The supplied PDF/handoff and downloaded primary papers are retained locally in
`artifacts/source-documents/`, with their hashes in `docs/source_provenance.json`.

## Fresh training on a Linux NVIDIA GPU

Use Python 3.12 and a driver compatible with the pinned CUDA 12.8 wheels. The
recorded experiment used one RTX 3090 Ti with 24 GB VRAM. Install `uv`, clone this
repository, and run from its root:

```bash
bash scripts/bootstrap_linux.sh
bash scripts/run_full.sh
```

The script checks CUDA, runs the equation regressions, downloads and verifies the
original MVTec AD archive, extracts it, and runs all 15 categories and all three
seeds. It then builds the report and performs a separate full feature/checkpoint
replay. Allow room for the 5.3 GB compressed dataset, extracted images, backbone
and several GB of run artifacts. Feature arrays are held in RAM one category at
a time, rather than saved as a full dataset cache on disk.

The final model for each neural stage is its prescribed terminal epoch. There is
no early stopping. Head batch size is 8192; autoencoder/SVDD/DROCC epoch counts
are 50/100/100, with every fit patch included in every epoch. The fixed 2048-point
RBF SVDD coreset is an explicit algorithm adaptation, not an omitted training
stage. Read [METHOD_PROTOCOL.md](METHOD_PROTOCOL.md) before comparing these
numbers with a paper's original benchmark table.

## Reuse the delivered local data and weights

Copy these paths from the local delivery to the same relative paths in a clone:

- `data/mvtec_anomaly_detection.tar.xz`
- `outputs/full-v2/` for the completed fitted models, predictions and histories
- `artifacts/backbone/wide_resnet50_2-95faca4d.pth`

The downloader recognizes a complete archive with the expected SHA-256 and
extracts it without fetching it again. Put the backbone in PyTorch's checkpoint
cache before an offline run:

```bash
mkdir -p "${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints"
cp artifacts/backbone/wide_resnet50_2-95faca4d.pth \
  "${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints/"
.venv/bin/python -m nbdbench.download --root data
```

The original archive SHA-256 is
`cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`.
The backbone SHA-256 is
`95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`.

## Recompute evidence without retraining

The published CSV table can be checked on a CPU without installing the training
dependencies:

```bash
python3 scripts/audit_published_results.py results/mvtec-full-v2
```

This independent standard-library implementation counts tied positive/negative
pairs for AUROC and threshold groups for AP. It checks the complete original test
path/label set, duplicate rows, confusion counts, threshold decisions and the
reported category-macro means/sample SD. This verifies the exported table; use
the GPU replay below to verify feature extraction and checkpoint predictions.

To recreate the supplemental category plot with a full 0–100 color scale, use
the training environment's plotting dependencies:

```bash
.venv/bin/python scripts/plot_category_results.py results/mvtec-full-v2
```

Use the same source revision and configuration recorded by `protocol_lock.json`.
Set the same numerical environment as the original job:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
.venv/bin/python -m nbdbench.report --output outputs/full-v2
.venv/bin/python -m nbdbench.verify --output outputs/full-v2 \
  --data data/mvtec_ad --replay
```

The first command validates run artifact hashes, recomputes AUROC/AP from saved
image predictions, checks epoch completeness and rebuilds the tables. The second
regenerates the complete CNN descriptor cache, checks its hash, reloads the saved
models and compares every score-calibration, threshold-calibration and test score.
Exact cache equality is a strict check; a different GPU/software stack can fail
that check because of numerical differences. Do not relabel a failed check as a
successful reproduction.

`scripts/run_full.sh` reuses completed runs only after checksums pass and resumes
an unfinished neural stage at a saved epoch boundary. To force a genuinely new
training run, choose a new output directory instead:

```bash
.venv/bin/python -m nbdbench.run --config configs/full.json \
  --data data/mvtec_ad --output outputs/repeat-001
.venv/bin/python -m nbdbench.report --output outputs/repeat-001
.venv/bin/python -m nbdbench.verify --output outputs/repeat-001 \
  --data data/mvtec_ad --replay
```

Always use a new directory after changing source, configuration, dataset or
backbone. Keep prior results and their provenance; do not overwrite an older
experiment to conceal a protocol change. The discarded development run
`full-v1` is described in [VALIDATION.md](VALIDATION.md) and contributes no rows
to the final `full-v2` table.
