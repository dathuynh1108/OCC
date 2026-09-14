#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
nbd_python="${NBD_PYTHON:-.venv/bin/python}"
"$nbd_python" -m pytest -q
"$nbd_python" -m nbdbench.download --root data
"$nbd_python" -m nbdbench.run --config configs/full.json --data data/mvtec_ad --output outputs/full-v1
"$nbd_python" -m nbdbench.report --output outputs/full-v1
