#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-science.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c 'import torch,torchvision; assert torch.cuda.is_available(); x=torch.randn(256,256,device="cuda"); assert torch.isfinite(x@x).all(); print(torch.__version__,torchvision.__version__,torch.cuda.get_device_name())'
