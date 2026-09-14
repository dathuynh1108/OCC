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
