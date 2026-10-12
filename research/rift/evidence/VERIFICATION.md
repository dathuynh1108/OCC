# Verification record — 2026-10-13

## Executed

- `python -m pytest research/rift/test_rift.py -q`: **20 passed**.
- `python -m compileall -q research/rift`: passed.
- `python -m research.rift.run --help`: passed.
- Graph probe: seeds 0,1,2; conductance noise 0,0.35; 32 fit normals,
  64 calibration normals, 64 examples per test kind. See synthetic_run.json.
- Full image CLI smoke: two seeds, generated random RGB images in LOCO directory
  layout, RGB debug backbone. Validated cache, fit/cal/test flow, model-state files,
  scores, grouped metrics, audit JSONL and seed summary. NOT a dataset result.
- Supplied graph JSONL/NPZ path exercised in the test suite.
- ResNet-18 forward exercised with random weights ONLY in a unit test; no
  download, pretrained-feature result, training convergence or CUDA claim.

## Numerical correction caught by the probe

A mathematically identical roleless graph descriptor differed by 2.78e-16.
Ranking that roundoff produced a spurious perfect AUROC on the degenerate clean
probe. NormalMemory now sets RMS distances below 1e-12 to zero for ALL methods.
A regression test verifies that such descriptors have equal zero scores. The
committed metrics were regenerated after this fix; the roleless clean AUROC is 0.5.

## Interpretation

RIFT detects both designed graph errors. The all-role diffusion and shortest-path
baselines also detect them perfectly. There is NO demonstrated advantage over
those baselines. Endpoint-only transfer fails the mediator-routing task (AUROC
0.5), which verifies what information the tensor adds to that weaker descriptor.

Global conductance scaling by 1.7 on valid normals remains a severe failure:
RIFT FPR is about 97.4% under noisy graphs at a normal-calibrated nominal 5% level.
This stress outcome is retained rather than hidden by test-time retuning.

## Not executed

No actual MVTec images or pretrained checkpoint were available in this runtime.
Torch reports no CUDA device. GPU execution and real-image performance remain
unverified. The existing OCC root suite was not run: the runtime has a standalone
new-folder snapshot, and network git clone failed (DNS). The repository was
inspected through the authenticated GitHub connector. No existing files change.
Independent agent code review was not available; tests and self-review only.

## Reproduction

Run the commands in README.md. Raw per-seed/per-image graph scores are generated
by pilot.py; the downloadable handoff archive also includes the measured raw CSVs.
No generated dataset images, cache, model weights or large arrays are committed.
