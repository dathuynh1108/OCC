# RIFT pilot implementation plan

## Scope and hypothesis
Role-conditioned Interventional Flow Tensor (RIFT): compare normal source–target
transport and the response to weakening individual edges, pooled by mediator role.
This is a proposed descriptor, not an established novel method or a SOTA claim.
Keep all existing OCC experiments unchanged. Everything lives under research/rift.

## Tasks
1. Write numerical tests for the exact finite-edge resolvent update, node
   permutation invariance, disconnected graphs, and role-sensitive routing.
2. Implement graph descriptors and identical normal-only nearest-neighbor scoring.
   Include direct-role adjacency, shortest paths, endpoint and all-role transfer,
   roleless spectra, mediator-collapsed response, and RIFT.
3. Implement a deterministic graph mechanism probe with clean/noisy observations,
   independent fit/calibration/test generators, and explicit synthetic provenance.
4. Implement an image-to-graph pilot for MVTec LOCO with frozen torchvision
   features, fit-only role proxies and bandwidth, CPU/CUDA support, content-addressed
   feature caching, normal-only calibration, per-kind metrics, raw scores and audit.
5. Run tests and CPU probes. Verify real-image plumbing on generated image fixtures;
   do not represent them as MVTec results. Publish code and measured evidence in a
   new branch and PR in dathuynh1108/OCC.

## Protocol constraints
No test-label tuning; no anomaly training; no silent random pretrained fallback.
No test masks used for graph construction. Image roles are learned proxies, NOT
certified physical endpoints. Graph-oracle results are NOT image AD performance.
No dataset/checkpoint uploads. Record actual dependency versions and commands.

## Review focus
Disconnected/zero graphs; negative/nonfinite conductance; zero-mass roles;
missing normal calibration; single-class metrics; cache parameter changes;
re-running into an existing output directory; CUDA request on CPU-only hosts.
