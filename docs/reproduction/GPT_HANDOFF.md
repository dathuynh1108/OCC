# OCC reproduction handoff for slide update

Use only the verified results below. Keep Phase 1 (31-run matched forensic replay) separate from Phase 2 (mapping and descriptor-parity status). Do not invent S0--S3 scores or present Phase 2 as a completed candidate-assignment experiment.

## Verified Phase 1: MVTec author-encoder controlled run

Scope: 31 of 45 completed category/seed entries had both `scores.npz` and `geometry.pt`. All values below are matched observed-run means over those 31 runs, not a result for all 45 runs.

| Comparison | Image AUROC change (pp) | Interpretation permitted |
| --- | ---: | --- |
| Raw Euclidean same center -> raw bubble B | -18.69735 | Observed compatibility-score plus candidate-assignment gap; not a coding-defect finding. |
| Raw B -> calibrated Bubble B | -0.824935 | ECDF calibration has a smaller measured effect. |
| Bubble BA -> Bubble BAD | -2.610233 | D hurts this fixed matched coverage. |
| Bubble BA -> Bubble BAF | +1.859775 | F has mixed benefit in this fixed protocol. |
| Bubble BAF -> NBD | -3.081382 | Adding D to BAF hurts this fixed matched coverage. |

Raw Euclidean same-center AUROC is 0.941131 and raw B is 0.754158. Raw D and F AUROCs are 0.522997 and 0.582182. The Phase 1 suite had 23 passing tests, including the CUDA BubbleModel path. It found no implementation, numerical, protocol, or metric-plumbing issue on the checked paths. This does not prove the whole implementation bug-free.

## Phase 2: geometry vs candidate-assignment status

- Checkpoint/raw-score hashes verified: 31/45 planned runs.
- FIT ownership derived exactly for all 31 verified checkpoints: 3,826,704 patch rows; 784 patches/image.
- Fixed sentinels: `bottle/seed-0`, then `metal_nut/seed-0` from deterministic NumPy RNG seed `20260914`.
- Local original MVTec data matches both historical sentinel manifests.
- Neighborhood summary across 3,968 bubbles: mean dominant-image fraction 0.120338; mean unique images 48.145. This describes sample ownership only and does not explain AUROC loss by itself.
- S0 (Euclidean same center), S1 (energy at that center), S2 (minimum energy in source top-K), S3 (minimum energy over all bubbles), patch witnesses, and covariance sensitivity are **not run**. The historical runs did not persist query descriptors, and the available host had no CUDA/Vast instance for source-faithful frozen encoder inference.

The Phase 2 conclusion is **blocked evidence**, not an implementation, numerical, or statistical-model conclusion. Do not add a numeric S0--S3 table to the slide.

## Exact next action

On an existing CUDA host, run the source-faithful frozen exporter for only the two locked sentinel categories. It saves nothing unless the ordered descriptor array exactly matches the historical `feature_sha256`; then run the assignment audit, which first replays S0 and S2 against historical raw arrays before permitting metrics.

```bash
python -m reproduction.export_author_descriptors \
  --category bottle --category metal_nut \
  --selection-plan results/geometry-assignment-forensics-2026-09-14-local-mapping-v4/selection_plan.json \
  --output results/phase2-descriptors-<run-id>

python -m reproduction.nbd_forensics.geometry_assignment \
  --descriptors results/phase2-descriptors-<run-id> \
  --output results/geometry-assignment-forensics-<run-id>
```

## Included evidence

- `phase1_REPORT.md`: full Phase 1 replay findings and limits.
- `phase2_REPORT.md`: current Phase 2 status and blocker.
- `phase2_selection_plan.json`: sentinel selection before labels/metrics.
- `phase2_dataset_preflight.json`: dataset-manifest parity.
- `phase2_source_and_feature_parity.json`: source/checkpoint/feature provenance.
- `phase2_neighborhood_image_diversity.csv` and `phase2_bubble_covariance_stats.csv`: compact mapping evidence.
- `review.pdf` and `review.tex`: current paper-faithful slide artifact/source.
