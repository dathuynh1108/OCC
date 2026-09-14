# Phase 2 geometry-assignment forensic

The canonical local artifact is `results/geometry-assignment-forensics-2026-09-14-local-mapping-v4/`; results remain ignored because `fit_patch_ownership.csv` contains 3,826,704 rows.

The audit verified 31 of 45 planned controlled runs against their `COMPLETE.json` hashes. It derived FIT ownership for all 31 using the saved split order and the recorded 784 patches per image. The two sentinels were fixed before descriptor or label content was read:

- A: `bottle/seed-0`
- B: `metal_nut/seed-0`, drawn at index 18 from the other sorted eligible runs with `numpy.default_rng(20260914)`

The local MVTec archive was extracted to `data/mvtec_ad`; both sentinel manifests match the persisted feature manifests. The local host has no CUDA device and `vastai show instances-v1 --raw` returned no active instance. The original artifacts do not contain query descriptors, so S0--S3, raw-B replay, patch witnesses, and covariance sensitivity controls were not run. This is a feature-parity blocker, not evidence of an implementation or scoring defect.

On an existing CUDA host, export only the locked sentinel descriptors, then run the assignment audit:

```bash
python -m reproduction.export_author_descriptors \
  --category bottle --category metal_nut \
  --selection-plan results/geometry-assignment-forensics-2026-09-14-local-mapping-v4/selection_plan.json \
  --output results/phase2-descriptors-<run-id>

python -m reproduction.nbd_forensics.geometry_assignment \
  --descriptors results/phase2-descriptors-<run-id> \
  --output results/geometry-assignment-forensics-<run-id>
```

The exporter performs frozen author-encoder inference only and refuses to save unless the complete ordered descriptor array equals the historical `feature_sha256`. The audit then verifies S0 and S2 against saved raw scores before it permits metrics or labels.
