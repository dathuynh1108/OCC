# MVTec AD: light image-method comparison

All 15 categories, the same 1,725 test images, seed 0. Image AUROC/AP in percent.

| Method | AUROC | AP |
| --- | ---: | ---: |
| RBF_SVDD_nu_0.01 | 53.09 | 78.18 |
| RBF_SVDD_nu_0.1 | 53.21 | 77.91 |
| DeepSVDD_one-class | 57.61 | 80.43 |
| DeepSVDD_soft-boundary | 57.96 | 80.42 |
| DROCC | 64.10 | 83.96 |
| PatchCore | 99.18 | 99.77 |
| Bubble_A+D | 73.56 | 88.01 |
| Bubble_B+D | 73.09 | 87.61 |

## What was run

75 primary results: 30 RBF SVDD fits, 30 Deep SVDD objective results, and 15 DROCC runs. The 15 fixed-final DROCC exports are retained separately. PatchCore and NBD are reused from verified seed-0 predictions; their averages across three seeds are not mixed into this table.

The new CNNs are the authors' native image networks, trained from scratch on MVTec inputs. They are not the earlier WR50 feature heads. This is a light application to MVTec, not a claim to reproduce a published MVTec result for these methods. See `../../docs/reproduction/MVTEC_IMAGE_LIGHT.md` for the locked preprocessing, model and selection choices.

Deep SVDD AE5/SVDD12; DROCC5. No full-schedule run was required. DROCC uses the source best-test-AUC selection in this table; a fixed-final export is also saved. No hyperparameters were selected from these measured scores.

## Verification

- All original input image hashes verified before creating the 32-pixel cache.
- Deep SVDD: six source loss/gradient/state cases and six interrupted/resume cases passed on the 4090.
- DROCC: Adam and SGD source parity/resume cases passed with zero measured loss/gradient/parameter/BN error.
- Shallow: source preprocessing/split and independent QP checks passed.
- All three execution tracks completed all 15 categories with exit code 0.
- Fresh-checkpoint prediction replay passed before each result was finalized.
- All 45 category/track artifact manifests verified on the GPU machine.
- Local independent recomputation checks 120 method/category metric rows, identical test IDs/labels and full native train splits. See `comparison_verification.json`.

## Checkpoint storage

Metrics, predictions, configs, histories and parity evidence are backed up locally with a verified archive hash. Full optimizer/model checkpoints and the complete 487 MB archive remain on instance 51088289. The full checkpoint archive has NOT been downloaded locally: transfer was stopped to avoid paying for idle compute. Stop preserves the instance filesystem; do not destroy it before deciding whether those checkpoints are still needed. `backup_receipt.json` records both archive hashes.
