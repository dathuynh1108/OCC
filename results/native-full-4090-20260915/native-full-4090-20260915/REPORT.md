# Full native paper-dataset reproduction

## Scope

Deep SVDD runs the released PyTorch target on MNIST and CIFAR-10 for all normal classes and seeds. DROCC runs its released CIFAR target on all normal classes and three seeds. These are source-locked reproductions; they are not a numeric claim for a different historical implementation.

## Locked protocol

- Plan: `run_plan.json` (`39019ebc0e3a8b57e8c52263c051b22778372a6119f51e6c84449b579649316d`)
- Source lock: `809666ff44691270b772d3a0228e584e98c4289265bb04a88ffe1715b03c1d6e`
- Data audit: `87037ad299a8bfd48cbeaebdd96ae345358f75320f91569730b6369243276e52`; original training images only for fitted preprocessing.
- Deep SVDD: MNIST AE 150, CIFAR-10 AE 350, then SVDD 150; normal classes 0–9; seeds 1–10; both objectives.
- DROCC: CIFAR-10 100 epochs; normal classes 0–9; seeds 0–2; frozen class-specific Table-11 settings.
- Reported DROCC row is fixed final epoch. The source best-test checkpoint is retained separately only as a diagnostic.

## Verified coverage

- Deep SVDD fixed-final exports: 400/400
- DROCC fixed-final exports: 30/30
- DROCC source test-selected diagnostic exports: 30/30
- Independent replay and tied-rank metric audit: 4600000 prediction rows checked.

## Fixed-final AUROC (%)

| Method | Dataset | Mean ± SD |
|---|---:|---:|
| DROCC | cifar10 | 65.58 ± 1.63 |
| Deep SVDD: one class | cifar10 | 61.70 ± 0.59 |
| Deep SVDD: soft | cifar10 | 60.99 ± 0.48 |
| Deep SVDD: one class | mnist | 93.83 ± 0.55 |
| Deep SVDD: soft | mnist | 93.30 ± 0.53 |

## DROCC diagnostic separation

Across the three class-macro seeds, source test-selected minus fixed-final AUROC is 7.49 ± 1.62 pp. It is excluded from the benchmark table.

## Artifacts

- `native_per_class_per_seed.csv`: fixed-final per-class/seed metrics.
- `drocc_test_selected_diagnostic.csv`: source-selection diagnostic only.
- `independent_metric_audit.json`: prediction, history, schedule, and hash checks.
- `deep_completion.json` and `drocc_completion.json`: immutable matrix completion manifests.
