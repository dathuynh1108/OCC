"""Write a concise provenance report for an audited full native matrix."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .common import ROOT, sha256


def rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def pct(value):
    return f"{float(value) * 100:.2f}"


def metric(row):
    return f"{pct(row['auroc_mean'])} ± {pct(row['auroc_sd'])}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/native-full-4090-20260915")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    audit = json.loads((root / "independent_metric_audit.json").read_text())
    if not audit["passed"] or audit["presentation_selection"] != "fixed_final_epoch":
        raise ValueError("Run the independent native audit before writing this report")
    deep = json.loads((root / "deep_completion.json").read_text())
    drocc = json.loads((root / "drocc_completion.json").read_text())
    if not deep["complete"] or not drocc["complete"]:
        raise ValueError("Native full matrix is incomplete")
    summary = rows(root / "native_summary.csv")
    expected = {
        ("DeepSVDD", "mnist", "one-class"),
        ("DeepSVDD", "cifar10", "one-class"),
        ("DeepSVDD", "mnist", "soft-boundary"),
        ("DeepSVDD", "cifar10", "soft-boundary"),
        ("DROCC", "cifar10", "standard"),
    }
    indexed = {(row["track"], row["dataset"], row["variant"]): row for row in summary}
    if set(indexed) != expected:
        raise ValueError("Unexpected native summary coverage")
    fixed = rows(root / "native_per_class_per_seed.csv")
    selected = rows(root / "drocc_test_selected_diagnostic.csv")
    fixed_drocc = {(row["class_or_category"], row["seed"]): row for row in fixed if row["track"] == "DROCC"}
    selected_drocc = {(row["class_or_category"], row["seed"]): row for row in selected}
    if set(fixed_drocc) != set(selected_drocc) or len(fixed_drocc) != 30:
        raise ValueError("DROCC diagnostic coverage mismatch")
    deltas = defaultdict(list)
    for (normal_class, seed), final in fixed_drocc.items():
        deltas[int(seed)].append(float(selected_drocc[normal_class, seed]["auroc"]) - float(final["auroc"]))
    macro_deltas = [statistics.mean(values) for _, values in sorted(deltas.items())]
    lines = [
        "# Full native paper-dataset reproduction",
        "",
        "## Scope",
        "",
        "Deep SVDD runs the released PyTorch target on MNIST and CIFAR-10 for all normal classes and seeds. DROCC runs its released CIFAR target on all normal classes and three seeds. These are source-locked reproductions; they are not a numeric claim for a different historical implementation.",
        "",
        "## Locked protocol",
        "",
        f"- Plan: `run_plan.json` (`{audit['run_plan_sha256']}`)",
        f"- Source lock: `{audit['source_lock_sha256']}`",
        f"- Data audit: `{sha256(root / 'evidence/native_data_audit.json')}`; original training images only for fitted preprocessing.",
        "- Deep SVDD: MNIST AE 150, CIFAR-10 AE 350, then SVDD 150; normal classes 0–9; seeds 1–10; both objectives.",
        "- DROCC: CIFAR-10 100 epochs; normal classes 0–9; seeds 0–2; frozen class-specific Table-11 settings.",
        "- Reported DROCC row is fixed final epoch. The source best-test checkpoint is retained separately only as a diagnostic.",
        "",
        "## Verified coverage",
        "",
        f"- Deep SVDD fixed-final exports: {deep['counts']['deep_result_rows']}/400",
        f"- DROCC fixed-final exports: {drocc['counts']['drocc_fixed_final_rows']}/30",
        f"- DROCC source test-selected diagnostic exports: {drocc['counts']['drocc_test_selected_rows']}/30",
        f"- Independent replay and tied-rank metric audit: {audit['prediction_rows_checked']} prediction rows checked.",
        "",
        "## Fixed-final AUROC (%)",
        "",
        "| Method | Dataset | Mean ± SD |",
        "|---|---:|---:|",
    ]
    labels = {
        ("DeepSVDD", "mnist", "one-class"): "Deep SVDD: one class",
        ("DeepSVDD", "cifar10", "one-class"): "Deep SVDD: one class",
        ("DeepSVDD", "mnist", "soft-boundary"): "Deep SVDD: soft",
        ("DeepSVDD", "cifar10", "soft-boundary"): "Deep SVDD: soft",
        ("DROCC", "cifar10", "standard"): "DROCC",
    }
    for key in sorted(indexed, key=lambda item: (item[0], item[1], item[2])):
        lines.append(f"| {labels[key]} | {key[1]} | {metric(indexed[key])} |")
    lines.extend([
        "",
        "## DROCC diagnostic separation",
        "",
        f"Across the three class-macro seeds, source test-selected minus fixed-final AUROC is {pct(statistics.mean(macro_deltas))} ± {pct(statistics.stdev(macro_deltas))} pp. It is excluded from the benchmark table.",
        "",
        "## Artifacts",
        "",
        "- `native_per_class_per_seed.csv`: fixed-final per-class/seed metrics.",
        "- `drocc_test_selected_diagnostic.csv`: source-selection diagnostic only.",
        "- `independent_metric_audit.json`: prediction, history, schedule, and hash checks.",
        "- `deep_completion.json` and `drocc_completion.json`: immutable matrix completion manifests.",
        "",
    ])
    (root / "REPORT.md").write_text("\n".join(lines))
    print(root / "REPORT.md")


if __name__ == "__main__":
    main()
