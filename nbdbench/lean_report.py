"""Verified report for the lean A+D and B+D MVTec protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import sha256_file
from .run import score_method_names, write_json


def comparison_pairs(cfg):
    if cfg.get("score_protocol") != "lean_ad_bd":
        raise ValueError("lean report requires score_protocol=lean_ad_bd")
    return [("Bubble_A+D", "Bubble_A"), ("Bubble_B+D", "Bubble_B")]


def collect_verified(output: Path):
    lock = json.loads((output / "protocol_lock.json").read_text())
    cfg = lock["config"]
    methods = score_method_names(cfg)
    metrics_all, prediction_all, runs = [], [], []
    for category in cfg["categories"]:
        for seed in cfg["seeds"]:
            directory = output / category / f"seed-{seed}"
            complete = json.loads((directory / "COMPLETE.json").read_text())
            for name, digest in complete["artifacts"].items():
                if sha256_file(directory / name) != digest:
                    raise RuntimeError(f"artifact hash mismatch: {directory / name}")
            verification = json.loads((directory / "verification.json").read_text())
            if verification["status"] != "passed" or verification["methods"] != methods:
                raise RuntimeError(f"unverified or mismatched method set: {directory}")
            metrics = pd.read_csv(directory / "metrics.csv")
            predictions = pd.read_csv(directory / "predictions.csv")
            if sorted(metrics.method) != sorted(methods):
                raise RuntimeError(f"missing or duplicated score: {directory}")
            for method in methods:
                row = metrics[metrics.method == method].iloc[0]
                pred = predictions[predictions.method == method]
                if pred.path.duplicated().any() or not np.isclose(
                    roc_auc_score(pred.label, pred.score), row.auroc, atol=1e-12, rtol=0
                ) or not np.isclose(
                    average_precision_score(pred.label, pred.score), row.average_precision,
                    atol=1e-12, rtol=0,
                ):
                    raise RuntimeError(f"saved prediction metric mismatch: {directory}/{method}")
            metrics_all.append(metrics)
            prediction_all.append(predictions)
            runs.append({"category": category, "seed": seed, "seconds": verification["seconds"]})
    return cfg, methods, pd.concat(metrics_all), pd.concat(prediction_all), pd.DataFrame(runs)


def render(output: Path):
    cfg, methods, metrics, predictions, runs = collect_verified(output)
    per_seed = metrics.groupby(["method", "seed"], sort=False)[
        ["auroc", "average_precision", "test_fpr", "test_tpr"]
    ].mean().reset_index()
    summary = per_seed.groupby("method", sort=False).agg(
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        ap_mean=("average_precision", "mean"), ap_sd=("average_precision", "std"),
        fpr_mean=("test_fpr", "mean"), tpr_mean=("test_tpr", "mean"),
    ).reindex(methods).reset_index()
    category = metrics.groupby(["category", "method"], sort=False).agg(
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        ap_mean=("average_precision", "mean"),
    ).reset_index()
    pivot = metrics.pivot(index=["category", "seed"], columns="method", values="auroc")
    deltas = []
    for final, component in comparison_pairs(cfg):
        part = (pivot[final] - pivot[component]).rename("auroc_delta")
        for (category_name, seed), value in part.items():
            deltas.append({
                "category": category_name, "seed": seed, "final_score": final,
                "component_score": component, "auroc_delta": value,
            })
    paired = pd.DataFrame(deltas)
    class_delta = paired.groupby(["category", "final_score", "component_score"], sort=False).agg(
        auroc_delta_mean=("auroc_delta", "mean"), auroc_delta_sd=("auroc_delta", "std"),
    ).reset_index()
    class_delta["status"] = np.select(
        [class_delta.auroc_delta_mean > 1e-12, class_delta.auroc_delta_mean < -1e-12],
        ["improved", "reduced"], default="tied",
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "per_category_per_seed.csv", index=False)
    predictions.to_csv(output / "predictions_all.csv", index=False)
    per_seed.to_csv(output / "macro_by_seed.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    category.to_csv(output / "category_summary.csv", index=False)
    paired.to_csv(output / "paired_deltas.csv", index=False)
    class_delta.to_csv(output / "class_delta_report.csv", index=False)
    runs.to_csv(output / "run_seconds.csv", index=False)
    lines = [
        "# Lean NBD full MVTec report",
        "",
        f"Verified full coverage: {len(runs)} category-seed runs (15 classes × 3 seeds).",
        "",
        "## Final score candidates",
        "",
        "| Score | Macro image AUROC (%) | AP (%) |",
        "|---|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.method} | {100 * row.auroc_mean:.2f} ± {100 * row.auroc_sd:.2f} | {100 * row.ap_mean:.2f} ± {100 * row.ap_sd:.2f} |")
    lines += ["", "## Per-class change after adding D", "", "| Class | Final | Compared with | Delta AUROC (pp) | Status |", "|---|---|---|---:|---|"]
    for row in class_delta.itertuples():
        lines.append(f"| {row.category} | {row.final_score} | {row.component_score} | {100 * row.auroc_delta_mean:+.2f} ± {100 * row.auroc_delta_sd:.2f} | {row.status} |")
    lines += [
        "",
        "D is the diffusion component of the graph, whose edge construction includes frame geometry. No direct F score or F fusion is reported.",
        "All metrics use saved predictions after checkpoint replay; top-1% patch mean is the common image score.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "lean_report_verification.json", {
        "status": "passed", "runs": len(runs), "metric_rows": len(metrics),
        "methods": methods, "comparison_pairs": comparison_pairs(cfg),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
