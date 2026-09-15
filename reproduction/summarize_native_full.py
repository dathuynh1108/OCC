"""Independently audit the immutable full native Deep SVDD and DROCC matrices.

The source runners write one prediction export for every completed task.  This
report recomputes AUROC/AP from those exports, checks the fixed schedules and
writes only fixed-final DROCC values to the presentation summary.  The
historical test-selected DROCC export remains audited as a diagnostic artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from .common import ROOT, load_plan, plan_path, sha256, write_json

sys.path.insert(0, str(ROOT / "scripts"))
from audit_published_results import rank_metrics


def read_json(path: Path):
    return json.loads(path.read_text())


def read_csv(path: Path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def close(actual, expected, context):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-12, rel_tol=0):
        raise ValueError(f"Metric mismatch for {context}: {actual} != {expected}")


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_predictions(path: Path, result, normal_class):
    rows = read_csv(path)
    if len(rows) != result["test_count"]:
        raise ValueError(f"Unexpected test count in {path}")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != {str(i) for i in range(10000)}:
        raise ValueError(f"Test ID coverage mismatch in {path}")
    if not all(
        int(row["anomaly_label"]) == int(int(row["original_label"]) != normal_class)
        for row in rows
    ):
        raise ValueError(f"One-vs-rest labels mismatch in {path}")
    pairs = [(float(row["score"]), int(row["anomaly_label"])) for row in rows]
    if not all(math.isfinite(score) for score, _ in pairs):
        raise ValueError(f"Nonfinite score in {path}")
    auc, ap = rank_metrics(pairs)
    close(auc, result["auroc"], f"{path}/AUROC")
    close(ap, result["average_precision"], f"{path}/AP")
    expected_hash = result.get("prediction_sha256", result.get("predictions_sha256"))
    if expected_hash and digest(path) != expected_hash:
        raise ValueError(f"Prediction hash mismatch in {path}")
    return len(rows), auc, ap


def validate_history(path: Path, epochs: int, examples: int, batch_size: int, losses):
    rows = read_json(path)
    if len(rows) != epochs or [row["epoch"] for row in rows] != list(range(1, epochs + 1)):
        raise ValueError(f"Epoch history mismatch in {path}")
    expected_steps = math.ceil(examples / batch_size)
    for row in rows:
        if row["examples"] != examples or row["optimizer_steps"] != expected_steps:
            raise ValueError(f"Example/step count mismatch in {path}")
        if not all(math.isfinite(float(row[key])) for key in (*losses, "gradient_norm")):
            raise ValueError(f"Nonfinite history value in {path}")


def summary_rows(rows):
    summaries = []
    for (track, dataset, variant), group in sorted(rows.items()):
        by_seed = defaultdict(list)
        for row in group:
            by_seed[int(row["seed"])].append(row)
        expected_classes = set(range(10))
        if any({int(row["class_or_category"]) for row in samples} != expected_classes for samples in by_seed.values()):
            raise ValueError(f"Incomplete class coverage for {track}/{dataset}/{variant}")
        auc_macros = [statistics.mean(float(row["auroc"]) for row in samples) for _, samples in sorted(by_seed.items())]
        ap_macros = [statistics.mean(float(row["average_precision"]) for row in samples) for _, samples in sorted(by_seed.items())]
        summaries.append({
            "track": track,
            "dataset": dataset,
            "variant": variant,
            "selection": group[0]["selection"],
            "balanced_seeds": ",".join(str(seed) for seed in sorted(by_seed)),
            "repeats": len(by_seed),
            "classes": 10,
            "auroc_mean": statistics.mean(auc_macros),
            "auroc_sd": statistics.stdev(auc_macros) if len(auc_macros) > 1 else None,
            "ap_mean": statistics.mean(ap_macros),
            "ap_sd": statistics.stdev(ap_macros) if len(ap_macros) > 1 else None,
        })
    return summaries


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to create empty report: {path}")
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=ROOT / "results/native-full-4090-20260915",
        help="Immutable full-schedule native result root.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    plan = load_plan()
    if plan_path().resolve() != (ROOT / "run_plan.json").resolve():
        raise ValueError("Native full audit requires run_plan.json")
    for name in ("deep_completion.json", "drocc_completion.json"):
        completion = read_json(root / name)
        if not completion["complete"] or completion["failures"]:
            raise ValueError(f"Native full matrix incomplete: {name}")

    result_rows = []
    checked_prediction_rows = 0
    deep = plan["native"]["deep_svdd"]
    deep_base = root / "deep_svdd" / deep["target"]
    for dataset in deep["datasets"]:
        ae_epochs = deep["ae"][dataset]["epochs"]
        for normal_class in deep["classes"]:
            data_evidence = read_json(
                deep_base / "data_evidence" / f"{dataset}_class{normal_class}_data.json"
            )
            train_count = data_evidence["train"]["count"]
            for seed in deep["seeds"]:
                task = deep_base / dataset / f"class_{normal_class}" / f"seed_{seed}"
                validate_history(task / "ae/history.json", ae_epochs, train_count, deep["batch_size"], ("loss",))
                for objective in deep["objectives"]:
                    result_path = task / objective / "result.json"
                    result = read_json(result_path)
                    if not result["complete"] or result["smoke"]:
                        raise ValueError(f"Incomplete Deep SVDD result: {result_path}")
                    if (result["target"], result["dataset"], result["normal_class"], result["seed"], result["objective"], result["selection"]) != (
                        deep["target"], dataset, normal_class, seed, objective, "fixed_final_epoch"
                    ):
                        raise ValueError(f"Deep SVDD identity mismatch: {result_path}")
                    if result["ae_epochs"] != ae_epochs or result["svdd_epochs"] != deep["svdd_epochs"]:
                        raise ValueError(f"Deep SVDD schedule mismatch: {result_path}")
                    if result["test_count"] != 10000 or result["checkpoint_replay_max_abs_error"] > 1e-6:
                        raise ValueError(f"Deep SVDD export validation mismatch: {result_path}")
                    if result["run_plan_sha256"] != sha256(ROOT / "run_plan.json"):
                        raise ValueError(f"Deep SVDD plan hash mismatch: {result_path}")
                    validate_history(task / objective / "history.json", deep["svdd_epochs"], train_count, deep["batch_size"], ("loss",))
                    count, auc, ap = validate_predictions(task / objective / "predictions.csv", result, normal_class)
                    checked_prediction_rows += count
                    result_rows.append({
                        "track": "DeepSVDD", "dataset": dataset, "class_or_category": normal_class,
                        "seed": seed, "variant": objective, "selection": "fixed_final_epoch",
                        "selected_epoch": deep["svdd_epochs"],
                        "auroc": auc, "average_precision": ap, "test_count": result["test_count"],
                    })

    drocc = plan["native"]["drocc"]
    drocc_base = root / "drocc" / drocc["target"]
    diagnostic_rows = []
    for normal_class in drocc["classes"]:
        for seed in drocc["seeds"]:
            task = drocc_base / f"class_{normal_class}" / f"seed_{seed}"
            config = read_json(task / "config.json")
            expected_config = {
                **{key: drocc[key] for key in (
                    "target", "dataset", "epochs", "batch_size", "gamma", "momentum",
                    "weight_decay", "ascent_num_steps", "projection_interval", "only_ce_epochs",
                )},
                **drocc["class_params"][normal_class],
                "seed": seed,
                "smoke": False,
            }
            if any(config[key] != value for key, value in expected_config.items()):
                raise ValueError(f"DROCC frozen config mismatch: {task}")
            validate_history(task / "history.json", drocc["epochs"], 5000, drocc["batch_size"], ("ce_loss", "adv_loss"))
            for selection, destination in (("fixed_final_epoch", result_rows), ("test_selected", diagnostic_rows)):
                result = read_json(task / f"{selection}_result.json")
                if not result["complete"] or result["smoke"]:
                    raise ValueError(f"Incomplete DROCC result: {task}/{selection}")
                if (result["target"], result["dataset"], result["normal_class"], result["seed"], result["selection"]) != (
                    drocc["target"], "cifar10", normal_class, seed, selection
                ):
                    raise ValueError(f"DROCC identity mismatch: {task}/{selection}")
                if result["trained_epochs"] != drocc["epochs"] or result["test_count"] != 10000:
                    raise ValueError(f"DROCC schedule mismatch: {task}/{selection}")
                if selection == "fixed_final_epoch" and result["selected_epoch"] != drocc["epochs"]:
                    raise ValueError(f"DROCC final checkpoint mismatch: {task}")
                if result["checkpoint_replay_max_abs_error"] > 1e-6:
                    raise ValueError(f"DROCC replay mismatch: {task}/{selection}")
                count, auc, ap = validate_predictions(task / f"{selection}_predictions.csv", result, normal_class)
                checked_prediction_rows += count
                destination.append({
                    "track": "DROCC", "dataset": "cifar10", "class_or_category": normal_class,
                    "seed": seed, "variant": "standard", "selection": selection,
                    "selected_epoch": result["selected_epoch"], "auroc": auc,
                    "average_precision": ap, "test_count": result["test_count"],
                })

    keyed = defaultdict(list)
    for row in result_rows:
        keyed[row["track"], row["dataset"], row["variant"]].append(row)
    summaries = summary_rows(keyed)
    if len(result_rows) != 430 or len(diagnostic_rows) != 30 or len(summaries) != 5:
        raise ValueError("Unexpected full native metric coverage")
    write_csv(root / "native_per_class_per_seed.csv", result_rows)
    write_csv(root / "native_summary.csv", summaries)
    write_csv(root / "drocc_test_selected_diagnostic.csv", diagnostic_rows)
    write_json(root / "independent_metric_audit.json", {
        "passed": True,
        "run_plan": "run_plan.json",
        "run_plan_sha256": sha256(ROOT / "run_plan.json"),
        "source_lock_sha256": sha256(ROOT / "source_lock.json"),
        "prediction_rows_checked": checked_prediction_rows,
        "fixed_final_metric_rows": len(result_rows),
        "test_selected_diagnostic_rows": len(diagnostic_rows),
        "summary_rows": len(summaries),
        "presentation_selection": "fixed_final_epoch",
        "metric_implementation": "stdlib tied-rank AUROC and threshold-group AP",
    })
    print(json.dumps({"passed": True, "fixed_final_rows": len(result_rows), "diagnostic_rows": len(diagnostic_rows)}, indent=2))


if __name__ == "__main__":
    main()
