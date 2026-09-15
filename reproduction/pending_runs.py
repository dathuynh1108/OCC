"""Audit the selected reproduction budget and list missing jobs without launching them."""

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

from .common import ROOT, sha256


def completed_result(path, expected, prediction_name, expected_config):
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    for key, value in {"complete": True, "smoke": False, **expected}.items():
        if result.get(key) != value:
            raise ValueError(f"Result differs from selected native target: {path}: {key}")
    config_path, config_values = expected_config
    config = json.loads(config_path.read_text())
    for key, value in config_values.items():
        if config.get(key) != value:
            raise ValueError(f"Saved config differs from author target: {config_path}: {key}")
    if sha256(path.parent / prediction_name) != result["prediction_sha256"]:
        raise ValueError(f"Prediction hash mismatch: {path}")
    return True


def inventory(root=ROOT, plan_name="run_plan_budgeted.json"):
    # The user selected a light reproduction; longer schedules are not missing work.
    plan_path = root / plan_name
    plan = json.loads(plan_path.read_text())
    if sha256(root / "source_lock.json") != plan["source_lock_sha256"]:
        raise ValueError("Source lock differs from the selected run plan")
    rows = []
    cfg = plan["native"]["deep_svdd"]
    for dataset, normal, seed, objective in itertools.product(
        cfg["datasets"], cfg["classes"], cfg["seeds"], cfg["objectives"]
    ):
        directory = (
            root / "results/native/deep_svdd" / cfg["target"] / dataset
            / f"class_{normal}" / f"seed_{seed}"
        )
        result = directory / objective / "result.json"
        expected = {
            "target": cfg["target"], "dataset": dataset, "normal_class": normal,
            "seed": seed, "objective": objective, "selection": cfg["selection"],
            "ae_epochs": cfg["ae"][dataset]["epochs"], "svdd_epochs": cfg["svdd_epochs"],
            "test_count": cfg["test_count"], "run_plan_sha256": sha256(plan_path),
        }
        complete = completed_result(
            result, expected, "predictions.csv", (directory / "config.json", cfg)
        )
        rows.append({
            "method": "Deep SVDD", "dataset": dataset, "class": normal, "seed": seed,
            "variant": objective, "status": "complete" if complete else "missing",
            "result": str(result.relative_to(root)),
            "command": (
                f"OCC_RUN_PLAN={plan_name} python -m reproduction.deep_svdd"
                f" --dataset {dataset} --normal-class {normal} --seed {seed}"
            ),
        })
    cfg = plan["native"]["drocc"]
    for normal, seed in itertools.product(cfg["classes"], cfg["seeds"]):
        directory = (
            root / "results/native/drocc" / cfg["target"]
            / f"class_{normal}" / f"seed_{seed}"
        )
        expected = {
            "target": cfg["target"], "dataset": cfg["dataset"], "normal_class": normal,
            "seed": seed, "trained_epochs": cfg["epochs"], "test_count": 10000,
        }
        complete = []
        for selection in cfg["selection_exports"]:
            result = directory / f"{selection}_result.json"
            complete.append(completed_result(
                result, {**expected, "selection": selection}, f"{selection}_predictions.csv",
                (directory / "config.json", {**cfg, **cfg["class_params"][normal]}),
            ))
        rows.append({
            "method": "DROCC", "dataset": cfg["dataset"], "class": normal, "seed": seed,
            "variant": "author test-selected + final export",
            "status": "complete" if all(complete) else "missing",
            "result": str(directory.relative_to(root)),
            "command": (
                f"OCC_RUN_PLAN={plan_name} python -m reproduction.drocc"
                f" --normal-class {normal} --seed {seed}"
            ),
        })
    comparison_root = root / "results/mvtec-native-light-4090-20260915"
    for (method, variant, relative), category, seed in itertools.product(
        [("RBF SVDD", "nu=0.01", "shallow/nu_0.01"),
         ("RBF SVDD", "nu=0.1", "shallow/nu_0.1"),
         ("Deep SVDD", "one-class", "deep/one-class"),
         ("Deep SVDD", "soft-boundary", "deep/soft-boundary"),
         ("DROCC", "author CNN", "drocc/author_test_selected")],
        plan["native"]["patchcore"]["categories"], [0],
    ):
        result_path = comparison_root / category / relative / "result.json"
        complete = False
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert result["complete"] and result["dataset"] == "mvtec"
            assert result["category"] == category and result["seed"] == seed
            assert sha256(result_path.parent / "predictions.csv") == result["predictions_sha256"]
            complete = True
        rows.append({
            "method": method, "dataset": "mvtec", "class": category, "seed": seed,
            "variant": variant, "status": "complete" if complete else "missing",
            "result": str(result_path.relative_to(root)),
            "command": (
                f"OCC_RUN_PLAN=run_plan_budgeted.json python -m reproduction.mvtec_image_baselines"
                f" --method {relative.split('/')[0]} --category {category}"
            ),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", choices=("run_plan_budgeted.json", "run_plan.json"),
        default="run_plan_budgeted.json",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Write the missing-job CSV; no training is launched.",
    )
    args = parser.parse_args()
    rows = inventory(plan_name=args.plan)
    counts = Counter(
        (r["method"], r["dataset"], r["variant"], r["status"]) for r in rows
    )
    for key, count in sorted(counts.items()):
        print(" | ".join(map(str, (*key, count))))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(r for r in rows if r["status"] != "complete")


if __name__ == "__main__":
    main()
