"""Independently check exported metrics using only the Python standard library."""

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, context):
    require(
        math.isclose(actual, float(expected), rel_tol=0, abs_tol=1e-12),
        f"metric mismatch: {context}: {actual} != {expected}",
    )


def rank_metrics(pairs):
    """AUROC by pair counting; AP by tied-score threshold groups."""
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    require(positives > 0 and negatives > 0, "both classes are required")
    groups = []
    for _, group in itertools.groupby(sorted(pairs), key=lambda pair: pair[0]):
        labels = [label for _, label in group]
        groups.append((sum(labels), len(labels) - sum(labels)))
    lower_negatives = 0
    concordance = 0.0
    for positive, negative in groups:
        concordance += positive * (lower_negatives + negative / 2)
        lower_negatives += negative
    seen_positive = seen_total = 0
    ap = 0.0
    for positive, negative in reversed(groups):
        seen_positive += positive
        seen_total += positive + negative
        ap += (positive / positives) * (seen_positive / seen_total)
    return concordance / (positives * negatives), ap


def audit(directory):
    cfg = json.loads((directory / "protocol_lock.json").read_text())["config"]
    dataset = json.loads((directory / "dataset_manifest.json").read_text())
    summary = read_csv(directory / "summary.csv")
    methods = [row["method"] for row in summary]
    require(len(methods) == len(set(methods)) == 10, "expected ten distinct variants")
    expected_test = defaultdict(dict)
    for image in dataset["images"]:
        if image["split"] == "test":
            category = image["path"].split("/")[0]
            expected_test[category][image["path"]] = image["label"]
    grouped = defaultdict(list)
    predictions = read_csv(directory / "predictions_all.csv")
    for row in predictions:
        grouped[row["category"], int(row["seed"]), row["method"]].append(row)
    expected_groups = set(itertools.product(cfg["categories"], cfg["seeds"], methods))
    require(set(grouped) == expected_groups, "missing or extra prediction group")
    metrics = read_csv(directory / "per_category_per_seed.csv")
    keyed_metrics = {
        (row["category"], int(row["seed"]), row["method"]): row for row in metrics
    }
    require(
        set(keyed_metrics) == expected_groups and len(metrics) == len(expected_groups),
        "missing, extra or duplicate metric rows",
    )
    seed_values = defaultdict(lambda: defaultdict(list))
    for key, rows in grouped.items():
        category, seed, method = key
        labels = {row["path"]: int(row["label"]) for row in rows}
        require(len(labels) == len(rows), f"duplicate prediction: {key}")
        require(
            labels == expected_test[category], f"test coverage or label mismatch: {key}"
        )
        metric = keyed_metrics[key]
        threshold = float(metric["threshold"])
        require(not math.isnan(threshold), f"NaN threshold: {key}")
        pairs = [(float(row["score"]), int(row["label"])) for row in rows]
        require(
            all(math.isfinite(score) for score, _ in pairs), f"nonfinite score: {key}"
        )
        require(
            all(float(row["threshold"]) == threshold for row in rows),
            f"inconsistent prediction thresholds: {key}",
        )
        auc, ap = rank_metrics(pairs)
        negatives = sum(label == 0 for _, label in pairs)
        positives = len(pairs) - negatives
        fp = sum(score > threshold and label == 0 for score, label in pairs)
        tp = sum(score > threshold and label == 1 for score, label in pairs)
        values = {
            "auroc": auc,
            "average_precision": ap,
            "test_fpr": fp / negatives,
            "test_tpr": tp / positives,
        }
        for field, value in values.items():
            close(value, metric[field], f"{key}/{field}")
            seed_values[method, seed][field].append(value)
        for field, value in {
            "false_positives": fp,
            "true_positives": tp,
            "test_normal": negatives,
            "test_anomaly": positives,
        }.items():
            require(value == int(metric[field]), f"count mismatch: {key}/{field}")
    for row in summary:
        method = row["method"]
        for field, mean_field, sd_field in [
            ("auroc", "auroc_mean", "auroc_sd"),
            ("average_precision", "ap_mean", "ap_sd"),
            ("test_fpr", "fpr_mean", None),
            ("test_tpr", "tpr_mean", None),
        ]:
            means = [
                statistics.mean(seed_values[method, seed][field])
                for seed in cfg["seeds"]
            ]
            close(
                statistics.mean(means),
                row[mean_field],
                f"summary/{method}/{mean_field}",
            )
            if sd_field:
                close(
                    statistics.stdev(means),
                    row[sd_field],
                    f"summary/{method}/{sd_field}",
                )
    return {
        "status": "passed",
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "summary_rows": len(summary),
        "implementation": "Python standard library; tied-rank AUROC and threshold-group AP",
        "checked": [
            "all original test paths and labels",
            "no duplicate predictions",
            "AUROC",
            "AP",
            "thresholds",
            "confusion counts",
            "FPR",
            "TPR",
            "equal-category seed macro means",
            "sample SD across seeds",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="results/mvtec-full-v2")
    args = parser.parse_args()
    print(json.dumps(audit(Path(args.directory)), indent=2))
