"""Audit exported predictions with independent rank metrics and build result tables.

No GPU/deep-learning dependencies; run after copying the result evidence locally.
Incomplete repeats remain in coverage.csv and never become a full-matrix claim.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_published_results import rank_metrics

OUT = ROOT / "results/reproduction-2026-09-14"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-11, rel_tol=0), (a, b)


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def validate_prediction(path, result, expected_ids=None):
    rows = csv_rows(path)
    assert len(rows) == result["test_count"]
    ids = [r["sample_id"] for r in rows]
    assert len(ids) == len(set(ids))
    if expected_ids is not None:
        assert set(ids) == set(map(str, expected_ids))
    if "normal_class" in result:
        assert all(
            int(r["anomaly_label"])
            == int(int(r["original_label"]) != result["normal_class"])
            for r in rows
        )
    pairs = [(float(r["score"]), int(r["anomaly_label"])) for r in rows]
    assert all(math.isfinite(x) for x, _ in pairs)
    auc, ap = rank_metrics(pairs)
    close(auc, result["auroc"])
    close(ap, result["average_precision"])
    expected_hash = result.get("prediction_sha256", result.get("predictions_sha256"))
    if expected_hash:
        assert digest(path) == expected_hash
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-common-complete", action="store_true")
    args = parser.parse_args()
    plan = read(ROOT / "run_plan_budgeted.json")
    OUT.mkdir(parents=True, exist_ok=True)
    coverage = []
    native = []
    audited_rows = 0
    # Native PatchCore uses full normal train; it is a different protocol from the shared FIT table.
    pc = plan["native"]["patchcore"]
    pcroot = ROOT / "results/native/patchcore" / pc["target"]
    expected_test = defaultdict(dict)
    for row in read(ROOT / "results/mvtec-full-v2/dataset_manifest.json")["images"]:
        if row["split"] == "test":
            expected_test[row["path"].split("/")[0]][row["path"]] = row["label"]
    for category, seed in itertools.product(pc["categories"], pc["seeds"]):
        p = pcroot / category / f"seed-{seed}" / "result.json"
        coverage.append(
            dict(
                track="native_patchcore",
                dataset="mvtec",
                class_or_category=category,
                seed=seed,
                variant="author_code",
                complete=p.exists(),
            )
        )
        if not p.exists():
            continue
        r = read(p)
        assert r["complete"] and not r["smoke"]
        assert all(x <= 1e-6 for x in r["replay_errors"].values())
        audited_rows += validate_prediction(
            p.with_name("predictions.csv"), r, expected_test[category]
        )
        assert {
            row["sample_id"]: int(row["anomaly_label"])
            for row in csv_rows(p.with_name("predictions.csv"))
        } == expected_test[category]
        native.append(
            dict(
                track="PatchCore",
                dataset="mvtec",
                class_or_category=category,
                seed=seed,
                variant="author_code",
                auroc=r["auroc"],
                average_precision=r["average_precision"],
                pixel_auroc=r["pixel_auroc"],
                test_count=r["test_count"],
                memory_bytes=r["memory_bytes"],
            )
        )
    # Full-schedule interrupted runs are retained separately and excluded here.
    for track, key in [
        ("DeepSVDD", "deep_svdd"),
        ("DROCC", "drocc"),
        ("Gaussian_OCSVM_equiv_SVDD", "shallow"),
    ]:
        cfg = plan["native"][key]
        base = ROOT / "results/native" / key / cfg["target"]
        for dataset in cfg.get("datasets", [cfg.get("dataset", "cifar10")]):
            for normal, seed in itertools.product(cfg["classes"], cfg["seeds"]):
                if key == "drocc":
                    directory = base / f"class_{normal}" / f"seed_{seed}"
                    variants = cfg["selection_exports"]
                else:
                    directory = base / dataset / f"class_{normal}" / f"seed_{seed}"
                    variants = (
                        cfg["objectives"]
                        if key == "deep_svdd"
                        else [f"nu_{nu}" for nu in cfg["nu"]]
                    )
                for variant in variants:
                    p = (
                        directory / f"{variant}_result.json"
                        if key == "drocc"
                        else directory / variant / "result.json"
                    )
                    coverage.append(
                        dict(
                            track=track,
                            dataset=dataset,
                            class_or_category=normal,
                            seed=seed,
                            variant=variant,
                            complete=p.exists(),
                        )
                    )
                    if not p.exists():
                        continue
                    r = read(p)
                    assert r["complete"] and not r.get("smoke", False)
                    assert r.get("checkpoint_replay_max_abs_error", 0) <= 1e-6
                    if key == "deep_svdd":
                        assert r["ae_epochs"] == 5 and r["svdd_epochs"] == 12
                    if key == "drocc":
                        assert r["trained_epochs"] == 5
                    ids = (
                        range(10000)
                        if key != "shallow"
                        else read(directory / "splits.json")["evaluated_test"]
                    )
                    prediction = (
                        directory / f"{variant}_predictions.csv"
                        if key == "drocc"
                        else p.with_name("predictions.csv")
                    )
                    audited_rows += validate_prediction(prediction, r, ids)
                    native.append(
                        dict(
                            track=track,
                            dataset=dataset,
                            class_or_category=str(normal),
                            seed=seed,
                            variant=variant,
                            auroc=r["auroc"],
                            average_precision=r["average_precision"],
                            test_count=r["test_count"],
                            selected_epoch=r.get("selected_epoch"),
                            nu=r.get("nu"),
                            selected_gamma=r.get("selected_gamma"),
                        )
                    )
    controlled = ROOT / plan["controlled"]["output"]
    common = []
    predictions = []
    resources = []
    methods = pd.read_csv(
        ROOT / "results/mvtec-full-v2/summary.csv"
    ).method.tolist() + ["PatchCore_same_fit_top1pct"]
    for category, seed in itertools.product(
        plan["controlled"]["categories"], plan["controlled"]["seeds"]
    ):
        directory = controlled / category / f"seed-{seed}"
        for kind in ["base", "patchcore_same_fit"]:
            d = directory / kind
            gate = d / ("COMPLETE.json" if kind == "base" else "result.json")
            coverage.append(
                dict(
                    track="common_mvtec",
                    dataset="mvtec",
                    class_or_category=category,
                    seed=seed,
                    variant=kind,
                    complete=gate.exists(),
                )
            )
            if not gate.exists():
                continue
            if kind == "base":
                complete = read(gate)
                assert (
                    digest(d / "predictions.csv")
                    == complete["artifacts"]["predictions.csv"]
                )
                assert read(d / "verification.json")["status"] == "passed"
                diag = read(d / "geometry_diagnostics.json")
                resources.append(
                    {
                        "category": category,
                        "seed": seed,
                        **{k: v for k, v in diag.items() if not isinstance(v, dict)},
                    }
                )
            else:
                assert read(gate)["replay_max_abs_error"] <= 1e-6
            metric = csv_rows(d / "metrics.csv")
            pred = csv_rows(d / "predictions.csv")
            grouped = defaultdict(list)
            for row in pred:
                grouped[row["method"]].append(row)
            for row in metric:
                rows = grouped[row["method"]]
                assert {r["path"]: int(r["label"]) for r in rows} == expected_test[
                    category
                ]
                assert len(rows) == len(expected_test[category])
                pairs = [(float(r["score"]), int(r["label"])) for r in rows]
                assert all(math.isfinite(s) for s, _ in pairs)
                auc, ap = rank_metrics(pairs)
                close(auc, row["auroc"])
                close(ap, row["average_precision"])
                threshold = float(row["threshold"])
                assert not math.isnan(threshold)
                assert all(float(r["threshold"]) == threshold for r in rows)
                neg = sum(y == 0 for _, y in pairs)
                pos = len(pairs) - neg
                fp = sum(s > threshold and y == 0 for s, y in pairs)
                tp = sum(s > threshold and y == 1 for s, y in pairs)
                close(fp / neg, row["test_fpr"])
                close(tp / pos, row["test_tpr"])
                audited_rows += len(rows)
                common.append(
                    {
                        **row,
                        "auroc": auc,
                        "average_precision": ap,
                        "test_fpr": fp / neg,
                        "test_tpr": tp / pos,
                    }
                )
            predictions.extend(pred)
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(OUT / "coverage.csv", index=False)
    complete_common = len(common) == 15 * 3 * 11
    if args.require_common_complete:
        assert complete_common, f"Common rows {len(common)}/495"
    if common:
        table = pd.DataFrame(common)
        table["seed"] = table.seed.astype(int)
        table.to_csv(OUT / "common_per_category_per_seed.csv", index=False)
        pd.DataFrame(predictions).to_csv(OUT / "common_predictions.csv", index=False)
        if complete_common:
            assert set(table.method) == set(methods)
            summary = []
            for method in methods:
                part = table[table.method == method]
                assert len(part) == 45
                seed_mean = part.groupby("seed")[
                    ["auroc", "average_precision", "test_fpr", "test_tpr"]
                ].mean()
                summary.append(
                    dict(
                        method=method,
                        auroc_mean=seed_mean.auroc.mean(),
                        auroc_sd=seed_mean.auroc.std(ddof=1),
                        ap_mean=seed_mean.average_precision.mean(),
                        ap_sd=seed_mean.average_precision.std(ddof=1),
                        fpr_mean=seed_mean.test_fpr.mean(),
                        tpr_mean=seed_mean.test_tpr.mean(),
                    )
                )
            pd.DataFrame(summary).to_csv(OUT / "common_summary.csv", index=False)
            table.groupby(["category", "method"]).agg(
                auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std")
            ).reset_index().to_csv(OUT / "common_category_summary.csv", index=False)
            pivot = table.pivot(
                index=["category", "seed"], columns="method", values="auroc"
            )
            deltas = pd.DataFrame(
                {f"NBD_minus_{m}": pivot.NBD - pivot[m] for m in methods if m != "NBD"}
            )
            deltas.to_csv(OUT / "common_paired_deltas.csv")
    pd.DataFrame(resources).to_csv(OUT / "common_resources.csv", index=False)
    summaries = []
    if native:
        nt = pd.DataFrame(native)
        nt.to_csv(OUT / "native_per_class_per_seed.csv", index=False)
        for key, group in nt.groupby(["track", "dataset", "variant"]):
            expected_classes = 15 if key[0] == "PatchCore" else 10
            balanced = []
            balanced_seeds = []
            for seed, part in group.groupby("seed"):
                if (
                    len(part) == expected_classes
                    and part.class_or_category.nunique() == expected_classes
                ):
                    balanced.append(part)
                    balanced_seeds.append(int(seed))
            if balanced:
                means = (
                    pd.concat(balanced)
                    .groupby("seed")[["auroc", "average_precision", "pixel_auroc"]]
                    .mean()
                )
                summaries.append(
                    dict(
                        track=key[0],
                        dataset=key[1],
                        variant=key[2],
                        balanced_seeds=",".join(map(str, balanced_seeds)),
                        repeats=len(means),
                        classes=expected_classes,
                        auroc_mean=means.auroc.mean(),
                        auroc_sd=means.auroc.std(ddof=1) if len(means) > 1 else None,
                        ap_mean=means.average_precision.mean(),
                        ap_sd=means.average_precision.std(ddof=1)
                        if len(means) > 1
                        else None,
                        pixel_auroc_mean=means.pixel_auroc.mean(),
                    )
                )
    pd.DataFrame(summaries).to_csv(OUT / "native_summary.csv", index=False)
    counts = (
        coverage_frame.groupby("track").complete.agg(["sum", "count"]).reset_index()
    )
    counts.to_csv(OUT / "coverage_summary.csv", index=False)
    dump(
        OUT / "independent_metric_audit.json",
        {
            "passed": True,
            "audited_prediction_rows": audited_rows,
            "common_complete": complete_common,
            "common_metric_rows": len(common),
            "native_metric_rows": len(native),
            "metric_implementation": "stdlib tied-rank AUROC and threshold-group AP, strict > threshold FPR/TPR",
            "plan_sha256": digest(ROOT / "run_plan_budgeted.json"),
            "partial_rows_excluded_from_balanced_native_summary": True,
        },
    )
    print(counts.to_string(index=False))
    print("Audited prediction rows:", audited_rows)


if __name__ == "__main__":
    main()
