"""Ruff image-table Gaussian OC-SVM, with its constant-diagonal SVDD equivalence.

Explicit modern sklearn replication of the pinned source protocol. This is not
an executed MATLAB dd_tools run or a reproduced Tax--Duin 2004 numeric table.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_limits
from torchvision.datasets import CIFAR10, MNIST

from .common import ROOT, load_plan, sha256, verify_source, write_json

spec = importlib.util.spec_from_file_location(
    "occ_author_preprocessing",
    ROOT / "vendor/deep_svdd_theano/src/datasets/preprocessing.py",
)
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)


def raw_dataset(name):
    cls = MNIST if name == "mnist" else CIFAR10
    train, test = [
        cls(str(ROOT / "data/native"), train=split, download=True)
        for split in (True, False)
    ]

    def convert(source):
        raw = np.asarray(source.data)
        raw = raw[:, None] if name == "mnist" else raw.transpose(0, 3, 1, 2)
        return raw.astype(np.float32), np.asarray(source.targets)

    x, y = convert(train)
    tx, ty = convert(test)
    return x, y, tx, ty


def split_ids(y, test_y, normal, seed):
    """Trace source loader RNG, including its Python2 batch-multiple truncation."""
    rng = np.random.RandomState(seed)
    normals = np.flatnonzero(y == normal)
    outliers = np.flatnonzero(y != normal)
    order_norm = rng.permutation(len(normals))
    rng.permutation(len(outliers))  # Source shuffles outliers even at out_frac=0.
    training = normals[order_norm]
    training = training[rng.permutation(len(training))]
    rng.permutation(0)  # Source's zero-length validation split.
    batch_multiple = (len(training) // 200) * 200
    training = training[rng.choice(len(training), batch_multiple, replace=False)]
    test = np.r_[np.flatnonzero(test_y == normal), np.flatnonzero(test_y != normal)]
    test = test[rng.permutation(len(test))]
    while True:
        order = rng.permutation(len(test))
        holdout, evaluated = test[order[:1000]], test[order[1000:]]
        if (
            len(np.unique(test_y[holdout] == normal))
            == len(np.unique(test_y[evaluated] == normal))
            == 2
        ):
            break
    return training, holdout, evaluated, rng.get_state()


def prepare(x, tx, train_ids):
    training = x[train_ids].copy()
    test = tx.copy()
    empty = np.empty((0, *training.shape[1:]), dtype=np.float32)
    author.normalize_data(training, empty, test, scale=np.float32(255))
    minimum, maximum = float(training.min()), float(training.max())
    author.rescale_to_unit_interval(training, empty, test)
    training, test = training.reshape(len(training), -1), test.reshape(len(test), -1)
    # Source sklearn0.18 auto used full SVD for a fractional variance target.
    # Explicit solver prevents a current auto-policy change to covariance_eigh.
    pca = PCA(n_components=0.95, svd_solver="full")
    pca.fit(training)
    return pca.transform(training), pca.transform(test), pca, minimum, maximum


def svdd_diagnostics(model, x, gamma, nu):
    n = len(x)
    alpha = np.zeros(n, dtype=np.float64)
    alpha[model.support_] = model.dual_coef_.ravel() / (nu * n)
    rho = float(-model.intercept_[0] / (nu * n))
    k = np.exp(-gamma * cdist(x.astype(float), x.astype(float), metric="sqeuclidean"))
    center_norm = float(alpha @ k @ alpha)
    radius_squared = 1 - 2 * rho + center_norm
    decision = k @ alpha - rho
    bound = 1 / (nu * n)
    tol = 1e-9
    free = (alpha > tol) & (alpha < bound - tol)
    lower = alpha <= tol
    upper = alpha >= bound - tol
    residual = max(
        float(np.abs(2 * decision[free]).max(initial=0)),
        float(np.maximum(-2 * decision[lower], 0).max(initial=0)),
        float(np.maximum(2 * decision[upper], 0).max(initial=0)),
    )
    distance_squared = 1 - 2 * (k @ alpha) + center_norm
    source_excess = -2 * model.decision_function(x) / (nu * n)
    equivalence_error = float(
        np.max(np.abs(distance_squared - radius_squared - source_excess))
    )
    assert (
        abs(alpha.sum() - 1) < 1e-6
        and alpha.min() >= -1e-10
        and alpha.max() <= bound + 1e-8
    )
    assert equivalence_error < 1e-7
    return {
        "alpha_sum": float(alpha.sum()),
        "alpha_min": float(alpha.min()),
        "alpha_max": float(alpha.max()),
        "box_upper": bound,
        "free_support_count": int(free.sum()),
        "rho_normalized": rho,
        "center_norm_squared": center_norm,
        "radius_squared": radius_squared,
        "KKT_stationarity_residual": residual,
        "score_equivalence_max_abs_error": equivalence_error,
        "gamma": gamma,
        "sigma_dd_tools": gamma**-0.5,
        "fit_status": int(model.fit_status_),
        "iterations": int(model.n_iter_),
        "fit_count": n,
        "solver_tolerance": model.tol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    parser.add_argument("--normal-class", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    verify_source("deep_svdd_theano")
    plan = load_plan()["native"]["shallow"]
    threadpool_limits(4)
    out = (
        ROOT
        / "results/native/shallow"
        / plan["target"]
        / args.dataset
        / f"class_{args.normal_class}"
        / f"seed_{args.seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    x, y, tx, ty = raw_dataset(args.dataset)
    training_ids, holdout_ids, test_ids, random_state = split_ids(
        y, ty, args.normal_class, args.seed
    )
    write_json(
        out / "splits.json",
        {
            "train": training_ids.tolist(),
            "labeled_test_tuning": holdout_ids.tolist(),
            "evaluated_test": test_ids.tolist(),
            "normal_class": args.normal_class,
            "source_batch_multiple": 200,
            "original_train_normal_count": int((y == args.normal_class).sum()),
        },
    )
    started = time.perf_counter()
    train, test_all, pca, minimum, maximum = prepare(x, tx, training_ids)
    del x
    labels = (ty != args.normal_class).astype(int)
    for nu in plan["nu"]:
        destination = out / f"nu_{nu}"
        destination.mkdir(exist_ok=True)
        if (destination / "result.json").exists():
            continue
        history, best_auc, best_model, best_gamma = [], 0.0, None, None
        for exponent in plan["gamma_exponents"]:
            gamma = float(2.0**exponent)
            tick = time.perf_counter()
            model = OneClassSVM(
                kernel="rbf",
                gamma=gamma,
                nu=nu,
                tol=0.001,
                cache_size=200,
                shrinking=True,
                max_iter=-1,
            )
            model.fit(train)
            assert model.fit_status_ == 0
            tuning_score = -model.decision_function(test_all[holdout_ids])
            auc = roc_auc_score(labels[holdout_ids], tuning_score)
            history.append(
                {
                    "gamma": gamma,
                    "nu": nu,
                    "labeled_test_holdout_auroc": auc,
                    "fit_seconds": time.perf_counter() - tick,
                    "support_count": len(model.support_),
                    "iterations": int(model.n_iter_),
                    "fit_status": int(model.fit_status_),
                }
            )
            if auc > best_auc:
                best_auc, best_model, best_gamma = auc, model, gamma
            print(
                json.dumps(
                    {
                        "dataset": args.dataset,
                        "normal_class": args.normal_class,
                        "seed": args.seed,
                        **history[-1],
                    }
                ),
                flush=True,
            )
        assert best_model is not None
        pd.DataFrame(history).to_csv(destination / "gamma_search.csv", index=False)
        diagnostic = svdd_diagnostics(best_model, train, best_gamma, nu)
        scores = -best_model.decision_function(test_all[test_ids])
        checkpoint = destination / "model.joblib"
        joblib.dump(
            {
                "model": best_model,
                "pca": pca,
                "scale_min": minimum,
                "scale_max": maximum,
                "training_ids": training_ids,
                "holdout_ids": holdout_ids,
                "test_ids": test_ids,
                "numpy_rng_after_split": random_state,
                "config": plan,
                "diagnostics": diagnostic,
            },
            checkpoint,
            compress=3,
        )
        saved = joblib.load(checkpoint)
        # Preserve the original full-test PCA GEMM shape before subset selection.
        # Float32 BLAS can differ when replaying only the 9000 evaluated rows.
        raw_test = tx.copy()
        raw_test /= np.float32(255)
        raw_test -= saved["scale_min"]
        raw_test /= saved["scale_max"] - saved["scale_min"]
        replay = -saved["model"].decision_function(
            saved["pca"].transform(raw_test.reshape(len(raw_test), -1))[
                saved["test_ids"]
            ]
        )
        error = float(np.max(np.abs(scores - replay)))
        assert error <= 1e-6
        pd.DataFrame(
            {
                "sample_id": test_ids,
                "original_label": ty[test_ids],
                "anomaly_label": labels[test_ids],
                "score": scores,
                "svdd_squared_radius_excess": 2 * scores / (nu * len(train)),
            }
        ).to_csv(destination / "predictions.csv", index=False)
        table = pd.read_csv(
            destination / "predictions.csv", float_precision="round_trip"
        )
        assert np.array_equal(table.score.to_numpy(), scores)
        write_json(
            destination / "result.json",
            {
                "target": plan["target"],
                "csv_float_parser": "round_trip",
                "dataset": args.dataset,
                "normal_class": args.normal_class,
                "seed": args.seed,
                "nu": nu,
                "selected_gamma": best_gamma,
                "gamma_selection": "1000_labeled_test_holdout",
                "nu_selection": "not_selected_this_row",
                "train_count": len(train),
                "test_count": len(test_ids),
                "pca_dimension": train.shape[1],
                "auroc": roc_auc_score(table.anomaly_label, table.score),
                "average_precision": average_precision_score(
                    table.anomaly_label, table.score
                ),
                "checkpoint_replay_max_abs_error": error,
                "checkpoint_sha256": sha256(checkpoint),
                "predictions_sha256": sha256(destination / "predictions.csv"),
                "diagnostics": diagnostic,
                "seconds_since_preprocessing_start": time.perf_counter() - started,
                "complete": True,
            },
        )


if __name__ == "__main__":
    main()
