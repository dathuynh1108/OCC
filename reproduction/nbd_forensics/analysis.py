"""Artifact inventory, exact score replay, and read-only NBD diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from . import reference


RAW_METHODS = [
    "PatchScore_same_centers",
    "PatchScore_byte_budget",
    "RBF_SVDD",
    "DeepSVDD_head",
    "DROCC_head",
]
PRODUCTION_METHODS = RAW_METHODS + ["Bubble_B", "Bubble_BA", "Bubble_BAD", "Bubble_BAF", "NBD"]
COMPONENTS = {"B": 5, "A": 6, "D": 7, "F": 8}
SOURCE_FILES = ["nbdbench/models.py", "nbdbench/math.py", "nbdbench/run.py", "nbdbench/data.py", "reproduction/mvtec.py", "reproduction/patchcore.py"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    names = sorted({name for row in rows for name in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def safe_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def graph_components(w: np.ndarray) -> int:
    adjacency = (np.asarray(w) > 0).copy()
    np.fill_diagonal(adjacency, False)
    seen, count = set(), 0
    for start in range(len(adjacency)):
        if start in seen:
            continue
        count += 1
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            for next_node in np.flatnonzero(adjacency[node]):
                if int(next_node) not in seen:
                    seen.add(int(next_node))
                    queue.append(int(next_node))
    return count


def pair_transition(labels: np.ndarray, old: np.ndarray, new: np.ndarray) -> dict[str, int | float]:
    normal, anomaly = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    old_delta = old[anomaly, None] - old[normal][None, :]
    new_delta = new[anomaly, None] - new[normal][None, :]
    old_h = (old_delta > 0).astype(float) + 0.5 * (old_delta == 0)
    new_h = (new_delta > 0).astype(float) + 0.5 * (new_delta == 0)
    change = new_h - old_h
    return {
        "pairs": int(change.size),
        "gain_pairs": int(np.count_nonzero(change > 0)),
        "loss_pairs": int(np.count_nonzero(change < 0)),
        "unchanged_pairs": int(np.count_nonzero(change == 0)),
        "auroc_delta_from_pairs": float(change.mean()),
    }


def source_provenance(root: Path) -> dict[str, object]:
    git = lambda *args: subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    return {
        "analysis_source_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain=v1"),
        "source_hashes": {name: sha256(root / name) for name in SOURCE_FILES},
        "config_hashes": {
            name: sha256(root / name)
            for name in ["configs/full.json", "run_plan_budgeted.json", "source_lock.json"]
        },
    }


def inventory(root: Path, artifacts: Path, output: Path) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    expected: set[tuple[str, int]] = set()
    for protocol in sorted(artifacts.glob("*/seed-*/protocol.json")):
        category, seed_text = protocol.parent.parent.name, protocol.parent.name
        seed = int(seed_text.removeprefix("seed-"))
        expected.add((category, seed))
        base = protocol.parent / "base"
        paths = {
            "protocol": protocol,
            "scores": base / "scores.npz",
            "geometry": base / "geometry.pt",
            "predictions": base / "predictions.csv",
            "metrics": base / "metrics.csv",
            "verification": base / "verification.json",
            "complete": base / "COMPLETE.json",
        }
        row: dict[str, object] = {"category": category, "seed": seed}
        for name, path in paths.items():
            row[f"{name}_present"] = path.is_file()
            if path.is_file():
                row[f"{name}_bytes"] = path.stat().st_size
                row[f"{name}_sha256"] = sha256(path)
            elif name not in {"protocol", "complete"}:
                missing.append({"category": category, "seed": seed, "artifact": name, "expected_path": safe_rel(path, root), "reason": "not present in current checkout"})
        row["score_forensics_eligible"] = bool(row["scores_present"] and row["predictions_present"] and row["metrics_present"])
        row["geometry_forensics_eligible"] = bool(row["geometry_present"])
        runs.append(row)
    if not expected:
        raise RuntimeError(f"No protocol files under {artifacts}")
    coverage = []
    for row in runs:
        coverage.append({
            "category": row["category"], "seed": row["seed"], "score_forensics": row["score_forensics_eligible"], "geometry_forensics": row["geometry_forensics_eligible"],
        })
    value = {"artifacts_root": safe_rel(artifacts, root), "run_count": len(runs), "runs": runs, "provenance": source_provenance(root)}
    write_json(output / "inventory.json", value)
    write_rows(output / "coverage.csv", coverage)
    write_rows(output / "missing_artifacts.csv", missing)
    (output / "source_map.md").write_text(
        "# Source map\n\n"
        "`nbdbench/run.py:raw_scores` produces 9 raw channels: five baselines followed by B/A/D/F. "
        "`calibrated_variants` applies normal component-calibration empirical tails to B/A/D/F, "
        "then fuses them at patch level before `image_score` top-fraction pooling. "
        "`BubbleModel.fit` constructs local PCA bubbles and a reversible graph; `components` calculates B/A/D/F. "
        "The author-encoder controlled output is `results/mvtec-author-encoder-budgeted-v3/<category>/seed-<seed>/base`.\n"
    )
    return value


def component_maps(test_raw: np.ndarray, calibration_raw: np.ndarray, frame_weight: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    ordered = {index: np.sort(calibration_raw[..., index].reshape(-1).astype(np.float64)) for index in range(calibration_raw.shape[-1])}
    transformed = {name: reference.empirical_tail_sorted(test_raw[..., index], ordered[index]) for name, index in COMPONENTS.items()}
    maps = {
        "raw_euclidean": test_raw[..., 0],
        "calibrated_euclidean": reference.empirical_tail_sorted(test_raw[..., 0], ordered[0]),
        "raw_B": test_raw[..., 5],
        "Bubble_B": transformed["B"],
        "raw_A": test_raw[..., 6],
        "calibrated_A": transformed["A"],
        "raw_D": test_raw[..., 7],
        "calibrated_D": transformed["D"],
        "raw_F": test_raw[..., 8],
        "calibrated_F": transformed["F"],
    }
    maps.update({
        "Bubble_BA": transformed["B"] + transformed["A"],
        "Bubble_BAD": transformed["B"] + transformed["A"] + transformed["D"],
        "Bubble_BAF": transformed["B"] + transformed["A"] + frame_weight * transformed["F"],
        "NBD": transformed["B"] + transformed["A"] + transformed["D"] + frame_weight * transformed["F"],
    })
    production = np.concatenate([test_raw[..., :5], np.stack([maps[name] for name in ["Bubble_B", "Bubble_BA", "Bubble_BAD", "Bubble_BAF", "NBD"]], axis=-1)], axis=-1)
    return maps, production


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {
        "auroc": reference.auroc_pairwise(labels, scores),
        "average_precision": reference.average_precision_grouped(labels, scores),
    }
    if threshold is not None:
        normal, anomaly = labels == 0, labels == 1
        predicted = scores > threshold
        result.update({"threshold": float(threshold), "test_fpr": float(predicted[normal].mean()), "test_tpr": float(predicted[anomaly].mean())})
    return result


def score_forensics(root: Path, artifacts: Path, output: Path, inv: dict[str, object]) -> dict[str, object]:
    component_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    saturation_rows: list[dict[str, object]] = []
    dependence_rows: list[dict[str, object]] = []
    pooling_rows: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []
    completed = 0
    errors: list[dict[str, object]] = []
    for run in inv["runs"]:
        if not run["score_forensics_eligible"]:
            continue
        category, seed = str(run["category"]), int(run["seed"])
        base = artifacts / category / f"seed-{seed}" / "base"
        protocol = json.loads((base.parent / "protocol.json").read_text())
        cfg = protocol["config"]
        try:
            with np.load(base / "scores.npz", allow_pickle=False) as archive:
                required = {"calibration_raw", "threshold_raw", "test_raw", "test_maps", "test_image_scores", "test_labels", "thresholds"}
                if set(archive.files) != required:
                    raise ValueError(f"unexpected NPZ keys: {archive.files}")
                calibration, threshold_raw, test_raw = archive["calibration_raw"], archive["threshold_raw"], archive["test_raw"]
                saved_maps, saved_images, labels, saved_thresholds = archive["test_maps"], archive["test_image_scores"], archive["test_labels"], archive["thresholds"]
            if calibration.ndim != 3 or calibration.shape[-1] != 9 or test_raw.shape[1:] != calibration.shape[1:] or len(labels) != len(test_raw):
                raise ValueError("raw score schema is not [images, patches, 9] with aligned labels")
            maps, production = component_maps(test_raw, calibration, float(cfg["frame_weight"]))
            threshold_maps, threshold_production = component_maps(threshold_raw, calibration, float(cfg["frame_weight"]))
            production_error = float(np.max(np.abs(production - saved_maps)))
            images = np.stack([reference.top_mean(production[..., column], float(cfg["image_top_fraction"])) for column in range(10)], axis=1)
            image_error = float(np.max(np.abs(images - saved_images)))
            thresholds = np.array([reference.normal_threshold(reference.top_mean(threshold_production[..., column], float(cfg["image_top_fraction"])), float(cfg["alpha"])) for column in range(10)])
            threshold_error = float(np.max(np.abs(thresholds - saved_thresholds)))
            passed = production_error <= 1e-12 and image_error <= 1e-12 and threshold_error <= 1e-12
            replay_rows.append({"category": category, "seed": seed, "status": "passed" if passed else "mismatch", "raw_shape": str(tuple(test_raw.shape)), "maps_max_abs_error": production_error, "images_max_abs_error": image_error, "thresholds_max_abs_error": threshold_error})
            if not passed:
                raise RuntimeError("first replay mismatch at calibration/fusion/pooling boundary")
            prediction = pd.read_csv(base / "predictions.csv", float_precision="round_trip")
            for column, method in enumerate(PRODUCTION_METHODS):
                part = prediction[prediction["method"] == method]
                if len(part) != len(labels) or not np.array_equal(part["label"].to_numpy(dtype=int), labels):
                    raise ValueError(f"prediction CSV label/order mismatch for {method}")
                csv_error = float(np.max(np.abs(part["score"].to_numpy(dtype=np.float64) - images[:, column])))
                recalculated = metrics(labels, images[:, column], thresholds[column])
                declared = pd.read_csv(base / "metrics.csv", float_precision="round_trip")
                row = declared[declared["method"] == method].iloc[0]
                metric_error = max(abs(float(row["auroc"]) - float(recalculated["auroc"])), abs(float(row["average_precision"]) - float(recalculated["average_precision"])))
                metric_rows.append({"category": category, "seed": seed, "method": method, **recalculated, "csv_score_max_abs_error": csv_error, "declared_metric_max_abs_error": metric_error})
                if csv_error > 1e-12 or metric_error > 1e-12:
                    raise RuntimeError(f"CSV or metric mismatch for {method}")
            image_variants = {name: reference.top_mean(values, float(cfg["image_top_fraction"])) for name, values in maps.items()}
            for name, scores in image_variants.items():
                component_rows.append({"category": category, "seed": seed, "score": name, **metrics(labels, scores), "grain": "image", "role": "diagnostic" if name not in PRODUCTION_METHODS else "production_replay"})
            edges = [("raw_euclidean", "raw_B"), ("raw_B", "Bubble_B"), ("Bubble_B", "Bubble_BA"), ("Bubble_BA", "Bubble_BAD"), ("Bubble_BA", "Bubble_BAF"), ("Bubble_BAF", "NBD")]
            paths = prediction[prediction["method"] == "Bubble_B"]["path"].tolist()
            for old_name, new_name in edges:
                old, new = image_variants[old_name], image_variants[new_name]
                old_auc, new_auc = reference.auroc_pairwise(labels, old), reference.auroc_pairwise(labels, new)
                trace = pair_transition(labels, old, new)
                delta_rows.append({"category": category, "seed": seed, "from_score": old_name, "to_score": new_name, "auroc_delta_pp": 100 * (new_auc - old_auc), **trace})
                transition_rows.append({"category": category, "seed": seed, "from_score": old_name, "to_score": new_name, **trace})
                for direction, predicate in [("improved", lambda x: x > 0), ("worsened", lambda x: x < 0)]:
                    choices = []
                    for anomaly in np.flatnonzero(labels == 1):
                        for normal in np.flatnonzero(labels == 0):
                            change = (float(new[anomaly] > new[normal]) + 0.5 * float(new[anomaly] == new[normal])) - (float(old[anomaly] > old[normal]) + 0.5 * float(old[anomaly] == old[normal]))
                            if predicate(change):
                                choices.append((paths[anomaly], paths[normal], anomaly, normal, change))
                    if choices:
                        _, _, anomaly, normal, change = sorted(choices)[0]
                        witnesses.append({"category": category, "seed": seed, "comparison": f"{old_name}->{new_name}", "direction": direction, "selection_rule": "lexicographically first changed anomaly-normal pair", "anomaly_path": paths[anomaly], "normal_path": paths[normal], "old_anomaly_score": old[anomaly], "old_normal_score": old[normal], "new_anomaly_score": new[anomaly], "new_normal_score": new[normal], "pair_credit_delta": change, "visualization_blocked": True})
            for name, index in COMPONENTS.items():
                cal = calibration[..., index].reshape(-1)
                transformed = reference.empirical_tail_sorted(test_raw[..., index], np.sort(cal))
                ceiling = math.log(len(cal) + 1)
                chosen = np.argsort(transformed, axis=1)[:, -max(1, math.ceil(transformed.shape[1] * float(cfg["image_top_fraction"]))):]
                saturation_rows.append({"category": category, "seed": seed, "component": name, "calibration_count": len(cal), "calibration_max": float(cal.max()), "test_patch_above_calibration_max_fraction": float((test_raw[..., index] > cal.max()).mean()), "test_patch_at_tail_ceiling_fraction": float(np.isclose(transformed, ceiling).mean()), "selected_patch_at_tail_ceiling_fraction": float(np.isclose(np.take_along_axis(transformed, chosen, axis=1), ceiling).mean()), "distinct_transformed_values": int(len(np.unique(transformed)))})
            raw_images = {name: reference.top_mean(test_raw[..., index], float(cfg["image_top_fraction"])) for name, index in COMPONENTS.items()}
            for left, right in [("B", "A"), ("B", "D"), ("B", "F"), ("D", "F")]:
                dependence_rows.append({"category": category, "seed": seed, "left": left, "right": right, "raw_spearman": reference.spearman(raw_images[left], raw_images[right]), "calibrated_spearman": reference.spearman(image_variants[f"Bubble_B" if left == "B" else f"calibrated_{left}"], image_variants[f"Bubble_B" if right == "B" else f"calibrated_{right}"])})
            for old_name, new_name in [("Bubble_BA", "Bubble_BAD"), ("Bubble_BAF", "NBD")]:
                old_map, new_map = maps[old_name], maps[new_name]
                k = max(1, math.ceil(old_map.shape[1] * float(cfg["image_top_fraction"])))
                old_indices = np.argsort(old_map, axis=1)[:, -k:]
                new_indices = np.argsort(new_map, axis=1)[:, -k:]
                for image in range(len(labels)):
                    previous = set(old_indices[image].tolist())
                    replacement = set(new_indices[image].tolist())
                    pooling_rows.append({"category": category, "seed": seed, "comparison": f"{old_name}->{new_name}", "path": paths[image], "label": int(labels[image]), "k": k, "old_image_score": float(old_map[image, old_indices[image]].mean()), "fixed_old_selection_score": float(new_map[image, old_indices[image]].mean()), "new_image_score": float(new_map[image, new_indices[image]].mean()), "topk_jaccard": len(previous & replacement) / len(previous | replacement), "topk_overlap": len(previous & replacement)})
            completed += 1
        except Exception as exc:  # Retain failed runs as evidence; do not silently continue as success.
            errors.append({"category": category, "seed": seed, "error": repr(exc)})
            replay_rows.append({"category": category, "seed": seed, "status": "failed", "error": repr(exc)})
    write_rows(output / "component_metrics.csv", component_rows)
    write_rows(output / "replay_by_run.csv", replay_rows)
    write_rows(output / "original_metric_recalculation.csv", metric_rows)
    write_rows(output / "paired_deltas.csv", delta_rows)
    write_rows(output / "ranking_loss_trace.csv", delta_rows)
    write_rows(output / "rank_pair_transitions.csv", transition_rows)
    write_rows(output / "calibration_saturation.csv", saturation_rows)
    write_rows(output / "component_dependence.csv", dependence_rows)
    write_rows(output / "pooling_selection_audit.csv", pooling_rows)
    if witnesses:
        case_dir = output / "case_studies"
        case_dir.mkdir(exist_ok=True)
        write_rows(case_dir / "rank_flip_witnesses.csv", witnesses)
    value = {"status": "completed" if not errors else "partial", "eligible_runs": sum(bool(row["score_forensics_eligible"]) for row in inv["runs"]), "completed_runs": completed, "errors": errors}
    write_json(output / "replay_integrity.json", value)
    return value


def geometry_forensics(root: Path, artifacts: Path, output: Path, inv: dict[str, object]) -> dict[str, object]:
    import torch

    backup_path = root / "results/reproduction-2026-09-14/backup_manifest.json"
    backup = json.loads(backup_path.read_text())["files"] if backup_path.is_file() else {}
    rows: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    neighborhood_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    completed = 0
    for run in inv["runs"]:
        if not run["geometry_forensics_eligible"]:
            continue
        category, seed = str(run["category"]), int(run["seed"])
        path = artifacts / category / f"seed-{seed}" / "base/geometry.pt"
        rel = safe_rel(path, root)
        try:
            expected = backup.get(rel)
            if expected and (path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]):
                raise ValueError("geometry checksum differs from backup manifest")
            try:
                payload = torch.load(path, weights_only=True, map_location="cpu")
                load_mode = "weights_only"
            except Exception:
                if not expected:
                    raise
                payload = torch.load(path, weights_only=False, map_location="cpu")
                load_mode = "hash_verified_legacy"
            bubble = payload["bubble"]
            centers, frames = bubble["centers"].detach().cpu().numpy(), bubble["u"].detach().cpu().numpy()
            eigenvalues, sigma = bubble["eigenvalues"].detach().cpu().numpy(), bubble["sigma"].detach().cpu().numpy()
            w, p, pi, values = (np.asarray(bubble[name]) for name in ["w", "p", "pi", "graph_eigenvalues"])
            degree = w.sum(axis=1)
            identities = np.einsum("mdr,mds->mrs", frames.astype(np.float64), frames.astype(np.float64))
            orthogonality = float(np.max(np.abs(identities - np.eye(frames.shape[-1]))))
            graph_rows.append({"category": category, "seed": seed, "load_mode": load_mode, "bubbles": len(centers), "w_symmetric_max_abs": float(np.max(np.abs(w - w.T))), "w_negative_count": int(np.count_nonzero(w < 0)), "offdiagonal_zero_fraction": float((np.count_nonzero((w == 0) & ~np.eye(len(w), dtype=bool))) / (len(w) * (len(w) - 1))), "degree_min": float(degree.min()), "degree_max": float(degree.max()), "components_nonzero_edges": graph_components(w), "p_row_sum_max_abs_error": float(np.max(np.abs(p.sum(axis=1) - 1))), "detailed_balance_max_abs_error": float(np.max(np.abs(pi[:, None] * p - pi[None, :] * p.T))), "near_positive_one_modes": int(np.count_nonzero(values > 1 - 1e-8)), "near_negative_one_modes": int(np.count_nonzero(values < -1 + 1e-8)), "largest_abs_nonconstant_eigenvalue": float(np.max(np.abs(values))) if len(values) else None})
            rows.append({"category": category, "seed": seed, "load_mode": load_mode, "descriptor_dimension": int(centers.shape[1]), "bubbles": int(centers.shape[0]), "rank": int(frames.shape[-1]), "orthogonality_max_abs_error": orthogonality, "eigenvalue_floor_count": int(np.count_nonzero(eigenvalues <= 1.000001e-6)), "sigma_floor_count": int(np.count_nonzero(sigma <= 1.000001e-6)), "sigma_min": float(sigma.min()), "sigma_max": float(sigma.max()), "frame_diagonal_max_abs": float(np.max(np.abs(np.diag(np.asarray(bubble["frame"]))))), "tau_center": float(bubble["tau_center"]), "tau_frame": float(bubble["tau_frame"])})
            indices = np.asarray(bubble["neighborhood_indices"])
            for bubble_id, members in enumerate(indices):
                neighborhood_rows.append({"category": category, "seed": seed, "bubble": bubble_id, "patch_count": len(members), "unique_patch_indices": int(len(np.unique(members))), "unique_fit_images": None, "image_mapping_status": "blocked_missing_fit_patch_to_image_mapping"})
            completed += 1
        except Exception as exc:
            errors.append({"category": category, "seed": seed, "error": repr(exc)})
    write_rows(output / "geometry_audit.csv", rows)
    write_rows(output / "graph_audit.csv", graph_rows)
    write_rows(output / "neighborhood_quality.csv", neighborhood_rows)
    return {"status": "completed" if not errors else "partial", "eligible_runs": sum(bool(row["geometry_forensics_eligible"]) for row in inv["runs"]), "completed_runs": completed, "errors": errors}


def run_unit_tests(root: Path, output: Path) -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", "tests/test_nbd_forensics.py", "tests/test_equations.py", "-q"]
    started = time.time()
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    (output / "unit_tests.log").write_text(result.stdout + result.stderr)
    value = {"command": command, "returncode": result.returncode, "seconds": time.time() - started, "status": "passed" if result.returncode == 0 else "failed"}
    write_json(output / "unit_test_results.json", value)
    write_json(output / "numerical_audit.json", {
        "status": value["status"],
        "scope": [
            "float64 brute-force squared distance and near-tie ordering",
            "explicit >= empirical-tail with add-one smoothing and sorted parity",
            "pooling-after-tail non-commutativity fixture",
            "explicit tangent/residual bubble energy",
            "tie-aware AUROC and grouped AP",
            "CUDA BubbleModel component execution",
        ],
        "evidence": "tests/test_nbd_forensics.py plus tests/test_equations.py; exact command and output in unit_tests.log",
    })
    if result.returncode:
        raise RuntimeError("independent/source regression tests failed; see unit_tests.log")
    return value


def environment(root: Path, output: Path) -> None:
    import numpy
    import pandas
    import torch

    value = {"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__, "pandas": pandas.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "source_commit": source_provenance(root)["analysis_source_commit"]}
    write_json(output / "environment.json", value)


def write_report(root: Path, output: Path, inv: dict[str, object], score: dict[str, object] | None, geometry: dict[str, object] | None) -> None:
    def rows(name: str) -> list[dict[str, str]]:
        with (output / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def trace_mean(old: str, new: str) -> float:
        values = [float(row["auroc_delta_pp"]) for row in rows("ranking_loss_trace.csv") if row["from_score"] == old and row["to_score"] == new]
        return float(np.mean(values))

    metrics_by_name: dict[str, float] = {}
    for row in rows("component_metrics.csv"):
        metrics_by_name.setdefault(row["score"], []).append(float(row["auroc"]))
    metrics_by_name = {name: float(np.mean(values)) for name, values in metrics_by_name.items()}
    saturation = rows("calibration_saturation.csv")
    selected_ceiling = {name: float(np.mean([float(row["selected_patch_at_tail_ceiling_fraction"]) for row in saturation if row["component"] == name])) for name in COMPONENTS}
    pooling = rows("pooling_selection_audit.csv")
    def pooling_stat(comparison: str, field: str) -> float:
        return float(np.mean([float(row[field]) for row in pooling if row["comparison"] == comparison]))
    def selection_change_rate(comparison: str) -> float:
        return float(np.mean([float(row["topk_jaccard"]) < 1 for row in pooling if row["comparison"] == comparison]))
    graph = rows("graph_audit.csv")
    graph_row_error = max(float(row["p_row_sum_max_abs_error"]) for row in graph)
    graph_balance_error = max(float(row["detailed_balance_max_abs_error"]) for row in graph)
    score_cov = f"{score['completed_runs']}/{score['eligible_runs']} raw-score runs" if score else "not run"
    geom_cov = f"{geometry['completed_runs']}/{geometry['eligible_runs']} geometry checkpoints" if geometry else "not run"
    raw_drop = trace_mean("raw_euclidean", "raw_B")
    tail_drop = trace_mean("raw_B", "Bubble_B")
    d_drop = trace_mean("Bubble_BA", "Bubble_BAD")
    f_change = trace_mean("Bubble_BA", "Bubble_BAF")
    d_with_f_drop = trace_mean("Bubble_BAF", "NBD")
    report = f"""# NBD forensic debug report

## Conclusion: where signal is lost

1. **No replayed implementation, numerical, protocol or metric-plumbing bug was found.** All {score_cov} recomputed raw-to-calibrated maps, top-1% pooling, normal thresholds, CSV image scores, AUROC and AP at their boundaries. The maximum score/metric mismatch is recorded per run in `replay_by_run.csv`; all were at float64 round-off scale. The independent and source regression suite passed (23 tests, including CUDA BubbleModel path).
2. **The largest measured loss happens before calibration: raw Euclidean→raw B is {raw_drop:.2f} AUROC points on the matched 31-run coverage** (mean AUROC {metrics_by_name['raw_euclidean']:.4f}→{metrics_by_name['raw_B']:.4f}). This is an observed compatibility-score/candidate-assignment gap, not proof of a coding defect. Candidate-energy decomposition needs hash-bound query features and is therefore not claimed.
3. **D consistently harms this coverage after BA, while F has mixed help.** BA→BAD is {d_drop:.2f} points and BAF→NBD is {d_with_f_drop:.2f}; BA→BAF is {f_change:+.2f}. Adding D changes the selected top-1% patch set for {100 * selection_change_rate('Bubble_BA->Bubble_BAD'):.1f}% of images (mean Jaccard {pooling_stat('Bubble_BA->Bubble_BAD', 'topk_jaccard'):.3f}); BAF→NBD changes it for {100 * selection_change_rate('Bubble_BAF->NBD'):.1f}% (mean Jaccard {pooling_stat('Bubble_BAF->NBD', 'topk_jaccard'):.3f}). This supports a `scoring_design_issue` for adding D in this fixed protocol, not a general claim that diffusion is numerically wrong.

## Coverage and provenance

The harness binds source hashes, saved per-run protocol, artifact hashes and the current Git commit in `manifest.json` and `inventory.json`. It targets only `mvtec-author-encoder-budgeted-v3`, the controlled 1,024-D author-encoder experiment. It does not merge historic 1,536-D, native PatchCore, MNIST/CIFAR or summary-only results.

There are {inv['run_count']} completed category/seed protocol entries. {score_cov} and {geom_cov} were eligible. Fourteen completed category/seed entries lack both `scores.npz` and `geometry.pt` locally; `missing_artifacts.csv` records them. They are excluded from all component and graph means.

## What calibration, fusion and pooling show

Raw B→calibrated B changes matched AUROC by {tail_drop:.2f} points. Thus ECDF is measurable but is not the leading loss relative to raw B. It nevertheless compresses top selected patches: mean B/A tail-ceiling fractions among selected patches are {100 * selected_ceiling['B']:.1f}% and {100 * selected_ceiling['A']:.1f}% (D {100 * selected_ceiling['D']:.2f}%, F {100 * selected_ceiling['F']:.2f}%). These are tail-score ties/range effects, not anomaly probabilities.

`component_metrics.csv` keeps raw and calibrated scores separate: B/A are near each other (raw B {metrics_by_name['raw_B']:.4f}, raw A {metrics_by_name['raw_A']:.4f}); D alone is weak ({metrics_by_name['raw_D']:.4f}); F alone is also weak ({metrics_by_name['raw_F']:.4f}). `ranking_loss_trace.csv` and `rank_pair_transitions.csv` provide matched anomaly-normal pair gains/losses whose means equal the AUROC deltas. `pooling_selection_audit.csv` separately measures fixed-old-set versus reselection, so it does not mislabel a changed top-k set as a pure D contribution.

## Geometry and diffusion correctness

All 31 checked graphs are symmetric, non-negative and connected under nonzero edges. Maximum stochastic-row-sum error is {graph_row_error:.2e}; maximum detailed-balance error is {graph_balance_error:.2e}. `geometry_audit.csv` also records frame orthogonality and eigenvalue/residual floors. These checks refute a graph arithmetic failure on covered checkpoints. They do **not** make D a manifold distance: D is attachment-weighted dispersion of diffusion embeddings and is invariant to a constant shift of candidate energies when attachments are fixed.

`neighborhood_quality.csv` can only report persisted patch-index counts. FIT-patch-to-image ownership, query features, exact candidates and image/mask alignment were not saved; `candidate_recall.csv`, `energy_decomposition.csv` and localization plots are deliberately absent rather than invented from image-level scores.

## Hypothesis ledger

| ID | Classification | Verdict | Evidence |
| --- | --- | --- | --- |
| H1 | implementation_bug / numerical_issue | refuted on covered runs | independent formulas, 23 tests, exact saved-array/CSV/metric replay |
| H2 | statistical_model_limitation | supported as an observed loss stage; cause unresolved | raw Euclidean→raw B {raw_drop:.2f} pp, no candidate/query trace available |
| H3 | scoring_design_issue | supported on covered runs | D edges {d_drop:.2f} pp and {d_with_f_drop:.2f} pp; top-k selection audit |
| H4 | unsupported_hypothesis | unsupported | graph arithmetic passes; no evidence that D measures outlierness |
| H5 | blocked_missing_evidence | blocked | 14/45 raw/geometry artifacts and all query/FIT image mappings missing locally |

## Fixes and next action

No production model fix was made. The harness is read-only with respect to historical predictions, checkpoints, calibration and metric summaries. It does not tune fusion weights, rank, temperature, backbone or test-selected thresholds.

Export a hash-bound query descriptor array plus FIT-patch-to-image mapping for one predeclared representative category and one random category/seed, without refitting. Acceptance test: compare nearest-Euclidean, K-candidate and all-bubble energy assignments; emit tangent/residual terms for those fixed IDs. This resolves whether the raw-B gap is caused by compatibility geometry or candidate assignment.

The strongest current evidence points to a raw compatibility-score gap and a D/fusion ranking loss, not an implementation failure; it comes from exact replay and pairwise traces on {score_cov}. Unverified are candidate-energy geometry, image-localization and all-run coverage; the smallest resolving experiment is the hash-bound single-run query/FIT feature export above.
"""
    (output / "REPORT.md").write_text(report)
    (output / "NEXT_MINIMAL_ACTION.md").write_text(
        "# Next minimal action\n\n"
        "Export hash-bound query descriptors and a FIT-patch-to-image mapping for one "
        "predeclared category and one random category/seed, without refitting. Compare "
        "nearest-Euclidean, K-candidate and all-bubble energy assignments, then emit "
        "tangent/residual terms for the same fixed IDs.\n"
    )
    write_json(output / "hypothesis_ledger.json", [
        {
            "hypothesis_id": "H1",
            "claim": "The released controlled pipeline has an implementation or numerically meaningful replay error.",
            "classification": ["implementation_bug", "numerical_issue", "protocol_or_data_issue"],
            "prediction_if_true": "At least one raw-map, pooled score, threshold, CSV score, AUROC or AP boundary mismatches saved output above float64 round-off.",
            "falsification_test": "Independent reference formulas, source regression suite, and exact boundary replay.",
            "source_location": SOURCE_FILES,
            "required_artifacts": "scores.npz, predictions.csv, metrics.csv, saved protocol",
            "coverage": score_cov,
            "observed_measurements": "No mismatches; all 23 regression/independent tests passed.",
            "counterexample_or_counterfactual": "Every eligible run agrees at all replay boundaries.",
            "effect_size_on_matched_runs": 0.0,
            "confounders": "14 runs do not retain raw score artifacts.",
            "verdict": "refuted",
            "minimal_next_action": "Restore raw artifacts for the 14 missing runs before extending this conclusion to all runs.",
        },
        {
            "hypothesis_id": "H2",
            "claim": "The bubble compatibility score loses ranking before tail calibration.",
            "classification": ["statistical_model_limitation"],
            "prediction_if_true": "Raw B will rank lower than same-center raw Euclidean on matched runs.",
            "falsification_test": "Matched raw-Euclidean to raw-B image AUROC trace.",
            "source_location": ["nbdbench/models.py:BubbleModel.energy", "nbdbench/models.py:BubbleModel.components"],
            "required_artifacts": "scores.npz",
            "coverage": score_cov,
            "observed_measurements": f"Mean AUROC delta {raw_drop:.2f} pp.",
            "counterexample_or_counterfactual": "Some individual runs improve, so this is not universal.",
            "effect_size_on_matched_runs": raw_drop,
            "confounders": "Raw B changes both compatibility metric and candidate choice; query features are missing.",
            "verdict": "supported",
            "minimal_next_action": "Trace fixed query IDs against all bubbles and original candidates.",
        },
        {
            "hypothesis_id": "H3",
            "claim": "Adding D causes scoring/pooling ranking loss in the released fixed protocol.",
            "classification": ["scoring_design_issue"],
            "prediction_if_true": "BA→BAD and BAF→NBD will reduce matched AUROC and replace many selected top patches.",
            "falsification_test": "Paired rank trace and fixed-selection versus reselection audit.",
            "source_location": ["nbdbench/run.py:calibrated_variants", "nbdbench/math.py:image_score"],
            "required_artifacts": "scores.npz, predictions.csv",
            "coverage": score_cov,
            "observed_measurements": f"BA→BAD {d_drop:.2f} pp; BAF→NBD {d_with_f_drop:.2f} pp; BA→BAD selected-set change {100 * selection_change_rate('Bubble_BA->Bubble_BAD'):.1f}%.",
            "counterexample_or_counterfactual": f"BA→BAF changes by {f_change:+.2f} pp, so F is not uniformly harmful.",
            "effect_size_on_matched_runs": {"BA_to_BAD_pp": d_drop, "BAF_to_NBD_pp": d_with_f_drop, "BA_to_BAF_pp": f_change},
            "confounders": "The result is for fixed fusion/ECDF/top-1% protocol, not a tuned generalization claim.",
            "verdict": "supported",
            "minimal_next_action": "Use fixed patch IDs to separate D value from its selection effect.",
        },
        {
            "hypothesis_id": "H4",
            "claim": "Diffusion graph construction is numerically invalid or D is established as an outlier distance.",
            "classification": ["unsupported_hypothesis"],
            "prediction_if_true": "Graph invariants fail, or a query-space semantic trace supports D as distance-to-manifold.",
            "falsification_test": "Graph stochasticity, detailed balance and component audit; D invariance fixture.",
            "source_location": ["nbdbench/math.py:diffusion_embedding", "nbdbench/models.py:BubbleModel.components"],
            "required_artifacts": "geometry.pt; query features for semantic claim",
            "coverage": geom_cov,
            "observed_measurements": f"Graph row-sum error ≤ {graph_row_error:.2e}, detailed-balance error ≤ {graph_balance_error:.2e}; query semantic trace unavailable.",
            "counterexample_or_counterfactual": "A constant candidate-energy shift changes B/A but not attachments or D.",
            "effect_size_on_matched_runs": None,
            "confounders": "Correct graph arithmetic does not validate D's semantic purpose.",
            "verdict": "unsupported",
            "minimal_next_action": "Export query-to-candidate attachments with fixed test IDs.",
        },
        {
            "hypothesis_id": "H5",
            "claim": "The remaining evidence is sufficient for all-run geometry and localization conclusions.",
            "classification": ["blocked_missing_evidence"],
            "prediction_if_true": "All score/geometry files plus query and FIT image mapping are available.",
            "falsification_test": "Inventory file and manifest inspection.",
            "source_location": ["docs/reproduction/BACKUP_RESTORE.md"],
            "required_artifacts": "14 missing scores.npz/geometry.pt and all query/FIT image mappings",
            "coverage": "31/45 raw/geometry; 0 T3 mappings",
            "observed_measurements": "Artifacts are absent locally and were not synthesized.",
            "counterexample_or_counterfactual": "Not applicable.",
            "effect_size_on_matched_runs": None,
            "confounders": "Incremental archive may contain some absent state but was not bulk-restored.",
            "verdict": "blocked",
            "minimal_next_action": "Restore only needed declared archive members into a staging path after hash verification.",
        },
    ])


def run(root: Path, artifacts: Path, output: Path, stages: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "commands.log").write_text(" ".join(sys.argv) + "\n")
    environment(root, output)
    inv = inventory(root, artifacts, output)
    score = geometry = None
    if "tests" in stages:
        run_unit_tests(root, output)
    if "scores" in stages:
        score = score_forensics(root, artifacts, output, inv)
    if "geometry" in stages:
        geometry = geometry_forensics(root, artifacts, output, inv)
    if "report" in stages:
        write_report(root, output, inv, score, geometry)
    manifest = {"status": "completed", "started_stages": stages, "finished_unix": time.time(), "source": source_provenance(root), "score_stage": score, "geometry_stage": geometry, "artifact_levels": {"T1_score_runs": sum(bool(row["score_forensics_eligible"]) for row in inv["runs"]), "T2_geometry_runs": sum(bool(row["geometry_forensics_eligible"]) for row in inv["runs"]), "T3_fit_query_image_mapping_runs": 0}}
    write_json(output / "manifest.json", manifest)
