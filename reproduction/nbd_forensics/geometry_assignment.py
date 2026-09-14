"""Read-only Phase 2 geometry/assignment diagnostics for persisted NBD runs.

This module never calls ``fit`` or ``run_one``.  It first records the fixed
sentinel selection and verifies the persisted checkpoint hashes.  It can then
derive FIT-patch ownership from the saved split order for every eligible run.
If a separately exported, hash-bound descriptor archive is supplied, it also
replays the S0--S3 controls from the Phase 2 protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nbdbench.math import squared_distances
from nbdbench.models import BubbleModel

from . import reference
from .analysis import metrics, pair_transition


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = ROOT / "results/mvtec-author-encoder-budgeted-v3"
SOURCE_FILES = [
    "nbdbench/models.py",
    "nbdbench/math.py",
    "nbdbench/run.py",
    "nbdbench/data.py",
    "reproduction/mvtec.py",
    "reproduction/patchcore.py",
]
RAW_REPLAY_ATOL = 1e-8
ENERGY_EQ_ATOL = 1e-10
TIE_ATOL = 1e-12
PRIMARY_NAMES = ("S0_euclidean_same_center", "S1_energy_nearest", "S2_energy_topk", "S3_energy_all")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    names = fieldnames or (sorted({name for row in rows for name in row}) if rows else [])
    with path.open("w", newline="") as stream:
        if not names:
            return
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def source_provenance(root: Path) -> dict[str, object]:
    current = git(root, "rev-parse", "HEAD")
    checkpoint_commit = "65a7e4cd80c2410685fb6fe7c29a7d858375b033"
    return {
        "analysis_source_commit": current,
        "checkpoint_reviewed_commit": checkpoint_commit,
        "current_head_matches_reviewed_commit": current == checkpoint_commit,
        "dirty_status": git(root, "status", "--porcelain=v1"),
        "source_hashes": {name: sha256(root / name) for name in SOURCE_FILES},
        "reviewed_commit_source_hashes": {
            name: hashlib.sha256(
                subprocess.check_output(["git", "show", f"{checkpoint_commit}:{name}"], cwd=root)
            ).hexdigest()
            for name in SOURCE_FILES
        },
    }


def runtime_environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }


def artifact_rows(artifacts: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for protocol_path in sorted(artifacts.glob("*/seed-*/protocol.json")):
        category = protocol_path.parent.parent.name
        seed = int(protocol_path.parent.name.removeprefix("seed-"))
        base = protocol_path.parent / "base"
        names = {
            "protocol": protocol_path,
            "complete": base / "COMPLETE.json",
            "geometry": base / "geometry.pt",
            "scores": base / "scores.npz",
            "splits": base / "splits.json",
        }
        row: dict[str, object] = {"category": category, "seed": seed}
        for name, path in names.items():
            row[f"{name}_path"] = path
            row[f"{name}_present"] = path.is_file()
        row["eligible"] = all(bool(row[f"{name}_present"]) for name in names)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"no controlled protocol files under {artifacts}")
    return rows


def verify_complete(row: dict[str, object]) -> dict[str, Any]:
    if not row["eligible"]:
        raise ValueError("cannot verify an ineligible run")
    complete_path = Path(row["complete_path"])
    complete = json.loads(complete_path.read_text())
    expected = complete.get("artifacts")
    if not isinstance(expected, dict):
        raise ValueError(f"{complete_path}: COMPLETE.json has no artifact map")
    verified: dict[str, str] = {}
    for name in ("geometry.pt", "scores.npz", "splits.json"):
        path = complete_path.parent / name
        if name not in expected:
            raise ValueError(f"{complete_path}: missing expected hash for {name}")
        actual = sha256(path)
        if actual != expected[name]:
            raise ValueError(f"{path}: hash differs from COMPLETE.json")
        verified[name] = actual
    protocol_path = Path(row["protocol_path"])
    protocol_sha = sha256(protocol_path)
    return {
        "complete": complete,
        "hashes": {"protocol.json": protocol_sha, **verified},
        "protocol": json.loads(protocol_path.read_text()),
    }


def load_geometry(row: dict[str, object], verified: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["geometry_path"])
    # The hash was just checked against COMPLETE.json.  Prefer tensor-only load,
    # then allow legacy NumPy metadata only after that provenance check.
    try:
        state = torch.load(path, weights_only=True, map_location="cpu")
    except Exception:
        state = torch.load(path, weights_only=False, map_location="cpu")
    if state.get("identity") != verified["complete"].get("identity"):
        raise ValueError(f"{path}: checkpoint identity differs from COMPLETE.json")
    bubble = state.get("bubble")
    if not isinstance(bubble, dict):
        raise ValueError(f"{path}: missing bubble state")
    return bubble


def feature_manifest(artifacts: Path, category: str) -> dict[str, Any]:
    path = artifacts / f"{category}_features.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    shape = value.get("feature_shape")
    if not (isinstance(shape, list) and len(shape) == 3 and all(isinstance(n, int) for n in shape)):
        raise ValueError(f"{path}: invalid feature_shape")
    return value


def fit_ownership(
    category: str,
    seed: int,
    splits: dict[str, list[str]],
    bubble: dict[str, Any],
    feature_shape: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Derive exact image ownership from run_one's persisted FIT flattening."""
    if "fit" not in splits or not isinstance(splits["fit"], list):
        raise ValueError("splits.json has no ordered fit list")
    patches = int(feature_shape[1])
    if patches <= 0:
        raise ValueError("feature manifest has no patches per image")
    fit_paths = splits["fit"]
    expected = len(fit_paths) * patches
    neighborhoods = np.asarray(bubble["neighborhood_indices"], dtype=np.int64)
    seed_indices = np.asarray(bubble["seed_indices"], dtype=np.int64)
    if neighborhoods.ndim != 2 or neighborhoods.size == 0:
        raise ValueError("invalid neighborhood_indices schema")
    indices = np.concatenate([neighborhoods.reshape(-1), seed_indices.reshape(-1)])
    if np.any(indices < 0) or np.any(indices >= expected):
        raise ValueError("neighborhood/seed index outside flattened FIT range")
    ownership: list[dict[str, object]] = []
    for fit_index in range(expected):
        image_slot, patch_offset = divmod(fit_index, patches)
        ownership.append(
            {
                "category": category,
                "seed": seed,
                "fit_patch_index": fit_index,
                "fit_image_slot": image_slot,
                "patch_offset": patch_offset,
                "image_id": fit_paths[image_slot],
                "patches_per_image": patches,
            }
        )
    diversity: list[dict[str, object]] = []
    all_sets = [set(row.tolist()) for row in neighborhoods]
    for bubble_id, row in enumerate(neighborhoods):
        slots = row // patches
        unique, counts = np.unique(slots, return_counts=True)
        other = set().union(*(all_sets[:bubble_id] + all_sets[bubble_id + 1 :]))
        diversity.append(
            {
                "category": category,
                "seed": seed,
                "bubble_id": bubble_id,
                "neighborhood_size": len(row),
                "neighborhood_unique_patch_count": len(set(row.tolist())),
                "neighborhood_duplicate_index_count": int(len(row) - len(set(row.tolist()))),
                "unique_images": len(unique),
                "dominant_image_id": fit_paths[int(unique[np.argmax(counts)])],
                "dominant_image_fraction": float(counts.max() / len(row)),
                "overlap_patch_count_with_other_neighborhoods": len(all_sets[bubble_id] & other),
                "source_order_verified": True,
            }
        )
    return ownership, diversity


def covariance_rows(category: str, seed: int, bubble: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, object]]:
    centers = np.asarray(bubble["centers"], dtype=np.float64)
    eigenvalues = np.asarray(bubble["eigenvalues"], dtype=np.float64)
    sigma = np.asarray(bubble["sigma"], dtype=np.float64)
    if centers.ndim != 2 or eigenvalues.ndim != 2 or len(centers) != len(eigenvalues):
        raise ValueError("incompatible center/eigenvalue schema")
    d, rank = centers.shape[1], eigenvalues.shape[1]
    beta, epsilon = float(cfg["beta"]), float(cfg["epsilon"])
    if not (0 < rank < d and beta > 0):
        raise ValueError("invalid rank/beta for covariance audit")
    ell, residual = eigenvalues + epsilon, sigma + epsilon
    value: list[dict[str, object]] = []
    for bubble_id in range(len(centers)):
        c0_trace_per_dim = (ell[bubble_id].sum() + (d - rank) * residual[bubble_id]) / d
        ceff_trace_per_dim = (
            rank * ell[bubble_id].sum() + ((d - rank) ** 2 / beta) * residual[bubble_id]
        ) / d
        value.append(
            {
                "category": category,
                "seed": seed,
                "bubble_id": bubble_id,
                "dimension": d,
                "rank": rank,
                "eigenvalue_min": float(ell[bubble_id].min()),
                "eigenvalue_max": float(ell[bubble_id].max()),
                "eigenvalue_mean": float(ell[bubble_id].mean()),
                "stored_sigma": float(sigma[bubble_id]),
                "sigma_plus_epsilon": float(residual[bubble_id]),
                "c0_trace_per_dimension": float(c0_trace_per_dim),
                "ceff_trace_per_dimension": float(ceff_trace_per_dim),
            }
        )
    return value


def energy_terms(
    query: np.ndarray,
    center: np.ndarray,
    frame: np.ndarray,
    eigenvalues: np.ndarray,
    sigma: float,
    beta: float,
    epsilon: float,
) -> dict[str, float]:
    """Independent float64 energy/Covariance calculation for one fixed bubble."""
    delta = np.asarray(query, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.float64)
    coordinates = frame.T @ delta
    ell = np.asarray(eigenvalues, dtype=np.float64) + epsilon
    residual_variance = float(sigma) + epsilon
    rank, dimension = frame.shape[1], len(delta)
    tangent_whitened = float(np.sum(coordinates**2 / ell))
    residual_squared = max(0.0, float(delta @ delta - coordinates @ coordinates))
    residual_whitened = residual_squared / residual_variance
    q = tangent_whitened + residual_whitened
    energy = tangent_whitened / rank + beta * residual_whitened / (dimension - rank)
    logdet_c0 = float(np.log(ell).sum() + (dimension - rank) * math.log(residual_variance))
    logdet_ceff = float(
        rank * math.log(rank)
        + np.log(ell).sum()
        + (dimension - rank) * math.log(((dimension - rank) / beta) * residual_variance)
    )
    return {
        "tangent_squared_norm": float(coordinates @ coordinates),
        "orthogonal_residual_squared_norm": residual_squared,
        "tangent_whitened_sum": tangent_whitened,
        "residual_whitened_sum": residual_whitened,
        "tangent_dimension_normalized": tangent_whitened / rank,
        "residual_dimension_normalized": residual_whitened / (dimension - rank),
        "energy_total": energy,
        "q_total": q,
        "logdet_c0": logdet_c0,
        "logdet_ceff": logdet_ceff,
        "nll0": 0.5 * (q + logdet_c0 + dimension * math.log(2 * math.pi)),
        "nll_eff": 0.5 * (energy + logdet_ceff + dimension * math.log(2 * math.pi)),
    }


def select_candidates(distances: np.ndarray, candidate_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable source-order top-K, with the first member defining jE on ties."""
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or candidate_count < 1:
        raise ValueError("distances must be [queries, bubbles] and K must be positive")
    count = min(candidate_count, distances.shape[1])
    ids = np.argsort(distances, axis=1, kind="stable")[:, :count]
    nearest = ids[:, 0]
    if not np.allclose(distances[np.arange(len(distances)), nearest], distances.min(1), atol=TIE_ATOL, rtol=0):
        raise AssertionError("jE must be a global Euclidean minimizer")
    return nearest, ids


def check_score_invariants(s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> None:
    if np.any(s2 > s1 + ENERGY_EQ_ATOL) or np.any(s3 > s2 + ENERGY_EQ_ATOL):
        raise AssertionError("energy search invariant S3 <= S2 <= S1 failed")


def descriptor_archive_path(root: Path, category: str, seed: int) -> Path:
    del seed  # Descriptors are category-level: all controlled seeds share image features.
    return root / category / "descriptors.npz"


def check_descriptor_archive(
    archive_path: Path,
    expected_feature_hash: str,
    splits: dict[str, list[str]],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != {"all_descriptors", "paths"}:
            raise ValueError(f"{archive_path}: expected all_descriptors and paths only")
        descriptors = np.asarray(archive["all_descriptors"], dtype=np.float32)
        paths = [str(path) for path in archive["paths"].tolist()]
    if descriptors.ndim != 3 or len(descriptors) != len(paths):
        raise ValueError(f"{archive_path}: descriptor/path schema mismatch")
    if array_sha256(descriptors) != expected_feature_hash:
        raise ValueError(f"{archive_path}: descriptor bytes do not match historical feature_sha256")
    expected_train = sorted(splits["fit"] + splits["score_calibration"] + splits["threshold_calibration"])
    expected_paths = expected_train + splits["test"]
    if paths != expected_paths:
        raise ValueError(f"{archive_path}: paths differ from reconstructed source order")
    positions = {path: index for index, path in enumerate(paths)}
    by_split = {name: descriptors[[positions[path] for path in values]] for name, values in splits.items()}
    return descriptors, paths, by_split


def _production_scores(
    descriptors: np.ndarray,
    bubble_state: dict[str, Any],
    cfg: dict[str, Any],
    chunk_size: int = 64,
) -> dict[str, np.ndarray]:
    """Replay the source candidate ordering and energy without fitting geometry."""
    model = BubbleModel(cfg).restore(bubble_state, "cpu")
    centers = model.centers
    count = min(int(cfg["candidate_count"]), len(centers))
    values: dict[str, list[np.ndarray]] = {
        "s0": [], "s1": [], "s2": [], "j_e": [], "j_k": [], "candidate_ids": [], "tie_count": []
    }
    flat = np.asarray(descriptors, dtype=np.float32).reshape(-1, descriptors.shape[-1])
    all_ids = torch.arange(len(centers), dtype=torch.long)[None]
    # S0--S2 use the source raw_scores batch size.  S3 is a new exhaustive
    # control and is evaluated separately with bounded working state.
    source_batch = int(cfg.get("score_batch", chunk_size))
    for start in range(0, len(flat), source_batch):
        x = torch.from_numpy(flat[start : start + source_batch])
        distances = squared_distances(x, centers)
        candidate_ids = distances.topk(count, largest=False).indices
        candidate_energy = model.energy(x, candidate_ids)
        j_e = candidate_ids[:, 0]
        j_k = candidate_ids.gather(1, candidate_energy.argmin(1, keepdim=True)).squeeze(1)
        global_min = distances.min(1).values
        tie_count = (distances - global_min[:, None]).abs().le(TIE_ATOL).sum(1)
        if not torch.allclose(distances.gather(1, j_e[:, None]).squeeze(1), global_min, atol=TIE_ATOL, rtol=0):
            raise AssertionError("source top-k first candidate is not a Euclidean minimizer")
        values["s0"].append(distances.gather(1, j_e[:, None]).squeeze(1).cpu().numpy())
        values["s1"].append(candidate_energy[:, 0].cpu().numpy())
        values["s2"].append(candidate_energy.min(1).values.cpu().numpy())
        values["j_e"].append(j_e.cpu().numpy())
        values["j_k"].append(j_k.cpu().numpy())
        values["candidate_ids"].append(candidate_ids.cpu().numpy())
        values["tie_count"].append(tie_count.cpu().numpy())
    result = {name: np.concatenate(parts) for name, parts in values.items()}
    all_scores, all_assignments = [], []
    exhaustive_batch = min(chunk_size, 64)
    for start in range(0, len(flat), exhaustive_batch):
        x = torch.from_numpy(flat[start : start + exhaustive_batch])
        energy = model.energy(x, all_ids.expand(len(x), -1))
        all_scores.append(energy.min(1).values.cpu().numpy())
        all_assignments.append(energy.argmin(1).cpu().numpy())
    result["s3"] = np.concatenate(all_scores)
    result["j_all"] = np.concatenate(all_assignments)
    check_score_invariants(result["s1"], result["s2"], result["s3"])
    return result


def _image_scores(values: np.ndarray, fraction: float, image_count: int, patches: int) -> np.ndarray:
    return reference.top_mean(values.reshape(image_count, patches), fraction)


def _raw_replay_errors(
    score_sets: dict[str, dict[str, np.ndarray]],
    scores_path: Path,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    with np.load(scores_path, allow_pickle=False) as archive:
        for split, saved_key in (("score_calibration", "calibration_raw"), ("threshold_calibration", "threshold_raw"), ("test", "test_raw")):
            saved = archive[saved_key]
            current = score_sets[split]
            if saved.shape[:2] != (len(current["s0"]) // saved.shape[1], saved.shape[1]):
                raise ValueError(f"{split}: descriptor shape does not match saved raw-score shape")
            errors[f"{split}_s0_max_abs_error"] = float(np.max(np.abs(current["s0"].reshape(saved.shape[:2]) - saved[..., 0])))
            errors[f"{split}_s2_max_abs_error"] = float(np.max(np.abs(current["s2"].reshape(saved.shape[:2]) - saved[..., 5])))
    return errors


def _trace_rows(
    category: str,
    seed: int,
    paths: list[str],
    descriptors: np.ndarray,
    scores: dict[str, np.ndarray],
    bubble: dict[str, Any],
    cfg: dict[str, Any],
) -> Iterable[dict[str, object]]:
    centers = np.asarray(bubble["centers"], dtype=np.float64)
    frames = np.asarray(bubble["u"], dtype=np.float64)
    eigenvalues = np.asarray(bubble["eigenvalues"], dtype=np.float64)
    sigma = np.asarray(bubble["sigma"], dtype=np.float64)
    flat = descriptors.reshape(-1, descriptors.shape[-1])
    patches = descriptors.shape[1]
    candidate_ids = scores["candidate_ids"]
    for index, query in enumerate(flat):
        image_slot, patch_offset = divmod(index, patches)
        distances = np.sum((centers - query.astype(np.float64)) ** 2, axis=1)
        for role, bubble_ids in (("jE", scores["j_e"]), ("jK", scores["j_k"]), ("jAll", scores["j_all"])):
            bubble_id = int(bubble_ids[index])
            terms = energy_terms(query, centers[bubble_id], frames[bubble_id], eigenvalues[bubble_id], float(sigma[bubble_id]), float(cfg["beta"]), float(cfg["epsilon"]))
            rank = int(np.count_nonzero(distances < distances[bubble_id] - TIE_ATOL) + 1)
            yield {
                "category": category,
                "seed": seed,
                "split": "test",
                "image_id": paths[image_slot],
                "patch_offset": patch_offset,
                "bubble_id": bubble_id,
                "assignment_role": role,
                "euclidean_rank_of_bubble": rank,
                "euclidean_squared_distance": float(distances[bubble_id]),
                "candidate_member": bool(bubble_id in candidate_ids[index]),
                "tie_count_at_global_minimum": int(scores["tie_count"][index]),
                "assignment_drop": float(scores["s1"][index] - scores["s2"][index]),
                "candidate_regret": float(scores["s2"][index] - scores["s3"][index]),
                "stored_sigma": float(sigma[bubble_id]),
                "sigma_plus_epsilon": float(sigma[bubble_id] + float(cfg["epsilon"])),
                "eigenvalue_min": float(eigenvalues[bubble_id].min()),
                "eigenvalue_max": float(eigenvalues[bubble_id].max()),
                **terms,
            }


def _primary_rows(
    category: str,
    seed: int,
    by_split: dict[str, np.ndarray],
    paths_by_split: dict[str, list[str]],
    scores_path: Path,
    bubble: dict[str, Any],
    cfg: dict[str, Any],
    trace_writer: csv.DictWriter[str] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    score_sets = {
        name: _production_scores(by_split[name], bubble, cfg)
        for name in ("score_calibration", "threshold_calibration", "test")
    }
    errors = _raw_replay_errors(score_sets, scores_path)
    if max(errors.values(), default=0.0) > RAW_REPLAY_ATOL:
        raise ValueError(f"raw replay exceeds locked tolerance {RAW_REPLAY_ATOL}: {errors}")
    with np.load(scores_path, allow_pickle=False) as archive:
        labels = archive["test_labels"].astype(int)
    test = score_sets["test"]
    patches = by_split["test"].shape[1]
    fraction = float(cfg["image_top_fraction"])
    images = {name: _image_scores(test[f"s{index}"], fraction, len(labels), patches) for index, name in enumerate(PRIMARY_NAMES)}
    rows: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    ranks: list[dict[str, object]] = []
    candidate: list[dict[str, object]] = []
    for name in PRIMARY_NAMES:
        rows.append({"category": category, "seed": seed, "score": name, "raw_replay_status": "passed", **metrics(labels, images[name])})
    for old_name, new_name in zip(PRIMARY_NAMES, PRIMARY_NAMES[1:]):
        old, new = images[old_name], images[new_name]
        transition = pair_transition(labels, old, new)
        transitions.append({"category": category, "seed": seed, "from_score": old_name, "to_score": new_name, "auroc_delta_pp": 100 * (reference.auroc_pairwise(labels, new) - reference.auroc_pairwise(labels, old)), **transition})
        ranks.append({"category": category, "seed": seed, "from_score": old_name, "to_score": new_name, **transition})
    in_candidate = np.any(test["candidate_ids"] == test["j_all"][:, None], axis=1)
    energy_equivalent = np.abs(test["s2"] - test["s3"]) <= ENERGY_EQ_ATOL
    for label_name, mask in (("all", np.ones(len(test["s0"]), dtype=bool)), ("normal", np.repeat(labels == 0, patches)), ("anomaly_image", np.repeat(labels == 1, patches))):
        candidate.append({
            "category": category,
            "seed": seed,
            "split": "test",
            "population": label_name,
            "patch_count": int(mask.sum()),
            "jE_in_candidate_fraction": 1.0,
            "exact_jAll_candidate_membership_recall": float(in_candidate[mask].mean()),
            "energy_equivalent_candidate_recall": float(energy_equivalent[mask].mean()),
            "candidate_regret_mean": float((test["s2"] - test["s3"])[mask].mean()),
            "assignment_drop_mean": float((test["s1"] - test["s2"])[mask].mean()),
            "tie_patch_fraction": float((test["tie_count"][mask] > 1).mean()),
        })
    if trace_writer is not None:
        trace_writer.writerows(_trace_rows(category, seed, paths_by_split["test"], by_split["test"], test, bubble, cfg))
    return rows, candidate, transitions, ranks


def run(root: Path, artifacts: Path, output: Path, descriptors: Path | None) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    provenance = source_provenance(root)
    environment = runtime_environment()
    rows = artifact_rows(artifacts)
    verified_runs: list[tuple[dict[str, object], dict[str, Any]]] = []
    coverage: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    if not environment["cuda_available"]:
        blockers.append({"category": "", "seed": "", "stage": "runtime", "reason": "no CUDA device on this host; the source-faithful author descriptor exporter refuses a CPU fallback"})
    for row in rows:
        basic = {"category": row["category"], "seed": row["seed"], "eligible": row["eligible"]}
        if not row["eligible"]:
            coverage.append({**basic, "checkpoint_status": "missing_input"})
            blockers.append({**basic, "stage": "inventory", "reason": "requires protocol, COMPLETE, geometry, scores and splits"})
            continue
        try:
            verified = verify_complete(row)
            verified_runs.append((row, verified))
            coverage.append({**basic, "checkpoint_status": "verified"})
        except Exception as error:  # Keep a per-run audit trail; do not infer eligibility.
            coverage.append({**basic, "checkpoint_status": "hash_or_identity_mismatch"})
            blockers.append({**basic, "stage": "checkpoint_verification", "reason": str(error)})
    eligible = [(str(row["category"]), int(row["seed"])) for row, _ in verified_runs]
    selection_hash = hashlib.sha256(json.dumps(eligible, separators=(",", ":")).encode()).hexdigest()
    if ("bottle", 0) in eligible:
        sentinel_a = ("bottle", 0)
        a_rule = "fixed bottle/seed-0"
    else:
        sentinel_a = eligible[0]
        a_rule = "fallback first lexicographic eligible run"
    alternatives = [item for item in eligible if item[0] != sentinel_a[0]]
    if alternatives:
        rng = np.random.default_rng(20260914)
        sentinel_b_index = int(rng.integers(len(alternatives)))
        sentinel_b = alternatives[sentinel_b_index]
        b_rule = "numpy.default_rng(20260914).integers over sorted eligible runs with a different category"
    else:
        sentinel_b_index = 0
        sentinel_b = next(item for item in eligible if item != sentinel_a)
        b_rule = "fallback only: second eligible run because no other category exists"
    selection = {
        "eligible_runs": eligible,
        "eligible_list_sha256": selection_hash,
        "sentinel_a": {"category": sentinel_a[0], "seed": sentinel_a[1], "rule": a_rule},
        "sentinel_b": {"category": sentinel_b[0], "seed": sentinel_b[1], "draw_index": sentinel_b_index, "rule": b_rule},
        "rng": {"library": "numpy", "version": np.__version__, "seed": 20260914},
        "saved_before_descriptor_or_label_read": True,
        "tolerances": {"raw_replay_atol": RAW_REPLAY_ATOL, "energy_equivalence_atol": ENERGY_EQ_ATOL, "tie_atol": TIE_ATOL},
    }
    # This is deliberately written before any scores.npz content or test labels are opened.
    write_json(output / "selection_plan.json", selection)

    dataset_preflight: list[dict[str, object]] = []
    data_root = root / "data/mvtec_ad"
    for category, seed in (sentinel_a, sentinel_b):
        try:
            manifest = feature_manifest(artifacts, category)
            if not (data_root / category).is_dir():
                raise FileNotFoundError(data_root / category)
            # Selection and tolerances are now immutable.  Dataset labels/masks
            # are inspected only to establish source parity, never to tune a score.
            from nbdbench.data import inspect_dataset

            observed = inspect_dataset(data_root, [category])
            matched = observed == manifest.get("dataset")
            dataset_preflight.append({
                "category": category,
                "seed": seed,
                "status": "matched_historical_manifest" if matched else "dataset_manifest_mismatch",
                "historical_dataset_manifest_sha256": hashlib.sha256(json.dumps(manifest.get("dataset"), sort_keys=True).encode()).hexdigest(),
                "observed_dataset_manifest_sha256": hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest(),
            })
            if not matched:
                blockers.append({"category": category, "seed": seed, "stage": "dataset_preflight", "reason": "extracted MVTec manifest differs from the category feature manifest"})
        except Exception as error:
            dataset_preflight.append({"category": category, "seed": seed, "status": "not_verified", "reason": str(error)})
            blockers.append({"category": category, "seed": seed, "stage": "dataset_preflight", "reason": str(error)})
    write_json(output / "dataset_preflight.json", {"data_root": str(data_root), "sentinels": dataset_preflight})

    ownership_path = output / "fit_patch_ownership.csv"
    ownership_fields = ["category", "seed", "fit_patch_index", "fit_image_slot", "patch_offset", "image_id", "patches_per_image"]
    diversity: list[dict[str, object]] = []
    covariance: list[dict[str, object]] = []
    with ownership_path.open("w", newline="") as stream:
        owner_writer = csv.DictWriter(stream, fieldnames=ownership_fields)
        owner_writer.writeheader()
        for row, verified in verified_runs:
            category, seed = str(row["category"]), int(row["seed"])
            try:
                bubble = load_geometry(row, verified)
                splits = json.loads(Path(row["splits_path"]).read_text())
                manifest = feature_manifest(artifacts, category)
                ownership, diversity_rows = fit_ownership(category, seed, splits, bubble, manifest["feature_shape"])
                owner_writer.writerows(ownership)
                diversity.extend(diversity_rows)
                covariance.extend(covariance_rows(category, seed, bubble, verified["protocol"]["config"]))
                for coverage_row in coverage:
                    if coverage_row["category"] == category and coverage_row["seed"] == seed:
                        coverage_row.update({"mapping_status": "derived", "patches_per_image": manifest["feature_shape"][1], "fit_patches": len(ownership)})
            except Exception as error:
                blockers.append({"category": category, "seed": seed, "stage": "fit_mapping", "reason": str(error)})
                for coverage_row in coverage:
                    if coverage_row["category"] == category and coverage_row["seed"] == seed:
                        coverage_row["mapping_status"] = "blocked"
    write_rows(output / "neighborhood_image_diversity.csv", diversity)
    write_rows(output / "bubble_covariance_stats.csv", covariance)

    parity: list[dict[str, object]] = []
    for category, seed in (sentinel_a, sentinel_b):
        row, verified = next(item for item in verified_runs if (item[0]["category"], item[0]["seed"]) == (category, seed))
        status: dict[str, object] = {
            "category": category,
            "seed": seed,
            "historical_feature_sha256": verified["protocol"]["feature_sha256"],
            "geometry_sha256": verified["hashes"]["geometry.pt"],
            "splits_sha256": verified["hashes"]["splits.json"],
            "status": "blocked_no_descriptor_archive",
        }
        if descriptors is None:
            status["blocker"] = "no descriptor root supplied; the historical runs did not persist query descriptors"
        else:
            archive_path = descriptor_archive_path(descriptors, category, seed)
            if not archive_path.is_file():
                status["blocker"] = f"missing {archive_path}"
            else:
                try:
                    splits = json.loads(Path(row["splits_path"]).read_text())
                    full, paths, _ = check_descriptor_archive(archive_path, verified["protocol"]["feature_sha256"], splits)
                    status.update({"status": "hash_parity_passed", "descriptor_path": str(archive_path), "descriptor_shape": list(full.shape), "descriptor_sha256": sha256(archive_path), "descriptor_content_sha256": array_sha256(full), "ordered_path_count": len(paths)})
                except Exception as error:
                    status.update({"status": "hash_parity_failed", "blocker": str(error)})
        parity.append(status)
        if status["status"] != "hash_parity_passed":
            blockers.append({"category": category, "seed": seed, "stage": "descriptor_parity", "reason": status.get("blocker", "unknown descriptor blocker")})
    write_json(output / "source_and_feature_parity.json", {"source": provenance, "sentinels": parity})

    metric_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    trace_fields = [
        "category", "seed", "split", "image_id", "patch_offset", "bubble_id", "assignment_role",
        "euclidean_rank_of_bubble", "euclidean_squared_distance", "candidate_member",
        "tie_count_at_global_minimum", "assignment_drop", "candidate_regret", "stored_sigma",
        "sigma_plus_epsilon", "eigenvalue_min", "eigenvalue_max", "tangent_squared_norm",
        "orthogonal_residual_squared_norm", "tangent_whitened_sum", "residual_whitened_sum",
        "tangent_dimension_normalized", "residual_dimension_normalized", "energy_total", "q_total",
        "logdet_c0", "logdet_ceff", "nll0", "nll_eff",
    ]
    with (output / "patch_energy_trace.csv").open("w", newline="") as trace_stream:
        trace_writer = csv.DictWriter(trace_stream, fieldnames=trace_fields)
        trace_writer.writeheader()
        for status in parity:
            if status["status"] != "hash_parity_passed":
                continue
            category, seed = str(status["category"]), int(status["seed"])
            row, verified = next(item for item in verified_runs if (item[0]["category"], item[0]["seed"]) == (category, seed))
            try:
                splits = json.loads(Path(row["splits_path"]).read_text())
                _, _, by_split = check_descriptor_archive(Path(str(status["descriptor_path"])), str(status["historical_feature_sha256"]), splits)
                bubble = load_geometry(row, verified)
                found = _primary_rows(
                    category,
                    seed,
                    by_split,
                    {name: list(paths) for name, paths in splits.items()},
                    Path(row["scores_path"]),
                    bubble,
                    verified["protocol"]["config"],
                    trace_writer,
                )
                metric_rows.extend(found[0])
                candidate_rows.extend(found[1])
                transition_rows.extend(found[2])
                rank_rows.extend(found[3])
            except Exception as error:
                blockers.append({"category": category, "seed": seed, "stage": "primary_raw_replay", "reason": str(error)})
    write_rows(output / "assignment_metrics.csv", metric_rows, ["category", "seed", "score", "raw_replay_status", "auroc", "average_precision"])
    write_rows(output / "candidate_recall_regret.csv", candidate_rows, ["category", "seed", "split", "population", "patch_count", "jE_in_candidate_fraction", "exact_jAll_candidate_membership_recall", "energy_equivalent_candidate_recall", "candidate_regret_mean", "assignment_drop_mean", "tie_patch_fraction"])
    write_rows(output / "assignment_transition_summary.csv", transition_rows, ["category", "seed", "from_score", "to_score", "auroc_delta_pp", "pairs", "gain_pairs", "loss_pairs", "unchanged_pairs", "auroc_delta_from_pairs"])
    write_rows(output / "rank_pair_transitions.csv", rank_rows, ["category", "seed", "from_score", "to_score", "pairs", "gain_pairs", "loss_pairs", "unchanged_pairs", "auroc_delta_from_pairs"])
    write_rows(output / "covariance_sensitivity.csv", [{"status": "not_run", "reason": "fixed-assignment sensitivity controls are deferred until a primary replay passes"}], ["status", "reason"])
    write_rows(output / "blockers.csv", blockers)
    write_rows(output / "coverage.csv", coverage)
    manifest = {
        "phase": "geometry_assignment_forensics",
        "artifacts_root": str(artifacts),
        "output": str(output),
        "source": provenance,
        "environment": environment,
        "selection_plan_sha256": sha256(output / "selection_plan.json"),
        "eligible_checkpoint_count": len(verified_runs),
        "descriptor_root": str(descriptors) if descriptors else None,
        "dataset_preflight_sha256": sha256(output / "dataset_preflight.json"),
        "primary_controls_status": "not_run_without_hash_parity_descriptors",
        "memory_plan": {
            "candidate_distance_state": "[chunk, 128] float64",
            "energy_state": "[chunk, candidate_count, 1024] for S0--S2; S3 must iterate bubbles/chunks rather than materialize all queries",
            "planned_precision": "float64 geometry arithmetic after frozen float32 descriptors",
        },
    }
    write_json(output / "manifest.json", manifest)
    report = (
        "# Geometry-assignment forensic — Phase 2\n\n"
        "## Raw-B gap and coverage\n\n"
        "Phase 2 has not re-created the raw-B gap on either sentinel: query descriptors were not persisted in the controlled artifacts. "
        f"It verified {len(verified_runs)} of 45 planned checkpoint/raw-score runs, then derived FIT ownership for all {len(verified_runs)} verified checkpoints. "
        "The two sentinels were fixed before descriptor or label content was read: bottle/seed-0 and metal_nut/seed-0.\n\n"
        "S0→S1, S1→S2 and S2→S3 AUROC/AP deltas are not run, as are patch witnesses and covariance sensitivities. "
        "The current host has no CUDA device, and no Vast instance was available to run the source-faithful frozen author encoder. "
        "This is a provenance/parity blocker, not evidence of an implementation, numerical, or scoring-model defect.\n\n"
        "The extracted local MVTec manifests for both sentinels match their persisted feature manifests; see `dataset_preflight.json`.\n\n"
        "## FIT mapping\n\n"
        "The persisted source schema supports exact FIT ownership: `run_one` flattens `features[split[\"fit\"]]` image-major, and the feature manifest proves 784 patches per image. "
        "`fit_patch_ownership.csv` and `neighborhood_image_diversity.csv` characterize sample ownership only; they do not establish an AUROC mechanism.\n\n"
        "## Minimal next action\n\n"
        "On an existing CUDA host, run `python -m reproduction.export_author_descriptors --category bottle --category metal_nut --selection-plan <this-output>/selection_plan.json --output <new-descriptor-root>`. "
        "The exporter refuses an output unless the ordered full descriptor array hashes to each historical `feature_sha256`; then rerun this module with `--descriptors <new-descriptor-root>`.\n"
    )
    (output / "REPORT.md").write_text(report)
    commands = (
        "# Local mapping/checkpoint-only run\n"
        "python -m pytest tests/test_nbd_forensics.py tests/test_geometry_assignment_forensics.py -q\n"
        "python -m reproduction.nbd_forensics.geometry_assignment --output results/geometry-assignment-forensics-<run-id>\n\n"
        "# After hash-bound GPU descriptor export\n"
        "python -m reproduction.export_author_descriptors --category bottle --category metal_nut --selection-plan results/geometry-assignment-forensics-<run-id>/selection_plan.json --output results/phase2-descriptors-<run-id>\n"
        "python -m reproduction.nbd_forensics.geometry_assignment --descriptors results/phase2-descriptors-<run-id> --output results/geometry-assignment-forensics-<run-id>\n"
    )
    (output / "tests_and_commands.txt").write_text(commands)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--descriptors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(ROOT, args.artifacts.resolve(), args.output.resolve(), args.descriptors.resolve() if args.descriptors else None)


if __name__ == "__main__":
    main()
