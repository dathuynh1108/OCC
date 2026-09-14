import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from .data import (
    EXPECTED,
    FrozenBackbone,
    array_sha256,
    extract_category,
    inspect_dataset,
    sha256_file,
    split_normal_indices,
)
from .heads import DeepHead, DROCCHead, atomic_save, fit_deep, fit_drocc
from .math import (
    empirical_tail,
    farthest_first,
    image_score,
    normal_threshold,
    squared_distances,
)
from .models import BubbleModel, KernelSVDD

METHODS = [
    "PatchScore_same_centers",
    "PatchScore_byte_budget",
    "RBF_SVDD",
    "DeepSVDD_head",
    "DROCC_head",
    "Bubble_B",
    "Bubble_BA",
    "Bubble_BAD",
    "Bubble_BAF",
    "NBD",
]


def write_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def raw_scores(features, bubble, memory, kernel, deep, drocc, cfg, device):
    shape = features.shape[:2]
    rows = []
    for offset in range(0, features.shape[0] * features.shape[1], cfg["score_batch"]):
        x = torch.from_numpy(
            features.reshape(-1, features.shape[-1])[
                offset : offset + cfg["score_batch"]
            ]
        ).to(device)
        baselines = [
            squared_distances(x, bubble.centers).min(1).values,
            squared_distances(x, memory).min(1).values,
            kernel.score(x),
            deep.score(x),
            drocc.score(x),
        ]
        scores = torch.cat(
            [torch.stack(baselines, dim=1).double(), bubble.components(x).double()],
            dim=1,
        )
        if not torch.isfinite(scores).all():
            raise RuntimeError("non-finite score")
        rows.append(scores.cpu().numpy())
    return np.concatenate(rows).reshape(*shape, 9)


def calibrated_variants(raw, cal, cfg):
    parts = np.stack(
        [empirical_tail(raw[..., i], cal[..., i].reshape(-1)) for i in range(5, 9)],
        axis=-1,
    )
    b, a, d, f = np.moveaxis(parts, -1, 0)
    variants = np.stack(
        [
            b,
            b + a,
            b + a + d,
            b + a + cfg["frame_weight"] * f,
            b + a + d + cfg["frame_weight"] * f,
        ],
        axis=-1,
    )
    return np.concatenate([raw[..., :5], variants], axis=-1)


def metric_row(category, seed, method, labels, scores, threshold, threshold_scores):
    predicted = scores > threshold
    normal, anomaly = labels == 0, labels == 1
    return {
        "category": category,
        "seed": seed,
        "method": method,
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "threshold": threshold if np.isfinite(threshold) else "inf",
        "test_fpr": float(predicted[normal].mean()),
        "test_tpr": float(predicted[anomaly].mean()),
        "false_positives": int(predicted[normal].sum()),
        "true_positives": int(predicted[anomaly].sum()),
        "test_normal": int(normal.sum()),
        "test_anomaly": int(anomaly.sum()),
        "threshold_calibration_count": len(threshold_scores),
        "threshold_calibration_fpr": float((threshold_scores > threshold).mean()),
    }


def run_one(features, paths, cfg, category, seed, directory, protocol_identity, device):
    directory.mkdir(parents=True, exist_ok=True)
    identity = identity_hash(
        {"protocol": protocol_identity, "category": category, "seed": seed}
    )
    if (directory / "COMPLETE.json").exists():
        complete = json.loads((directory / "COMPLETE.json").read_text())
        if complete["identity"] != identity:
            raise RuntimeError("completed output has different protocol identity")
        for name, digest in complete["artifacts"].items():
            if sha256_file(directory / name) != digest:
                raise RuntimeError(f"completed artifact changed: {name}")
        print(f"Already verified {category} seed={seed}", flush=True)
        return
    started = time.time()
    count = EXPECTED[category][0]
    split = split_normal_indices(count, seed, cfg)
    split_record = {k: [paths[i] for i in ids] for k, ids in split.items()}
    split_record["test"] = paths[count:]
    write_json(directory / "splits.json", split_record)
    print(
        f"FIT {category} seed={seed} images={len(split['fit'])} patches={len(split['fit']) * 784}",
        flush=True,
    )
    x = torch.from_numpy(features[split["fit"]].reshape(-1, features.shape[-1])).to(
        device
    )
    if (directory / "geometry.pt").exists():
        saved = torch.load(
            directory / "geometry.pt", weights_only=False, map_location="cpu"
        )
        if saved["identity"] != identity:
            raise RuntimeError("geometry protocol identity changed")
        bubble = BubbleModel(cfg).restore(saved["bubble"], device)
        memory = saved["memory"].to(device)
        kernel = KernelSVDD().restore(saved["kernel"], device)
        budget = saved["budget"]
    else:
        bubble = BubbleModel(cfg).fit(x, seed)
        nbd_bytes = bubble.geometry_bytes()
        memory_count = min(len(x), nbd_bytes // (x.shape[1] * x.element_size()))
        selected = farthest_first(
            x, max(memory_count, min(cfg["svdd_support_budget"], len(x))), seed
        )
        memory = x[selected[:memory_count]].clone()
        support_ids = selected[: cfg["svdd_support_budget"]]
        kernel = KernelSVDD().fit(x[support_ids], cfg["svdd_nu"])
        budget = {
            "nbd_geometry_bytes": nbd_bytes,
            "patchscore_geometry_bytes": memory.numel() * memory.element_size(),
            "patchscore_memory_vectors": len(memory),
            "svdd_support_vectors_for_fit": len(support_ids),
            "shared_backbone_and_component_calibration_excluded": True,
        }
        atomic_save(
            {
                "identity": identity,
                "bubble": bubble.state(),
                "memory": memory.cpu(),
                "memory_fit_indices": selected[:memory_count].cpu(),
                "kernel": kernel.state(),
                "svdd_fit_indices": support_ids.cpu(),
                "budget": budget,
            },
            directory / "geometry.pt",
        )
    write_json(
        directory / "geometry_diagnostics.json",
        {"nbd": bubble.fit_diagnostics, "svdd": kernel.diagnostics, **budget},
    )
    deep, deep_diagnostics = fit_deep(x, cfg, directory, seed, identity)
    drocc, drocc_diagnostics = fit_drocc(x, cfg, directory, seed, identity)
    del x
    torch.cuda.empty_cache()
    print(f"CALIBRATE {category} seed={seed}", flush=True)
    calibration = raw_scores(
        features[split["score_calibration"]],
        bubble,
        memory,
        kernel,
        deep,
        drocc,
        cfg,
        device,
    )
    threshold_raw = raw_scores(
        features[split["threshold_calibration"]],
        bubble,
        memory,
        kernel,
        deep,
        drocc,
        cfg,
        device,
    )
    threshold_maps = calibrated_variants(threshold_raw, calibration, cfg)
    threshold_images = image_score(
        threshold_maps.transpose(0, 2, 1), cfg["image_top_fraction"]
    )
    thresholds = np.array(
        [
            normal_threshold(threshold_images[:, i], cfg["alpha"])
            for i in range(len(METHODS))
        ]
    )
    write_json(
        directory / "thresholds.json",
        {
            method: {
                "alpha": cfg["alpha"],
                "threshold": float(thresholds[i])
                if np.isfinite(thresholds[i])
                else "inf",
                "heldout_normal_images": len(threshold_images),
            }
            for i, method in enumerate(METHODS)
        },
    )
    # Test prediction happens only after the terminal models and all thresholds are fixed.
    print(f"TEST {category} seed={seed}", flush=True)
    test_raw = raw_scores(
        features[count:], bubble, memory, kernel, deep, drocc, cfg, device
    )
    test_maps = calibrated_variants(test_raw, calibration, cfg)
    image_scores = image_score(test_maps.transpose(0, 2, 1), cfg["image_top_fraction"])
    labels = np.array([int(Path(p).parent.name != "good") for p in paths[count:]])
    rows, predictions = [], []
    for i, method in enumerate(METHODS):
        rows.append(
            metric_row(
                category,
                seed,
                method,
                labels,
                image_scores[:, i],
                thresholds[i],
                threshold_images[:, i],
            )
        )
        for j, path in enumerate(paths[count:]):
            predictions.append(
                {
                    "category": category,
                    "seed": seed,
                    "method": method,
                    "path": path,
                    "label": int(labels[j]),
                    "score": float(image_scores[j, i]),
                    "threshold": float(thresholds[i])
                    if np.isfinite(thresholds[i])
                    else "inf",
                }
            )
    pd.DataFrame(rows).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(directory / "predictions.csv", index=False)
    np.savez_compressed(
        directory / "scores.npz",
        calibration_raw=calibration,
        threshold_raw=threshold_raw,
        test_raw=test_raw,
        test_maps=test_maps,
        test_image_scores=image_scores,
        test_labels=labels,
        thresholds=thresholds,
    )
    del bubble, memory, kernel, deep, drocc
    gc.collect()
    torch.cuda.empty_cache()
    # Full saved-checkpoint replay on every calibration, threshold and test patch.
    print(f"REPLAY {category} seed={seed}", flush=True)
    saved = torch.load(
        directory / "geometry.pt", weights_only=False, map_location="cpu"
    )
    bubble = BubbleModel(cfg).restore(saved["bubble"], device)
    memory = saved["memory"].to(device)
    kernel = KernelSVDD().restore(saved["kernel"], device)
    deep = DeepHead(features.shape[-1], cfg["head_hidden"], cfg["head_output"]).to(
        device
    )
    deep.load_state_dict(
        torch.load(directory / "deep_svdd.pt", map_location=device, weights_only=False)[
            "model"
        ]
    )
    deep.eval()
    drocc = DROCCHead(features.shape[-1], cfg["head_hidden"]).to(device)
    drocc.load_state_dict(
        torch.load(directory / "drocc.pt", map_location=device, weights_only=False)[
            "model"
        ]
    )
    drocc.eval()
    maximum_error = 0.0
    for name, ids, expected in [
        ("score_calibration", split["score_calibration"], calibration),
        ("threshold_calibration", split["threshold_calibration"], threshold_raw),
        ("test", np.arange(count, len(features)), test_raw),
    ]:
        replay = raw_scores(
            features[ids], bubble, memory, kernel, deep, drocc, cfg, device
        )
        error = float(np.max(np.abs(replay - expected)))
        maximum_error = max(maximum_error, error)
        if not np.allclose(replay, expected, rtol=1e-6, atol=1e-7):
            raise RuntimeError(f"checkpoint replay failed {name}: max error={error}")
    reloaded = pd.read_csv(directory / "predictions.csv")
    for row in rows:
        part = reloaded[reloaded.method == row["method"]]
        if not np.isclose(
            roc_auc_score(part.label, part.score), row["auroc"], atol=1e-12, rtol=0
        ):
            raise RuntimeError("prediction CSV AUROC replay failed")
    verification = {
        "status": "passed",
        "identity": identity,
        "replayed_all_score_calibration_threshold_and_test_patches": True,
        "checkpoint_replay_max_absolute_error": maximum_error,
        "csv_auroc_recomputed": True,
        "all_epoch_patch_counts_verified": True,
        "test_images": len(labels),
        "methods": METHODS,
        "seconds": time.time() - started,
        "deep": deep_diagnostics,
        "drocc": drocc_diagnostics,
    }
    write_json(directory / "verification.json", verification)
    hashes = {
        p.name: sha256_file(p)
        for p in directory.iterdir()
        if p.is_file() and p.name != "COMPLETE.json"
    }
    write_json(directory / "COMPLETE.json", {"identity": identity, "artifacts": hashes})
    print(
        f"COMPLETE {category} seed={seed} seconds={time.time() - started:.1f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU smoke tests are not benchmark runs")
    torch.set_num_threads(8)
    threadpool_limits(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    source = {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")}
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
        "tf32": False,
        "precision": "float32 CNN/heads; float64 local PCA/graph eigensolver/kernel QP",
        "seed_protocol": cfg["seeds"],
    }
    write_json(out / "environment.json", environment)
    (out / "pip-freeze.txt").write_text(
        subprocess.check_output(
            [__import__("sys").executable, "-m", "pip", "freeze"], text=True
        )
    )
    print("Inspecting full original dataset and hashing every image/mask", flush=True)
    dataset = inspect_dataset(Path(args.data), cfg["categories"])
    write_json(out / "dataset_manifest.json", dataset)
    backbone = FrozenBackbone(cfg).to("cuda").eval()
    weights = Path(torch.hub.get_dir()) / "checkpoints" / "wide_resnet50_2-95faca4d.pth"
    if not weights.exists():
        raise RuntimeError("expected ImageNet V1 checkpoint path missing")
    identity = identity_hash(
        {
            "config": cfg,
            "source": source,
            "dataset_manifest_sha256": sha256_file(out / "dataset_manifest.json"),
            "backbone_sha256": sha256_file(weights),
        }
    )
    lock_path = out / "protocol_lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text())["identity"] != identity:
        raise RuntimeError(
            "source/config/data/backbone changed; use a fresh output directory"
        )
    write_json(
        lock_path,
        {
            "identity": identity,
            "config_sha256": sha256_file(args.config),
            "config": cfg,
            "source": source,
            "dataset_manifest_sha256": sha256_file(out / "dataset_manifest.json"),
            "backbone_sha256": sha256_file(weights),
        },
    )
    write_json(
        out / "backbone.json",
        {
            "name": cfg["backbone"],
            "weights": cfg["weights"],
            "checkpoint_sha256": sha256_file(weights),
            "checkpoint_filename": weights.name,
            "layers": cfg["layers"],
            "frozen": True,
            "eval": True,
        },
    )
    for category in cfg["categories"]:
        category_out = out / category
        category_out.mkdir(exist_ok=True)
        if all(
            (category_out / f"seed-{s}" / "COMPLETE.json").exists()
            for s in cfg["seeds"]
        ):
            # Verify all artifact hashes before honoring a resumed category.
            for seed in cfg["seeds"]:
                run_one(
                    None,
                    None,
                    cfg,
                    category,
                    seed,
                    category_out / f"seed-{seed}",
                    identity,
                    "cuda",
                )
            continue
        print(f"EXTRACT {category}", flush=True)
        features, paths = extract_category(
            backbone, Path(args.data), category, cfg, "cuda"
        )
        write_json(
            category_out / "feature_manifest.json",
            {
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "sha256": array_sha256(features),
                "paths": paths,
                "persisted": "RAM cache; regenerate from hashed images and backbone",
                "same_cache_for_all_methods_and_seeds": True,
            },
        )
        for seed in cfg["seeds"]:
            run_one(
                features,
                paths,
                cfg,
                category,
                seed,
                category_out / f"seed-{seed}",
                identity,
                "cuda",
            )
            gc.collect()
            torch.cuda.empty_cache()
        del features
        gc.collect()
    write_json(
        out / "ALL_RUNS_COMPLETE.json",
        {"identity": identity, "runs": len(cfg["categories"]) * len(cfg["seeds"])},
    )


if __name__ == "__main__":
    main()
