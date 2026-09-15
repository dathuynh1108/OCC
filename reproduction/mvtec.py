"""Native PatchCore by default; the controlled NBD experiment is an explicit track."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from .common import (
    ROOT,
    load_plan,
    plan_path,
    rng_state,
    save_checkpoint,
    seed_everything,
    sha256,
    write_json,
)
from nbdbench.data import EXPECTED, array_sha256, inspect_dataset, split_normal_indices
from nbdbench.math import image_score, normal_threshold
from nbdbench.run import metric_row, run_one


def selection_rng_from_source(adapter, data_root, category):
    from .patchcore import datasets

    # Source constructs model, then iterates the unshuffled train loader before
    # drawing projection weights. Creating its iterator consumes the RNG base seed;
    # deterministic image transforms/CNN extraction consume no further random draws.
    train, _ = datasets(data_root, category)
    iterator = iter(
        torch.utils.data.DataLoader(train, batch_size=1, shuffle=False, num_workers=0)
    )
    del iterator
    return rng_state()


def validate_historical_splits(paths, count, cfg, category, seed):
    split = split_normal_indices(count, seed, cfg)
    actual = {k: [paths[i] for i in ids] for k, ids in split.items()}
    actual["test"] = paths[count:]
    old = (
        ROOT / f"results/mvtec-full-v2/run_evidence/{category}/seed-{seed}/splits.json"
    )
    expected = json.loads(old.read_text())
    assert actual == expected, f"v2 split drift: {category}/{seed}"
    return split, sha256(old)


def save_predictions(out, paths, labels, scores):
    frame = pd.DataFrame({"sample_id": paths, "anomaly_label": labels, "score": scores})
    frame.to_csv(out / "predictions.csv", index=False)
    loaded = pd.read_csv(out / "predictions.csv")
    return {
        "auroc": roc_auc_score(loaded.anomaly_label, loaded.score),
        "average_precision": average_precision_score(
            loaded.anomaly_label, loaded.score
        ),
    }


def native(
    adapter,
    features,
    paths,
    masks,
    labels,
    count,
    category,
    seed,
    out,
    feature_hash,
    smoke,
):
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result["feature_sha256"] == feature_hash and result["smoke"] == smoke
        assert sha256(out / "memory.pt") == result["checkpoint_sha256"]
        assert sha256(out / "predictions.csv") == result["predictions_sha256"]
        return
    started = time.perf_counter()
    checkpoint = out / "memory.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert state["feature_sha256"] == feature_hash
        adapter.load_memory(state["memory"])
    else:
        memory, indices = adapter.fit_memory(features[:count])
        save_checkpoint(
            checkpoint,
            {
                "memory": memory,
                "selected_indices": indices,
                "rng_after_selection": rng_state(),
                "feature_sha256": feature_hash,
                "raw_backbone_state_sha256": adapter.before_probe,
                "effective_backbone_state_sha256": adapter.after_probe,
                "seed": seed,
                "config": adapter.cfg,
            },
        )
    scores, patch_scores, maps = adapter.score_descriptors(features[count:])
    metrics = save_predictions(out, paths[count:], labels, scores)
    pixels = masks.ravel().astype(int)
    metrics["pixel_auroc"] = roc_auc_score(pixels, maps.ravel())
    np.savez_compressed(
        out / "score_maps.npz",
        patch_scores=patch_scores,
        maps=maps,
        masks=masks,
        labels=labels,
    )
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    adapter.load_memory(saved["memory"])
    replay_scores, replay_patches, replay_maps = adapter.score_descriptors(
        features[count:]
    )
    errors = {
        "images": float(np.max(np.abs(scores - replay_scores))),
        "patches": float(np.max(np.abs(patch_scores - replay_patches))),
        "maps": float(np.max(np.abs(maps - replay_maps))),
    }
    assert all(value <= 1e-6 for value in errors.values()), errors
    write_json(out / "splits.json", {"train": paths[:count], "test": paths[count:]})
    write_json(
        result_path,
        {
            **metrics,
            "target": "PatchCore_author_code_WR50_10pct",
            "category": category,
            "seed": seed,
            "train_normal_count": count,
            "test_count": len(labels),
            "memory_descriptors": len(adapter.memory),
            "memory_bytes": adapter.memory.nbytes,
            "effective_backbone_state_sha256": adapter.after_probe,
            "feature_sha256": feature_hash,
            "checkpoint_sha256": sha256(checkpoint),
            "predictions_sha256": sha256(out / "predictions.csv"),
            "replay_errors": errors,
            "seconds": time.perf_counter() - started,
            "smoke": smoke,
            "image_score": "max_squared_L2_1NN_no_Eq7_reweighting",
            "AUPRO": "not_evaluated",
            "mask_rule": "upstream_resize_crop_then_astype_int",
            "complete": True,
        },
    )
    print("NATIVE COMPLETE", category, seed, metrics, flush=True)


def controlled_patchcore(
    adapter,
    features,
    paths,
    count,
    category,
    seed,
    cfg,
    split,
    out,
    feature_hash,
    smoke,
):
    """Same FIT as NBD; author sampled memory; same top1%mean and image calibration."""
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        result = json.loads((out / "result.json").read_text())
        assert result["feature_sha256"] == feature_hash and result["smoke"] == smoke
        return
    checkpoint = out / "memory.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert saved["feature_sha256"] == feature_hash
        adapter.load_memory(saved["memory"])
    else:
        memory, indices = adapter.fit_memory(features[split["fit"]])
        save_checkpoint(
            checkpoint,
            {
                "memory": memory,
                "selected_fit_patch_indices": indices,
                "fit_image_ids": split["fit"],
                "rng_after": rng_state(),
                "feature_sha256": feature_hash,
                "seed": seed,
            },
        )
    _, threshold_patches, _ = adapter.score_descriptors(
        features[split["threshold_calibration"]]
    )
    _, test_patches, _ = adapter.score_descriptors(features[count:])
    threshold_scores = image_score(threshold_patches, cfg["image_top_fraction"])
    scores = image_score(test_patches, cfg["image_top_fraction"])
    threshold = normal_threshold(threshold_scores, cfg["alpha"])
    labels = np.array([int(Path(p).parent.name != "good") for p in paths[count:]])
    row = metric_row(
        category,
        seed,
        "PatchCore_same_fit_top1pct",
        labels,
        scores,
        threshold,
        threshold_scores,
    )
    pd.DataFrame([row]).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "category": category,
            "seed": seed,
            "method": row["method"],
            "path": paths[count:],
            "label": labels,
            "score": scores,
            "threshold": threshold,
        }
    ).to_csv(out / "predictions.csv", index=False)
    np.savez_compressed(
        out / "scores.npz",
        threshold_scores=threshold_scores,
        scores=scores,
        labels=labels,
    )
    adapter.load_memory(
        torch.load(checkpoint, weights_only=False, map_location="cpu")["memory"]
    )
    _, replay, _ = adapter.score_descriptors(features[count:])
    error = float(np.max(np.abs(test_patches - replay)))
    assert error <= 1e-6
    write_json(
        out / "result.json",
        {
            "complete": True,
            "smoke": smoke,
            "feature_sha256": feature_hash,
            "replay_max_abs_error": error,
            "category": category,
            "seed": seed,
            "memory_bytes": adapter.memory.nbytes,
            "memory_descriptors": len(adapter.memory),
            "checkpoint_sha256": sha256(checkpoint),
            "uses_component_calibration": False,
            "protocol": "controlled_adaptation_same_FIT_top1pct",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(EXPECTED))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--track", choices=("native", "controlled"), default="native",
        help="Native author-code reproduction or the separate NBD comparison.",
    )
    args = parser.parse_args()
    from .patchcore import AuthorPatchCore, extract_category

    plan = load_plan()
    parity = json.loads(
        (ROOT / "artifacts/reproduction/fixtures/patchcore_parity.json").read_text()
    )
    assert parity["passed"]
    if not args.smoke and args.track == "native":
        assert json.loads(
            (
                ROOT
                / "artifacts/reproduction/smoke/mvtec/native/toothbrush/seed-0/result.json"
            ).read_text()
        )["complete"]
    if not args.smoke and args.track == "controlled":
        assert (
            ROOT
            / "artifacts/reproduction/smoke/mvtec/controlled/toothbrush/seed-0/base/COMPLETE.json"
        ).exists()
    categories = (
        ["toothbrush"]
        if args.smoke
        else [args.category]
        if args.category
        else plan["native"]["patchcore"]["categories"]
    )
    seeds = [0] if args.smoke else plan["native"]["patchcore"]["seeds"]
    if args.track == "controlled":
        cfg = json.loads((ROOT / "configs/full.json").read_text())
        cfg.update(
            protocol=plan["controlled"]
            .get("output", "results/mvtec-author-encoder-controlled-v3")
            .split("/")[-1],
            feature_batch=1,
        )
        cfg.update(plan["controlled"].get("epoch_overrides", {}))
        if args.smoke:
            cfg.update(ae_epochs=1, deep_epochs=1, drocc_epochs=1)
    data_root = ROOT / "data/mvtec_ad"
    if args.track == "native":
        result_root = ROOT / (
            "artifacts/reproduction/smoke/mvtec/native"
            if args.smoke
            else "results/native/patchcore/PatchCore_author_code_WR50_10pct"
        )
    else:
        result_root = ROOT / (
            "artifacts/reproduction/smoke/mvtec/controlled"
            if args.smoke
            else plan["controlled"].get(
                "output", "results/mvtec-author-encoder-controlled-v3"
            )
        )
    result_root.mkdir(parents=True, exist_ok=True)
    threadpool_limits(4)
    for category in categories:
        seed_everything(0)
        torch.cuda.reset_peak_memory_stats()
        data_manifest = inspect_dataset(data_root, [category])
        adapter = AuthorPatchCore()
        tick = time.perf_counter()
        features, paths, masks, labels, count = extract_category(
            adapter, data_root, category
        )
        feature_hash = array_sha256(features)
        info = {
            "dataset": data_manifest,
            "feature_shape": list(features.shape),
            "feature_sha256": feature_hash,
            "raw_backbone_state_sha256": adapter.before_probe,
            "effective_backbone_state_sha256": adapter.after_probe,
            "extract_seconds": time.perf_counter() - tick,
            "extract_peak_GPU_bytes": torch.cuda.max_memory_allocated(),
            "run_plan_sha256": sha256(plan_path()),
            "raw_v2_features_reused": False,
        }
        write_json(result_root / f"{category}_features.json", info)
        for seed in seeds:
            del adapter
            gc.collect()
            torch.cuda.empty_cache()
            seed_everything(seed)
            adapter = AuthorPatchCore()
            assert adapter.after_probe == info["effective_backbone_state_sha256"]
            if args.track == "native":
                selection_rng_from_source(adapter, data_root, category)
                native(
                    adapter, features, paths, masks, labels, count, category, seed,
                    result_root / category / f"seed-{seed}", feature_hash, args.smoke,
                )
                continue
            split, old_hash = validate_historical_splits(
                paths, count, cfg, category, seed
            )
            out = result_root / category / f"seed-{seed}"
            out.mkdir(parents=True, exist_ok=True)
            identity = hashlib.sha256(
                json.dumps(
                    {
                        "feature_hash": feature_hash,
                        "split_hash": old_hash,
                        "cfg": cfg,
                        "base_occ": "91223d63adc3289786bbdeff16d5feeb59aeb3d7",
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            write_json(
                out / "protocol.json",
                {
                    "identity": identity,
                    "config": cfg,
                    "historical_split_sha256": old_hash,
                    "feature_sha256": feature_hash,
                    "smoke": args.smoke,
                },
            )
            # Reuse unchanged v2 numerical routines; feature dimension comes from x.
            # Names are explicitly adapted when producing the common summary table.
            run_one(
                features, paths, cfg, category, seed, out / "base", identity, "cuda"
            )
            assert json.loads((out / "base/splits.json").read_text()) == json.loads(
                (
                    ROOT
                    / f"results/mvtec-full-v2/run_evidence/{category}/seed-{seed}/splits.json"
                ).read_text()
            )
            seed_everything(seed)
            controlled_patchcore(
                adapter,
                features,
                paths,
                count,
                category,
                seed,
                cfg,
                split,
                out / "patchcore_same_fit",
                feature_hash,
                args.smoke,
            )
        del adapter, features, masks
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
