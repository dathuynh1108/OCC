"""Isolated batch-one latency and stored-state accounting; run after training stops."""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from .common import ROOT, environment, load_plan, seed_everything, write_json
from .patchcore import AuthorPatchCore, datasets
from nbdbench.heads import DeepHead, DROCCHead
from nbdbench.models import BubbleModel, KernelSVDD
from nbdbench.math import squared_distances, empirical_tail, image_score


def bytes_of(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(bytes_of(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(bytes_of(v) for v in value)
    return 0


@torch.no_grad()
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    threadpool_limits(4)
    seed_everything(0)
    plan = load_plan()
    root = ROOT / plan["controlled"]["output"] / "bottle/seed-0"
    base = root / "base"
    import json

    cfg = json.loads((root / "protocol.json").read_text())["config"]
    saved = torch.load(base / "geometry.pt", weights_only=False, map_location="cpu")
    bubble = BubbleModel(cfg).restore(saved["bubble"], "cuda")
    memory = saved["memory"].cuda()
    kernel = KernelSVDD().restore(saved["kernel"], "cuda")
    deep = DeepHead(1024, cfg["head_hidden"], cfg["head_output"]).cuda().eval()
    deep.load_state_dict(
        torch.load(base / "deep_svdd.pt", weights_only=False, map_location="cuda")[
            "model"
        ]
    )
    drocc = DROCCHead(1024, cfg["head_hidden"]).cuda().eval()
    drocc.load_state_dict(
        torch.load(base / "drocc.pt", weights_only=False, map_location="cuda")["model"]
    )
    adapter = AuthorPatchCore()
    adapter.load_memory(
        torch.load(
            root / "patchcore_same_fit/memory.pt",
            weights_only=False,
            map_location="cpu",
        )["memory"]
    )
    _, test = datasets(ROOT / "data/mvtec_ad", "bottle")
    image = test[0]["image"][None]
    features = adapter.descriptors(image)
    cal = np.load(base / "scores.npz")["calibration_raw"][..., 5:9]

    def core(fn):
        chunks = []
        x = features.reshape(-1, 1024)
        for offset in range(0, len(x), cfg["score_batch"]):
            chunks.append(
                fn(torch.from_numpy(x[offset : offset + cfg["score_batch"]]).cuda())
                .cpu()
                .numpy()
            )
        return np.concatenate(chunks)

    def ordinary(fn):
        return float(
            image_score(core(fn).reshape(1, 784), cfg["image_top_fraction"])[0]
        )

    def nbd():
        raw = core(bubble.components)
        parts = np.stack(
            [empirical_tail(raw[:, i], cal[..., i].ravel()) for i in range(4)], axis=-1
        )
        scores = (
            parts[:, 0] + parts[:, 1] + parts[:, 2] + cfg["frame_weight"] * parts[:, 3]
        )
        return float(image_score(scores.reshape(1, 784), cfg["image_top_fraction"])[0])

    def pc():
        patches = adapter.model.anomaly_scorer.predict([features.reshape(-1, 1024)])[
            0
        ].reshape(1, 784)
        return float(image_score(patches, cfg["image_top_fraction"])[0])

    methods = {
        "shared_author_WR50_descriptor": lambda: adapter.descriptors(image),
        "PatchScore_same_centers": lambda: ordinary(
            lambda x: squared_distances(x, bubble.centers).min(1).values
        ),
        "PatchScore_byte_budget": lambda: ordinary(
            lambda x: squared_distances(x, memory).min(1).values
        ),
        "RBF_SVDD": lambda: ordinary(kernel.score),
        "DeepSVDD_head": lambda: ordinary(deep.score),
        "DROCC_head": lambda: ordinary(drocc.score),
        "NBD": nbd,
        "PatchCore_same_fit_top1pct": pc,
    }
    memory_bytes = {
        "shared_author_WR50_descriptor": bytes_of(adapter.backbone.state_dict()),
        "PatchScore_same_centers": bytes_of(bubble.centers),
        "PatchScore_byte_budget": bytes_of(memory),
        "RBF_SVDD": bytes_of(saved["kernel"]),
        "DeepSVDD_head": bytes_of(deep.state_dict()),
        "DROCC_head": bytes_of(drocc.state_dict()),
        "NBD": bytes_of(saved["bubble"]),
        "PatchCore_same_fit_top1pct": adapter.memory.nbytes,
    }
    prediction = pd.concat(
        [
            pd.read_csv(base / "predictions.csv"),
            pd.read_csv(root / "patchcore_same_fit/predictions.csv"),
        ]
    )
    sample_id = str(test[0]["image_path"]).split("/data/mvtec_ad/")[-1]
    validation = []
    for name, fn in methods.items():
        if name == "shared_author_WR50_descriptor":
            continue
        expected = float(
            prediction[(prediction.method == name) & (prediction.path == sample_id)]
            .iloc[0]
            .score
        )
        actual = fn()
        error = abs(expected - actual)
        assert error <= 1e-6, (name, expected, actual, error)
        validation.append({"method": name, "max_abs_error": error})
    if args.validate_only:
        write_json(
            ROOT / "artifacts/reproduction/fixtures/inference_replay.json",
            {"passed": True, "checks": validation},
        )
        return
    results = []
    for name, fn in methods.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        values = []
        for _ in range(30):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            values.append((time.perf_counter() - tick) * 1000)
        results.append(
            {
                "method": name,
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "samples": 30,
                "state_array_bytes": memory_bytes[name],
                "separate_NBD_component_calibration_bytes": cal.nbytes
                if name == "NBD"
                else 0,
            }
        )
    out = ROOT / "results/reproduction-2026-09-14"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "inference_bottle_seed0.csv", index=False)
    write_json(
        out / "inference_scope.json",
        {
            "runtime": environment(),
            "category": "bottle",
            "seed": 0,
            "image_id": str(test[0]["image_path"]).split("/data/mvtec_ad/")[-1],
            "batch_images": 1,
            "patches": 784,
            "descriptor_dim": 1024,
            "warmup": 3,
            "repetitions": 30,
            "timing": "wall clock with CUDA synchronization; CPU descriptor input, includes transfers and image pooling; NBD includes source ECDF sorting",
            "shared_descriptor_excludes": "image disk I/O and PIL preprocessing",
            "method_timings_exclude": "shared CNN; checkpoint loading; threshold comparison; pixel maps",
            "state_bytes_exclude": "Python overhead, allocator, optimizer; NBD calibration reported separately",
            "other_training_processes": "must be stopped before this command",
        },
    )


if __name__ == "__main__":
    main()
