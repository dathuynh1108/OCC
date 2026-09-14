"""Independent saved-score verification, optionally regenerating all CNN features."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
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
from .heads import DeepHead, DROCCHead
from .models import BubbleModel, KernelSVDD
from .report import collect_verified
from .run import raw_scores, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default="data/mvtec_ad")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    cfg, metrics, _, checks, _, _ = collect_verified(out)
    errors = []
    if args.replay:
        if not torch.cuda.is_available():
            raise RuntimeError("full feature/checkpoint replay requires CUDA")
        torch.set_num_threads(8)
        threadpool_limits(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        dataset = inspect_dataset(Path(args.data), cfg["categories"])
        if dataset != json.loads((out / "dataset_manifest.json").read_text()):
            raise RuntimeError("replay input dataset differs from frozen manifest")
        backbone = FrozenBackbone(cfg).cuda().eval()
        metadata = json.loads((out / "backbone.json").read_text())
        weights = (
            Path(torch.hub.get_dir()) / "checkpoints" / metadata["checkpoint_filename"]
        )
        if sha256_file(weights) != metadata["checkpoint_sha256"]:
            raise RuntimeError("replay backbone checkpoint changed")
        for category in cfg["categories"]:
            features, paths = extract_category(
                backbone, Path(args.data), category, cfg, "cuda"
            )
            manifest = json.loads(
                (out / category / "feature_manifest.json").read_text()
            )
            if (
                paths != manifest["paths"]
                or array_sha256(features) != manifest["sha256"]
            ):
                raise RuntimeError(f"CNN feature cache replay mismatch: {category}")
            for seed in cfg["seeds"]:
                directory = out / category / f"seed-{seed}"
                saved = torch.load(
                    directory / "geometry.pt", map_location="cpu", weights_only=False
                )
                bubble = BubbleModel(cfg).restore(saved["bubble"], "cuda")
                memory = saved["memory"].cuda()
                kernel = KernelSVDD().restore(saved["kernel"], "cuda")
                deep = DeepHead(1536, cfg["head_hidden"], cfg["head_output"]).cuda()
                deep.load_state_dict(
                    torch.load(
                        directory / "deep_svdd.pt",
                        map_location="cuda",
                        weights_only=False,
                    )["model"]
                )
                deep.eval()
                drocc = DROCCHead(1536, cfg["head_hidden"]).cuda()
                drocc.load_state_dict(
                    torch.load(
                        directory / "drocc.pt", map_location="cuda", weights_only=False
                    )["model"]
                )
                drocc.eval()
                split = split_normal_indices(EXPECTED[category][0], seed, cfg)
                original = np.load(directory / "scores.npz")
                for name, ids in [
                    ("calibration", split["score_calibration"]),
                    ("threshold", split["threshold_calibration"]),
                    ("test", np.arange(EXPECTED[category][0], len(features))),
                ]:
                    replay = raw_scores(
                        features[ids], bubble, memory, kernel, deep, drocc, cfg, "cuda"
                    )
                    expected = original[f"{name}_raw"]
                    error = float(np.abs(replay - expected).max())
                    if not np.allclose(replay, expected, rtol=1e-6, atol=1e-7):
                        raise RuntimeError(
                            f"checkpoint replay failed {category}/{seed}/{name}: {error}"
                        )
                    errors.append(
                        {
                            "category": category,
                            "seed": seed,
                            "split": name,
                            "max_absolute_error": error,
                        }
                    )
                original.close()
                del bubble, memory, kernel, deep, drocc
                torch.cuda.empty_cache()
                print(f"Replayed {category} seed {seed}", flush=True)
            del features
    result = {
        "status": "passed",
        "runs": len(checks),
        "metric_rows": len(metrics),
        "regenerated_full_CNN_cache": args.replay,
        "checkpoint_replay": errors,
    }
    write_json(out / "reverification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
