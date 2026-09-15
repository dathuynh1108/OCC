"""Prepare hash-verified MVTec images for the authors' fixed 32-pixel CNNs."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms

from .common import ROOT, sha256, write_json

CACHE = ROOT / "artifacts/reproduction/mvtec-images32"
MANIFEST = ROOT / "results/lean-nbd-5070-20260915/dataset_manifest.json"
RECIPE = {
    "convert": "RGB",
    "resize_short_side": 256,
    "center_crop": 224,
    "cnn_resize": [32, 32],
    "interpolation": "PIL bilinear",
    "cache_dtype": "uint8",
    "scope": "MVTec comparison input adapter; not a paper MVTec recipe",
}


def prepare(cache=CACHE):
    manifest = json.loads(MANIFEST.read_text())
    lookup = {r["path"]: r for r in manifest["images"]}
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.Resize((32, 32)),
    ])
    cache.mkdir(parents=True, exist_ok=True)

    def decode(path):
        source = ROOT / "data/mvtec_ad" / path
        if sha256(source) != lookup[path]["sha256"]:
            raise ValueError(f"MVTec image differs from saved benchmark: {path}")
        with Image.open(source) as image:
            return np.asarray(transform(image.convert("RGB"))).transpose(2, 0, 1)

    records = []
    for category in sorted(manifest["counts"]):
        split_path = (
            ROOT / "results/native/patchcore/PatchCore_author_code_WR50_10pct"
            / category / "seed-0/splits.json"
        )
        splits = json.loads(split_path.read_text())
        paths = splits["train"] + splits["test"]
        assert not set(splits["train"]) & set(splits["test"])
        assert all(lookup[p]["split"] == "train" and lookup[p]["label"] == 0 for p in splits["train"])
        with ThreadPoolExecutor(max_workers=8) as pool:
            pixels = np.stack(list(pool.map(decode, paths)))
        output = cache / f"{category}.npz"
        np.savez_compressed(
            output, pixels=pixels, paths=np.array(paths),
            labels=np.array([lookup[p]["label"] for p in paths]),
            train_count=len(splits["train"]),
        )
        record = {
            "category": category, "train_count": len(splits["train"]),
            "test_count": len(splits["test"]), "sha256": sha256(output),
            "shape": list(pixels.shape), "split_sha256": sha256(split_path),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    assert sum(r["train_count"] for r in records) == 3629
    assert sum(r["test_count"] for r in records) == 1725
    write_json(cache / "manifest.json", {
        "recipe": RECIPE, "source_manifest_sha256": sha256(MANIFEST),
        "all_original_image_hashes_verified": True, "categories": records,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CACHE)
    prepare(parser.parse_args().output)
