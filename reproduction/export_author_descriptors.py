"""Export hash-bound author-encoder descriptors without fitting any detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision

from nbdbench.data import array_sha256, inspect_dataset
from reproduction.common import ROOT, sha256, write_json


def selection_categories(path: Path) -> set[str]:
    value = json.loads(path.read_text())
    sentinels = value.get("sentinel_a"), value.get("sentinel_b")
    categories = {str(item["category"]) for item in sentinels if isinstance(item, dict) and "category" in item}
    if not categories:
        raise ValueError(f"{path}: no sentinel categories")
    return categories


def export_category(category: str, data_root: Path, output: Path, selection_plan: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("author descriptor export requires a CUDA runtime; it will not silently use a different CPU path")
    selected = selection_categories(selection_plan)
    if category not in selected:
        raise ValueError(f"{category} is not in the locked selection plan {selection_plan}")
    manifest_path = ROOT / "results/mvtec-author-encoder-budgeted-v3" / f"{category}_features.json"
    manifest = json.loads(manifest_path.read_text())
    target = output / category
    archive = target / "descriptors.npz"
    metadata = target / "metadata.json"
    if archive.exists() or metadata.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    if not (data_root / category).is_dir():
        raise FileNotFoundError(f"missing extracted MVTec category at {data_root / category}")
    # Match reproduction.mvtec.main: inspect original data, seed before the
    # author adapter's train-mode dimension probe, then run frozen inference.
    from reproduction.patchcore import AuthorPatchCore, extract_category

    dataset = inspect_dataset(data_root, [category])
    from reproduction.common import seed_everything

    seed_everything(0)
    adapter = AuthorPatchCore()
    descriptors, paths, _masks, _labels, _train_count = extract_category(adapter, data_root, category)
    content_hash = array_sha256(descriptors)
    expected = str(manifest["feature_sha256"])
    if content_hash != expected:
        raise RuntimeError(
            f"{category}: extracted descriptor hash {content_hash} differs from recorded {expected}; no archive written"
        )
    target.mkdir(parents=True)
    np.savez_compressed(archive, all_descriptors=descriptors, paths=np.asarray(paths))
    write_json(
        metadata,
        {
            "category": category,
            "selection_plan": str(selection_plan),
            "selection_plan_sha256": sha256(selection_plan),
            "dataset": dataset,
            "descriptor_shape": list(descriptors.shape),
            "descriptor_dtype": str(descriptors.dtype),
            "descriptor_content_sha256": content_hash,
            "historical_feature_sha256": expected,
            "archive_sha256": sha256(archive),
            "ordered_path_count": len(paths),
            "weights_sha256": sha256(ROOT / "artifacts/backbone/wide_resnet50_2-95faca4d.pth"),
            "raw_backbone_state_sha256": adapter.before_probe,
            "effective_backbone_state_sha256": adapter.after_probe,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "no_fit_or_detector_training": True,
            "labels_or_masks_used_for_selection": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", action="append", required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mvtec_ad")
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = args.selection_plan.resolve()
    if not plan.is_file():
        raise FileNotFoundError(plan)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to reuse output root {output}")
    output.mkdir(parents=True)
    for category in args.category:
        export_category(category, args.data_root.resolve(), output, plan)


if __name__ == "__main__":
    main()
