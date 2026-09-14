"""Real-image layer, descriptor, fixed-memory NN/image/map parity; not a benchmark."""

import json

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter

from .common import ROOT, seed_everything, write_json
from .patchcore import AuthorPatchCore, backbone_v1, datasets


def check(name, reference, actual, atol=1e-5, rtol=1e-5):
    a, b = np.asarray(reference), np.asarray(actual)
    error = np.abs(a.astype(float) - b.astype(float))
    passed = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    result = {
        "name": name,
        "shape": list(a.shape),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
    }
    assert passed, result
    return result


@torch.no_grad()
def main():
    seed_everything(0)
    adapter = AuthorPatchCore()
    train, test = datasets(ROOT / "data/mvtec_ad", "bottle")
    chosen = [
        train[0],
        next(test[i] for i in range(len(test)) if test[i]["is_anomaly"]),
    ]
    images = torch.stack([x["image"] for x in chosen]).cuda()
    reference_layers = adapter.model.forward_modules["feature_aggregator"](images)
    # Independent explicit CNN walk from the effective post-probe checkpoint.
    manual = backbone_v1().cuda().eval()
    manual.load_state_dict(adapter.backbone.state_dict())
    z = manual.maxpool(manual.relu(manual.bn1(manual.conv1(images))))
    z = manual.layer1(z)
    a = manual.layer2(z)
    b = manual.layer3(a)
    checks = [
        check("layer2", reference_layers["layer2"].cpu(), a.cpu()),
        check("layer3", reference_layers["layer3"].cpu(), b.cpu()),
    ]
    # Independent Unfold -> spatial alignment -> per-layer pooling -> aggregation.
    aligned = []
    for layer in (a, b):
        n, channels, height, width = layer.shape
        patches = F.unfold(layer, kernel_size=3, padding=1, stride=1)
        patches = patches.reshape(n, channels, 3, 3, height, width)
        if (height, width) != (28, 28):
            patches = F.interpolate(
                patches.reshape(-1, 1, height, width),
                size=(28, 28),
                mode="bilinear",
                align_corners=False,
            ).reshape(n, channels, 3, 3, 28, 28)
        patches = patches.permute(0, 4, 5, 1, 2, 3).reshape(n * 784, 1, channels * 9)
        aligned.append(F.adaptive_avg_pool1d(patches, 1024).squeeze(1))
    joined = torch.stack(aligned, dim=1).reshape(len(images) * 784, 1, 2048)
    independent = (
        F.adaptive_avg_pool1d(joined, 1024)
        .reshape(len(images), 784, 1024)
        .cpu()
        .numpy()
    )
    cached = adapter.descriptors(images)
    checks.append(check("descriptor", independent, cached))
    memory = cached[0, ::8].copy()
    adapter.load_memory(memory)
    source_scores, source_maps = adapter.model._predict(images)
    scores, patches, maps = adapter.score_descriptors(cached)
    checks.extend(
        [
            check("cached_image_scores", source_scores, scores, atol=1e-4),
            check("cached_score_maps", source_maps, maps, atol=1e-4),
        ]
    )
    # Independent direct pairwise squared-L2 check against GPU FAISS.
    expected = (
        (
            (
                torch.from_numpy(cached).double()[:, :, None]
                - torch.from_numpy(memory).double()[None, None]
            )
            ** 2
        )
        .sum(-1)
        .min(-1)
        .values.numpy()
    )
    checks.append(check("faiss_squared_L2", expected, patches, atol=2e-4, rtol=1e-4))
    maps_expected = F.interpolate(
        torch.tensor(patches).reshape(2, 1, 28, 28),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[:, 0].numpy()
    maps_expected = np.stack([gaussian_filter(m, sigma=4) for m in maps_expected])
    checks.append(check("bilinear_gaussian_maps", maps_expected, maps, atol=1e-4))
    by_one = np.concatenate([adapter.descriptors(images[i : i + 1]) for i in range(2)])
    checks.append(check("batch1_vs_batch2", cached, by_one, atol=1e-4, rtol=1e-4))
    # Cache serialization must preserve values exactly.
    cache = ROOT / "artifacts/reproduction/fixtures/patchcore-cache.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, cached)
    checks.append(check("cache_roundtrip", cached, np.load(cache), atol=0, rtol=0))
    write_json(
        ROOT / "artifacts/reproduction/fixtures/patchcore_parity.json",
        {
            "real_images": [
                str(x["image_path"]).removeprefix(str(ROOT / "data/mvtec_ad") + "/")
                for x in chosen
            ],
            "checks": checks,
            "passed": True,
            "raw_V1_state_sha256": adapter.before_probe,
            "post_author_dimension_probe_state_sha256": adapter.after_probe,
            "BN_changed_in_upstream_startup": adapter.before_probe
            != adapter.after_probe,
        },
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
