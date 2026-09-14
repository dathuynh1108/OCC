import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Original MVTec AD counts: training normal, test normal, test anomalous.
EXPECTED = {
    "bottle": (209, 20, 63),
    "cable": (224, 58, 92),
    "capsule": (219, 23, 109),
    "carpet": (280, 28, 89),
    "grid": (264, 21, 57),
    "hazelnut": (391, 40, 70),
    "leather": (245, 32, 92),
    "metal_nut": (220, 22, 93),
    "pill": (267, 26, 141),
    "screw": (320, 41, 119),
    "tile": (230, 33, 84),
    "toothbrush": (60, 12, 30),
    "transistor": (213, 60, 40),
    "wood": (247, 19, 60),
    "zipper": (240, 32, 119),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def inspect_dataset(root, categories):
    rows, counts = [], {}
    for category in categories:
        directory = Path(root) / category
        training = sorted(directory.glob("train/good/*.png"))
        test = sorted(directory.glob("test/*/*.png"))
        normal = [p for p in test if p.parent.name == "good"]
        actual = (len(training), len(normal), len(test) - len(normal))
        if actual != EXPECTED[category]:
            raise ValueError(
                f"{category}: expected original counts {EXPECTED[category]}, got {actual}"
            )
        if any(p.name != "good" for p in (directory / "train").iterdir()):
            raise ValueError("training data must be normal only")
        for p in training + test:
            with Image.open(p) as im:
                im.verify()
            anomalous = p.parent.name != "good"
            row = {
                "path": p.relative_to(root).as_posix(),
                "split": "train" if p in training else "test",
                "label": int(anomalous),
                "sha256": sha256_file(p),
            }
            if anomalous:
                mask = (
                    directory / "ground_truth" / p.parent.name / (p.stem + "_mask.png")
                )
                if not mask.exists():
                    raise ValueError(f"missing original anomaly mask: {mask}")
                with Image.open(mask) as im:
                    im.verify()
                row["mask_sha256"] = sha256_file(mask)
            rows.append(row)
        counts[category] = dict(
            zip(["train_normal", "test_normal", "test_anomaly"], actual)
        )
    return {"counts": counts, "images": rows}


def split_normal_indices(count, seed, cfg):
    order = np.random.default_rng(seed).permutation(count)
    fit_end = int(count * cfg["fit_fraction"])
    cal_end = fit_end + int(count * cfg["score_calibration_fraction"])
    return {
        "fit": np.sort(order[:fit_end]),
        "score_calibration": np.sort(order[fit_end:cal_end]),
        "threshold_calibration": np.sort(order[cal_end:]),
    }


class Images(Dataset):
    def __init__(self, paths, cfg):
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    cfg["resize"], interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(cfg["crop"]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


class FrozenBackbone(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg["backbone"] != "wide_resnet50_2" or cfg["weights"] != "IMAGENET1K_V1":
            raise ValueError(
                "this frozen protocol requires Wide ResNet-50-2 IMAGENET1K_V1"
            )
        self.net = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        self.net.requires_grad_(False).eval()
        self.pool = cfg["local_pool"]

    def forward(self, x):
        net = self.net
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = net.layer1(x)
        layer2 = net.layer2(x)
        layer3 = net.layer3(layer2)
        a = F.avg_pool2d(layer2, self.pool, stride=1, padding=self.pool // 2)
        b = F.avg_pool2d(layer3, self.pool, stride=1, padding=self.pool // 2)
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([a, b], dim=1).flatten(2).transpose(1, 2).contiguous()


@torch.inference_mode()
def extract_category(backbone, root, category, cfg, device):
    directory = Path(root) / category
    paths = sorted(directory.glob("train/good/*.png")) + sorted(
        directory.glob("test/*/*.png")
    )
    loader = DataLoader(
        Images(paths, cfg),
        batch_size=cfg["feature_batch"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    pieces = []
    for batch in loader:
        feature = backbone(batch.to(device))
        if feature.shape[1:] != (784, 1536) or not torch.isfinite(feature).all():
            raise RuntimeError("feature shape or finiteness failed")
        pieces.append(feature.cpu().numpy())
    features = np.concatenate(pieces)
    paths = [p.relative_to(root).as_posix() for p in paths]
    return features, paths
