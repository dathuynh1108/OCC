"""Pinned PatchCore adapter, retaining author initialization, extraction and scoring."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import models

from .common import ROOT, load_plan, sha256, verify_source

sys.path.insert(0, str(ROOT / "vendor/patchcore/src"))
from patchcore.common import FaissNN
from patchcore.datasets.mvtec import DatasetSplit, MVTecDataset
from patchcore.patchcore import PatchCore
from patchcore.sampler import ApproximateGreedyCoresetSampler


def state_sha256(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def backbone_v1():
    path = ROOT / "artifacts/backbone/wide_resnet50_2-95faca4d.pth"
    lock = json.loads((ROOT / "source_lock.json").read_text())["weights"]
    assert sha256(path) == lock["sha256"]
    backbone = models.wide_resnet50_2(weights=None)
    backbone.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    backbone.name, backbone.seed = "wideresnet50", None
    return backbone


class RecordedSampler(ApproximateGreedyCoresetSampler):
    def _compute_greedy_coreset_indices(self, features):
        indices = super()._compute_greedy_coreset_indices(features)
        self.selected_indices = indices.copy()
        return indices


class AuthorPatchCore:
    def __init__(self, gpu_nn=True):
        verify_source("patchcore")
        cfg = load_plan()["native"]["patchcore"]
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.backbone = backbone_v1()
        self.before_probe = state_sha256(self.backbone.state_dict())
        self.sampler = RecordedSampler(
            cfg["coreset_ratio"],
            self.device,
            number_of_starting_points=cfg["starting_points"],
            dimension_to_project_features_to=cfg["projection_dim"],
        )
        self.model = PatchCore(self.device)
        # Intentionally retain upstream's train-mode all-ones dimension probe.
        self.model.load(
            self.backbone,
            cfg["layers"],
            self.device,
            (3, cfg["crop"], cfg["crop"]),
            cfg["pretrain_dim"],
            cfg["target_dim"],
            patchsize=cfg["patchsize"],
            patchstride=cfg["patchstride"],
            anomaly_score_num_nn=cfg["num_nn"],
            featuresampler=self.sampler,
            nn_method=FaissNN(gpu_nn, 4),
        )
        self.after_probe = state_sha256(self.backbone.state_dict())
        self.backbone.requires_grad_(False).eval()
        self.model.forward_modules.eval()

    @torch.no_grad()
    def descriptors(self, images):
        array = np.asarray(self.model._embed(images.cuda()), dtype=np.float32)
        assert array.shape == (len(images) * 784, 1024)
        assert np.isfinite(array).all()
        return array.reshape(len(images), 784, 1024)

    @torch.no_grad()
    def fit_memory(self, features):
        self.memory = np.asarray(
            self.sampler.run(features.reshape(-1, 1024)), dtype=np.float32
        )
        self.model.anomaly_scorer.fit([self.memory])
        return self.memory, self.sampler.selected_indices

    def load_memory(self, memory):
        self.memory = np.asarray(memory, dtype=np.float32)
        self.model.anomaly_scorer.fit([self.memory])

    def score_descriptors(self, descriptors):
        n = len(descriptors)
        flat = descriptors.reshape(-1, 1024)
        # Batch by complete image to match source _predict and FAISS arithmetic.
        patches = np.stack(
            [
                self.model.anomaly_scorer.predict([flat[i * 784 : (i + 1) * 784]])[0]
                for i in range(n)
            ]
        )
        maps = self.model.anomaly_segmentor.convert_to_segmentation(
            patches.reshape(n, 28, 28)
        )
        return patches.max(1), patches, np.asarray(maps)


def datasets(root, category):
    return [
        MVTecDataset(str(root), category, resize=256, imagesize=224, split=split)
        for split in (DatasetSplit.TRAIN, DatasetSplit.TEST)
    ]


def extract_category(adapter, root, category):
    train, test = datasets(root, category)
    descriptors, paths, masks, labels = [], [], [], []
    for source in (train, test):
        # Same DataLoader creation/iteration RNG plumbing as upstream.
        loader = torch.utils.data.DataLoader(
            source, batch_size=1, shuffle=False, num_workers=0
        )
        for row in loader:
            descriptors.append(adapter.descriptors(row["image"]))
            paths.append(Path(row["image_path"][0]).relative_to(root).as_posix())
            if source is test:
                # Author metrics casts interpolated mask values to int, not >0.5.
                masks.append(row["mask"][0, 0].numpy().astype(np.uint8))
                labels.append(int(row["is_anomaly"][0]))
    return (
        np.concatenate(descriptors),
        paths,
        np.stack(masks),
        np.array(labels),
        len(train),
    )
