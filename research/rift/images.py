"""LOCO image plumbing. Learned roles are visual/spatial proxies, not part labels."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import MiniBatchKMeans

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
CATEGORIES = ('breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag', 'splicing_connectors')


@dataclass(frozen=True)
class Sample:
    id: str
    path: Path
    split: str
    kind: str
    label: int


@dataclass
class ImageConfig:
    backbone: str = 'resnet18'
    image_size: int = 256
    grid_size: int = 14
    feature_dim: int = 64
    roles: int = 8
    position_weight: float = .25
    temperature: float = .15
    kmeans_samples: int = 20000
    batch_size: int = 8
    times: tuple = (.5, 2., 8., 32.)
    eta: float = .5
    pca_dim: int = 0
    weights_path: str | None = None

    def validate(self):
        if self.backbone not in ('rgb', 'resnet18', 'wide_resnet50_2'):
            raise ValueError('backbone must be rgb, resnet18, or wide_resnet50_2')
        if not 2 <= self.grid_size <= 32 or self.image_size < 32:
            raise ValueError('grid_size must be 2..32 and image_size >=32')
        if min(self.feature_dim, self.roles, self.batch_size, self.kmeans_samples) < 1:
            raise ValueError('dimensions/sample limits/batch size must be positive')
        if self.position_weight < 0 or self.temperature <= 0 or self.pca_dim < 0:
            raise ValueError('invalid role/scoring settings')
        if not self.times or min(self.times) <= 0 or not 0 < self.eta <= 1:
            raise ValueError('invalid intervention settings')
        return self


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_loco(root, category, seed=0):
    root = Path(root).expanduser().resolve()
    folder = root if root.name == category else root / category
    if not (folder/'train/good').is_dir():
        raise FileNotFoundError(f'Expected {folder}/train/good; point --root to the LOCO root')
    samples = []
    for split, dirname in [('fit', 'train'), ('cal', 'validation'), ('test', 'test')]:
        base = folder / dirname
        if not base.exists():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                continue
            kind = path.relative_to(base).parts[0]
            allowed = ('good',) if split != 'test' else ('good', 'logical_anomalies', 'structural_anomalies')
            if kind not in allowed:
                raise ValueError(f'Unexpected subset {path}; expected {allowed}')
            samples.append(Sample(path.relative_to(folder).as_posix(), path, split, kind, int(kind != 'good')))
    fit = [s for s in samples if s.split == 'fit']
    if len(fit) < 4:
        raise ValueError('need at least four normal training images')
    if not any(s.split == 'cal' for s in samples):
        holdout = max(2, math.ceil(.2 * len(fit)))
        selected = set(np.random.default_rng(seed).choice(len(fit), holdout, replace=False).tolist())
        cal_ids = {fit[i].id for i in selected}
        samples = [Sample(s.id, s.path, 'cal' if s.id in cal_ids else s.split, s.kind, s.label) for s in samples]
    if not any(s.split == 'test' and s.label == 0 for s in samples):
        raise ValueError('test/good is required')
    if not any(s.split == 'test' and s.label == 1 for s in samples):
        raise ValueError('at least one test anomaly is required')
    return samples


def audit_samples(samples):
    rows, seen = [], {}
    for sample in samples:
        digest = sha256_file(sample.path)
        if digest in seen:
            # Reject duplicates even within a split: otherwise they silently reweight NN/calibration.
            raise ValueError(f'Exact duplicate image content: {seen[digest]} and {sample.id}')
        seen[digest] = sample.id
        rows.append({'id': sample.id, 'split': sample.split, 'kind': sample.kind,
                     'label': sample.label, 'sha256': digest})
    return rows


class CNNFeatures(nn.Module):
    """Frozen layer2+layer3 with fixed seeded Gaussian projection, no center crop.

    weights=None is exposed only for unit tests. Benchmark cache_features always
    uses IMAGENET1K_V1 or an explicit local checkpoint; it never silently falls back.
    """
    def __init__(self, backbone, weights, feature_dim, grid_size):
        super().__init__()
        from torchvision.models import resnet18, wide_resnet50_2
        constructor = {'resnet18': resnet18, 'wide_resnet50_2': wide_resnet50_2}[backbone]
        self.net = constructor(weights=weights)
        self.grid_size = grid_size
        channels = 384 if backbone == 'resnet18' else 1536
        rng = torch.Generator().manual_seed(314159)
        self.register_buffer('projection', torch.randn(channels, feature_dim, generator=rng) / math.sqrt(feature_dim))
        self.requires_grad_(False)

    def forward(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        x = n.layer1(x)
        low = n.layer2(x); high = n.layer3(low)
        maps = [F.normalize(F.adaptive_avg_pool2d(z, self.grid_size), dim=1) for z in (low, high)]
        features = torch.cat(maps, 1).flatten(2).transpose(1, 2)
        return F.normalize(features @ self.projection, dim=-1)


def _image_tensor(path, size):
    with Image.open(path) as image:
        image = image.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.
    return torch.from_numpy(array).permute(2, 0, 1)


def cache_features(samples, cache, cfg, device):
    cfg.validate()
    import torchvision
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; explicitly use --device cpu for a CPU run')
    source = {
        'version': 1, 'backbone': cfg.backbone, 'image_size': cfg.image_size,
        'grid_size': cfg.grid_size, 'feature_dim': cfg.feature_dim, 'projection_seed': 314159,
        'weights': sha256_file(cfg.weights_path) if cfg.weights_path else 'IMAGENET1K_V1',
        'resize': 'full-image square bilinear; no crop', 'torch': torch.__version__,
        'torchvision': torchvision.__version__, 'implementation': sha256_file(__file__)}
    fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    paths, pending = {}, []
    for sample in samples:
        key = hashlib.sha256((fingerprint + sha256_file(sample.path)).encode()).hexdigest()
        path = cache / f'{key}.npz'; paths[sample.id] = path
        if path.exists():
            with np.load(path, allow_pickle=False) as stored:
                features = stored['features']
                expected_dim = 3 if cfg.backbone == 'rgb' else cfg.feature_dim
                if features.shape != (cfg.grid_size**2, expected_dim) or not np.isfinite(features).all():
                    raise ValueError(f'Invalid cached features: {path}; remove this entry and rerun')
        else:
            pending.append(sample)
    model = None
    if pending and cfg.backbone != 'rgb':
        weights = None if cfg.weights_path else 'IMAGENET1K_V1'
        model = CNNFeatures(cfg.backbone, weights, cfg.feature_dim, cfg.grid_size)
        if cfg.weights_path:
            state = torch.load(cfg.weights_path, map_location='cpu', weights_only=True)
            model.net.load_state_dict(state, strict=True)
        model = model.to(device).eval()
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    with torch.inference_mode():
        for start in range(0, len(pending), cfg.batch_size):
            batch = pending[start:start+cfg.batch_size]
            images = torch.stack([_image_tensor(s.path, cfg.image_size) for s in batch]).to(device)
            if model is None:
                features = F.adaptive_avg_pool2d(images, cfg.grid_size).flatten(2).transpose(1, 2)
            else:
                features = model((images-mean)/std)
            for sample, feature in zip(batch, features.cpu().float().numpy()):
                # Atomic publication prevents interrupted extraction from poisoning the cache.
                with tempfile.NamedTemporaryFile(dir=cache, suffix='.npz', delete=False) as f:
                    tmp = Path(f.name)
                try:
                    np.savez_compressed(tmp, features=feature, fingerprint=np.array(fingerprint))
                    os.replace(tmp, paths[sample.id])
                finally:
                    tmp.unlink(missing_ok=True)
            if start % (cfg.batch_size * 10) == 0:
                print(f'cached {min(start+len(batch), len(pending))}/{len(pending)} new images', flush=True)
    return paths


class RoleGraph:
    """Fit-only role proxies + a local 4-neighbor observed feature graph.

    No nearest-neighbor semantic edges are added across the image. This still is
    a FEATURE graph, not a recovered physical wire graph. Both roles and bandwidth
    are frozen after fit. CUDA is used for the subsequent diffusion solves.
    """
    def __init__(self, cfg: ImageConfig, seed: int):
        self.cfg, self.seed = cfg.validate(), seed
        g = cfg.grid_size
        yy, xx = np.meshgrid(np.linspace(-1, 1, g), np.linspace(-1, 1, g), indexing='ij')
        self.xy = np.stack([xx.ravel(), yy.ravel()], -1)
        edges = [(y*g+x, y*g+x+1) for y in range(g) for x in range(g-1)]
        edges += [(y*g+x, (y+1)*g+x) for y in range(g-1) for x in range(g)]
        self.edges = np.asarray(edges).T

    def _augment(self, features):
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or len(features) != len(self.xy) or not np.isfinite(features).all():
            raise ValueError('invalid feature grid')
        return np.concatenate([features, self.xy * math.sqrt(self.cfg.position_weight)], -1)

    def fit(self, normal_features):
        if len(normal_features) < 2:
            raise ValueError('need >=2 fit images, excluding calibration')
        rng = np.random.default_rng(self.seed)
        x = np.concatenate([self._augment(f) for f in normal_features])
        if len(x) > self.cfg.kmeans_samples:
            x = x[rng.choice(len(x), self.cfg.kmeans_samples, replace=False)]
        if len(x) < self.cfg.roles:
            raise ValueError('fewer fit samples than role proxies')
        kmeans = MiniBatchKMeans(n_clusters=self.cfg.roles, random_state=self.seed,
                                 n_init=3, batch_size=min(2048, len(x)))
        self.centers = kmeans.fit(x).cluster_centers_
        u, v = self.edges
        distances = np.concatenate([np.sum((f[u]-f[v])**2, 1) for f in normal_features])
        self.bandwidth2 = max(float(np.median(distances)), 1e-8)
        return self

    def transform(self, features, device='cpu'):
        if not hasattr(self, 'centers'):
            raise RuntimeError('fit roles and bandwidth before transforming images')
        x = self._augment(features)
        d = ((x[:, None] - self.centers[None])**2).sum(2)
        logits = -d / self.cfg.temperature
        logits -= logits.max(1, keepdims=True)
        membership = np.exp(logits); membership /= membership.sum(1, keepdims=True)
        u, v = self.edges
        conductance = np.exp(-np.sum((features[u]-features[v])**2, 1) / (2*self.bandwidth2))
        w = np.zeros((len(features), len(features)), dtype=np.float64)
        w[u, v] = w[v, u] = conductance
        tensors = [torch.as_tensor(z, dtype=torch.float64, device=device) for z in (w, membership)]
        return tensors[0], tensors[1], tensors[1]

    def save(self, path):
        np.savez_compressed(path, centers=self.centers, bandwidth2=np.array(self.bandwidth2),
                            config=np.array(json.dumps(asdict(self.cfg))), seed=np.array(self.seed))
