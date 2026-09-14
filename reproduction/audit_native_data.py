"""Check supplied author GCN bounds against all original training images only."""

import ast
import hashlib
import numpy as np
import torch
from torchvision.datasets import MNIST, CIFAR10
from .common import ROOT, sha256, write_json


def main():
    torch.set_num_threads(4)
    rows = []
    for name, cls in [("mnist", MNIST), ("cifar10", CIFAR10)]:
        tree = ast.parse(
            (ROOT / f"vendor/deep_svdd_torch/src/datasets/{name}.py").read_text()
        )
        bounds = next(
            ast.literal_eval(n.value)
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "min_max" for t in n.targets)
        )
        data = cls(str(ROOT / "data/native"), train=True, download=False)
        raw = np.asarray(data.data)
        labels = np.asarray(data.targets)
        minimum = np.full(10, np.inf)
        maximum = np.full(10, -np.inf)
        for offset in range(0, len(raw), 256):
            x = (
                torch.from_numpy(raw[offset : offset + 256])
                .double()
                .reshape(len(raw[offset : offset + 256]), -1)
                / 255
            )
            x -= x.mean(1, keepdim=True)
            x /= x.abs().mean(1, keepdim=True)
            assert torch.isfinite(x).all()
            lows = x.min(1).values.numpy()
            highs = x.max(1).values.numpy()
            ys = labels[offset : offset + 256]
            for c in np.unique(ys):
                minimum[c] = min(minimum[c], lows[ys == c].min())
                maximum[c] = max(maximum[c], highs[ys == c].max())
        for c in range(10):
            error = float(
                np.max(np.abs(np.array(bounds[c]) - [minimum[c], maximum[c]]))
            )
            rows.append(
                {
                    "dataset": name,
                    "normal_class": c,
                    "original_train_normal_count": int((labels == c).sum()),
                    "source_bounds": bounds[c],
                    "all_train_float64_GCN_bounds": [minimum[c], maximum[c]],
                    "max_abs_error": error,
                    "matches_at_1e_4": error <= 1e-4,
                }
            )
    files = {}
    for directory in ["data/native/MNIST/raw", "data/native/cifar-10-batches-py"]:
        for p in sorted((ROOT / directory).glob("*")):
            if p.is_file():
                files[p.relative_to(ROOT).as_posix()] = {
                    "sha256": sha256(p),
                    "bytes": p.stat().st_size,
                    "md5": hashlib.md5(p.read_bytes()).hexdigest(),
                }
    write_json(
        ROOT / "artifacts/reproduction/fixtures/native_data_audit.json",
        {
            "GCN_bounds": rows,
            "all_bounds_match": all(r["matches_at_1e_4"] for r in rows),
            "files": files,
            "fit_population": "original training split only; no test images used for this audit",
            "training_transforms": "unchanged author constants regardless of audit result",
        },
    )
    print("All source bounds match:", all(r["matches_at_1e_4"] for r in rows))


if __name__ == "__main__":
    main()
