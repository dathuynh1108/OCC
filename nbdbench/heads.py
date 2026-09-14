"""Full-epoch heads on shared frozen CNN features; no test-based selection."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .math import annulus_project, squared_distances


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class DeepHead(nn.Module):
    def __init__(self, dimension, hidden, output):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, output, bias=False),
        )
        self.register_buffer("center", torch.zeros(output))

    def forward(self, x):
        return self.encoder(x)

    def score(self, x):
        return (self(x) - self.center).square().sum(1)


class DROCCHead(nn.Module):
    def __init__(self, dimension, hidden):
        super().__init__()
        self.first = nn.Linear(dimension, hidden)
        self.last = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.last(F.relu(self.first(x))).flatten()

    def score(self, x):
        return -self(x)

    @torch.no_grad()
    def input_gradient_direction(self, x):
        # BCEWithLogits(f(x), 0) = softplus(f(x)). Its strictly positive
        # scalar derivative cancels in normalized ascent. This is the exact
        # analytic input gradient for this two-layer ReLU MLP, tested against
        # autograd, not a substituted adversarial objective.
        active = (self.first(x) > 0).to(x.dtype)
        gradient = (active * self.last.weight) @ self.first.weight
        return gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def adversarial(self, x, radius, config):
        h = annulus_project(torch.randn_like(x), radius, config["drocc_gamma"])
        for _ in range(config["drocc_ascent_steps"]):
            h = h + config[
                "drocc_step_factor"
            ] * radius * self.input_gradient_direction(x + h)
            h = annulus_project(h, radius, config["drocc_gamma"])
        return (x + h).detach()


def _train_phase(model, x, cfg, path, epochs, objective, identity, extra=None):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["head_lr"], weight_decay=cfg["weight_decay"]
    )
    path = Path(path)
    history = []
    start_epoch = 0
    if path.exists():
        state = torch.load(path, map_location=x.device, weights_only=False)
        if state["identity"] != identity:
            raise RuntimeError(f"checkpoint identity changed: {path}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state(state["rng_cuda"].cpu())
        start_epoch, history = state["epoch"], state["history"]
    for epoch in range(start_epoch, epochs):
        started = time.time()
        model.train()
        order = torch.randperm(len(x), device=x.device)
        total, seen, steps = 0.0, 0, 0
        for ids in order.split(cfg["head_batch"]):
            batch = x[ids]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, batch, epoch)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite training loss {path.name} epoch {epoch + 1}"
                )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            seen += len(batch)
            steps += 1
        if seen != len(x):
            raise RuntimeError("incomplete epoch")
        row = {
            "epoch": epoch + 1,
            "loss": total / seen,
            "seen_patches": seen,
            "optimizer_steps": steps,
            "seconds": time.time() - started,
        }
        history.append(row)
        atomic_save(
            {
                "identity": identity,
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "extra": extra,
            },
            path,
        )
        path.with_suffix(".history.json").write_text(json.dumps(history, indent=2))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(json.dumps({"stage": path.stem, **row}), flush=True)
    model.eval()
    return history


def fit_deep(x, cfg, out, seed, identity):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = DeepHead(x.shape[1], cfg["head_hidden"], cfg["head_output"]).to(x.device)
    decoder = nn.Sequential(
        nn.Linear(cfg["head_output"], cfg["head_hidden"], bias=False),
        nn.ReLU(),
        nn.Linear(cfg["head_hidden"], x.shape[1], bias=False),
    ).to(x.device)
    ae = nn.Sequential(head.encoder, decoder)
    ae_history = _train_phase(
        ae,
        x,
        cfg,
        out / "autoencoder.pt",
        cfg["ae_epochs"],
        lambda m, b, e: (m(b) - b).square().sum(1).mean(),
        identity,
    )
    with torch.no_grad():
        initial = torch.cat([head(b) for b in x.split(cfg["head_batch"])])
        center = initial.mean(0)
        center = torch.where(
            center.abs() < 0.1, torch.where(center < 0, -0.1, 0.1), center
        )
        head.center.copy_(center)
        initial_variance = initial.var(0, unbiased=False).mean().item()
        del initial
    del ae, decoder
    history = _train_phase(
        head,
        x,
        cfg,
        out / "deep_svdd.pt",
        cfg["deep_epochs"],
        lambda m, b, e: m.score(b).mean(),
        identity,
    )
    with torch.no_grad():
        outputs = torch.cat([head(b) for b in x.split(cfg["head_batch"])])
        variance = outputs.var(0, unbiased=False)
        diagnostic = {
            "initial_output_variance_mean": initial_variance,
            "final_output_variance_mean": float(variance.mean()),
            "final_output_variance_min": float(variance.min()),
            "final_output_variance_per_dimension": variance.cpu().tolist(),
            "variance_ratio": float(variance.mean()) / max(initial_variance, 1e-30),
            "fixed_center": head.center.cpu().tolist(),
            "ae_epochs_completed": len(ae_history),
            "svdd_epochs_completed": len(history),
        }
    (out / "deep_svdd_diagnostics.json").write_text(json.dumps(diagnostic, indent=2))
    return head, diagnostic


@torch.no_grad()
def radius_estimate(x, cfg, seed):
    ids = np.random.default_rng(seed).choice(
        len(x), min(cfg["radius_query_count"], len(x)), replace=False
    )
    values = []
    for start in range(0, len(ids), 64):
        query = ids[start : start + 64]
        distances = squared_distances(x[query], x)
        distances[
            torch.arange(len(query), device=x.device),
            torch.tensor(query, device=x.device),
        ] = float("inf")
        values.append(distances.topk(5, largest=False).values[:, -1].sqrt())
    distance = float(torch.cat(values).median())
    radius = cfg["drocc_radius_factor"] * distance
    if not np.isfinite(radius) or radius <= 0:
        raise RuntimeError("DROCC training-only radius is not positive")
    return radius, ids, distance


def fit_drocc(x, cfg, out, seed, identity):
    torch.manual_seed(seed + 10000)
    torch.cuda.manual_seed_all(seed + 10000)
    head = DROCCHead(x.shape[1], cfg["head_hidden"]).to(x.device)
    radius, ids, median_distance = radius_estimate(x, cfg, seed)
    np.save(out / "drocc_radius_query_indices.npy", ids)

    def objective(model, batch, epoch):
        loss = F.softplus(-model(batch)).mean()
        if epoch >= cfg["drocc_warmup"]:
            negative = model.adversarial(batch, radius, cfg)
            loss = loss + cfg["drocc_mu"] * F.softplus(model(negative)).mean()
        return loss

    history = _train_phase(
        head,
        x,
        cfg,
        out / "drocc.pt",
        cfg["drocc_epochs"],
        objective,
        identity,
        {"radius": radius},
    )
    diagnostics = {
        "radius": radius,
        "normal_5nn_distance_median": median_distance,
        "radius_query_count": len(ids),
        "radius_reference_count": len(x),
        "epochs_completed": len(history),
        "ascent_steps_per_adversarial_batch": cfg["drocc_ascent_steps"],
        "adversarial_epochs": len(history) - cfg["drocc_warmup"],
    }
    (out / "drocc_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return head, diagnostics
