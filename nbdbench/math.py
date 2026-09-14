"""Equations from slides 10-16; numerical routines are independently tested."""

import math

import numpy as np
import torch


def squared_distances(x, y):
    return (
        x.square().sum(1, keepdim=True) + y.square().sum(1)[None] - 2 * x @ y.T
    ).clamp_min(0)


@torch.no_grad()
def farthest_first(x, count, seed):
    count = min(count, len(x))
    first = int(np.random.default_rng(seed).integers(len(x)))
    indices = torch.empty(count, dtype=torch.long, device=x.device)
    distances = torch.full((len(x),), float("inf"), device=x.device)
    norms = x.square().sum(1)
    selected = first
    for i in range(count):
        indices[i] = selected
        d = (norms + x[selected].square().sum() - 2 * (x @ x[selected])).clamp_min(0)
        distances.copy_(torch.minimum(distances, d))
        distances[indices[: i + 1]] = -1
        selected = int(distances.argmax())
    return indices


def empirical_tail(scores, calibration):
    """Exact >= tail, ties retained, normal-only add-one smoothing."""
    ordered = np.sort(np.asarray(calibration, dtype=np.float64))
    if not len(ordered) or not np.isfinite(ordered).all():
        raise ValueError("calibration must be nonempty and finite")
    below = np.searchsorted(ordered, scores, side="left")
    return -np.log((1 + len(ordered) - below) / (len(ordered) + 1))


def image_score(patches, fraction):
    count = max(1, math.ceil(patches.shape[-1] * fraction))
    return np.partition(patches, -count, axis=-1)[..., -count:].mean(axis=-1)


def normal_threshold(scores, alpha):
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    order = math.ceil((len(ordered) + 1) * (1 - alpha))
    return float(ordered[order - 1]) if order <= len(ordered) else float("inf")


def diffusion_embedding(w, times):
    """Reversible chain eigensystem; discard just one stationary mode."""
    w = np.asarray(w, dtype=np.float64)
    if not np.allclose(w, w.T) or np.any(w < 0):
        raise ValueError("affinity must be symmetric and nonnegative")
    degree = w.sum(1)
    pi = degree / degree.sum()
    p = w / degree[:, None]
    symmetric = w / np.sqrt(degree[:, None] * degree[None, :])
    # Explicitly remove the constant eigenfunction. In a disconnected graph,
    # eigenvalue 1 is repeated and dropping an arbitrary eigenvector is wrong.
    basis, _ = np.linalg.qr(np.column_stack([np.sqrt(pi), np.eye(len(pi))]))
    complement = basis[:, 1:]
    values, vectors = np.linalg.eigh(complement.T @ symmetric @ complement)
    order = np.argsort(-np.abs(values))
    values = values[order]
    psi = (complement @ vectors[:, order]) / np.sqrt(pi[:, None])
    phi = np.stack([psi * values[None, :] ** t for t in times])
    return p, pi, values, phi


def frame_distances(u):
    # Compute ||UU^T - VV^T||_F^2 for the actual stored frames. Substituting
    # integer ranks for projector norms amplifies float32 orthogonality error,
    # including a nonzero self-distance that empirical tail ranks can magnify.
    precise = u.double()
    cross = torch.einsum("idr,jds->ijrs", precise, precise)
    squared = cross.square().sum((-1, -2))
    norm = squared.diag()
    distances = (norm[:, None] + norm[None, :] - 2 * squared).clamp_min(0)
    distances.fill_diagonal_(0)
    return distances.to(u.dtype)


def annulus_project(h, radius, gamma):
    norms = h.norm(dim=1, keepdim=True)
    # Zero displacement has no direction: use a fixed axis, never NaN.
    fallback = torch.zeros_like(h)
    fallback[:, 0] = 1
    unit = torch.where(norms > 0, h / norms.clamp_min(1e-30), fallback)
    return unit * norms.clamp(min=radius, max=gamma * radius)
