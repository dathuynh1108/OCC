"""Small float64 reference formulas kept independent from production helpers."""

from __future__ import annotations

import math

import numpy as np


def squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Brute-force squared Euclidean distance for small fixtures."""
    out = np.empty((len(x), len(y)), dtype=np.float64)
    for row, a in enumerate(np.asarray(x, dtype=np.float64)):
        for column, b in enumerate(np.asarray(y, dtype=np.float64)):
            out[row, column] = sum((float(left) - float(right)) ** 2 for left, right in zip(a, b))
    return out


def empirical_tail(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Direct >= count definition; deliberately no production implementation reuse."""
    calibration = np.asarray(calibration, dtype=np.float64).reshape(-1)
    if not len(calibration) or not np.isfinite(calibration).all():
        raise ValueError("calibration must be finite and nonempty")
    result = np.empty(np.asarray(values).shape, dtype=np.float64)
    for index, value in np.ndenumerate(np.asarray(values, dtype=np.float64)):
        result[index] = -math.log((1 + int(np.count_nonzero(calibration >= value))) / (len(calibration) + 1))
    return result


def empirical_tail_sorted(values: np.ndarray, ordered_calibration: np.ndarray) -> np.ndarray:
    """Vectorized exact tail for production-scale arrays; caller sorts once per component."""
    ordered = np.asarray(ordered_calibration, dtype=np.float64).reshape(-1)
    if not len(ordered) or not np.isfinite(ordered).all() or np.any(ordered[1:] < ordered[:-1]):
        raise ValueError("calibration must be nonempty, finite and sorted")
    # `side=left` returns values strictly below score, exactly implementing >= ties.
    below = np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="left")
    return -np.log((1 + len(ordered) - below) / (len(ordered) + 1))


def top_mean(patches: np.ndarray, fraction: float) -> np.ndarray:
    patches = np.asarray(patches, dtype=np.float64)
    count = max(1, math.ceil(patches.shape[-1] * fraction))
    return np.sort(patches, axis=-1)[..., -count:].mean(axis=-1)


def normal_threshold(scores: np.ndarray, alpha: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    order = math.ceil((len(ordered) + 1) * (1 - alpha))
    return float(ordered[order - 1]) if order <= len(ordered) else float("inf")


def auroc_pairwise(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    normal, anomaly = scores[labels == 0], scores[labels == 1]
    if not len(normal) or not len(anomaly):
        raise ValueError("AUROC requires both labels")
    pairs = anomaly[:, None] - normal[None, :]
    return float((np.count_nonzero(pairs > 0) + 0.5 * np.count_nonzero(pairs == 0)) / pairs.size)


def average_precision_grouped(labels: np.ndarray, scores: np.ndarray) -> float:
    """AP as a score-tie grouped precision/recall step sum, not PR trapezoids."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("AP requires positives")
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    total = 0.0
    seen = true = 0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[end] == scores[start]:
            end += 1
        block = labels[start:end]
        previous_recall = true / positives
        seen += len(block)
        true += int(block.sum())
        total += (true / positives - previous_recall) * (true / seen)
        start = end
    return float(total)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Average ranks for ties; None for constant inputs."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        result = np.empty(len(values), dtype=np.float64)
        begin = 0
        while begin < len(values):
            end = begin + 1
            while end < len(values) and values[order[end]] == values[order[begin]]:
                end += 1
            result[order[begin:end]] = (begin + end - 1) / 2
            begin = end
        return result

    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def bubble_energy(
    query: np.ndarray,
    center: np.ndarray,
    frame: np.ndarray,
    eigenvalues: np.ndarray,
    sigma: float,
    beta: float,
    epsilon: float,
) -> float:
    """Reference for one bubble via explicit tangent/residual vectors."""
    delta = np.asarray(query, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.float64)
    coordinates = frame.T @ delta
    tangent = float(np.sum(coordinates**2 / (np.asarray(eigenvalues, dtype=np.float64) + epsilon)) / frame.shape[1])
    residual = delta - frame @ coordinates
    return tangent + beta * float(residual @ residual) / ((len(delta) - frame.shape[1]) * (sigma + epsilon))
