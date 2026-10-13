"""Finite-edge resolvent interventions and normal-only scoring.

For L=sum_e w_e b_e b_e^T, R=(I+tL)^-1, Q=RB, weakening edge e by eta gives
Delta_e(B^T R B) = eta*t*w_e / (1-eta*t*w_e*b_e^T R b_e) * v_e v_e^T,
where v_e=Q[u]-Q[v]. B is held fixed during the intervention.
These are interventions on a constructed graph, not causal claims about pixels.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_TIMES = (.5, 2., 8., 32.)


def validate(w: torch.Tensor, anchors: torch.Tensor, roles: torch.Tensor | None = None):
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError('w must be a square graph with at least two nodes')
    if w.dtype not in (torch.float32, torch.float64):
        raise ValueError('use float32 or float64 graph conductances')
    values = (w, anchors) if roles is None else (w, anchors, roles)
    for value in values:
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError('graph, anchors and roles must be finite and nonnegative')
        if value.device != w.device or value.dtype != w.dtype:
            raise ValueError('all tensors must have the same device and dtype')
    if not torch.allclose(w, w.T, atol=1e-7, rtol=1e-5):
        raise ValueError('w must be symmetric')
    if torch.diagonal(w).abs().max() > 1e-8:
        raise ValueError('self conductances must be zero')
    for value in values[1:]:
        if value.ndim != 2 or value.shape[0] != len(w) or value.shape[1] < 1:
            raise ValueError('memberships must have shape [nodes, roles]')
    if roles is not None and not torch.allclose(roles.sum(1), torch.ones_like(roles[:, 0]), atol=1e-5):
        raise ValueError('mediator memberships must sum to one per node')


def _normalize(a):
    # Absent roles stay zero; occupancy is retained in every relational descriptor.
    return a / a.sum(0, keepdim=True).clamp_min(1e-12)


def resolvent(w: torch.Tensor, tau: float) -> torch.Tensor:
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError('diffusion times must be positive and finite')
    lap = torch.diag(w.sum(1)) - w
    system = torch.eye(len(w), device=w.device, dtype=w.dtype) + tau * lap
    return torch.cholesky_inverse(torch.linalg.cholesky(system))


def transfer(w: torch.Tensor, anchors: torch.Tensor, tau: float) -> torch.Tensor:
    validate(w, anchors)
    b = _normalize(anchors)
    return b.T @ resolvent(w, tau) @ b


def _response(w, b, r, tau, eta, linear=False):
    edges = torch.nonzero(torch.triu(w, diagonal=1) > 0, as_tuple=False).T
    u, v = edges
    q = r @ b
    dv = q[u] - q[v]
    factor = eta * tau * w[u, v]
    if not linear:
        resistance = r[u, u] + r[v, v] - 2 * r[u, v]
        denominator = 1 - factor * resistance
        if (denominator <= 0).any():
            raise FloatingPointError('unstable intervention; use float64 or smaller times')
        factor = factor / denominator
    delta = factor[:, None, None] * dv[:, :, None] * dv[:, None, :]
    return delta, edges


def intervene(w, anchors, tau, eta=.5, *, linear=False):
    validate(w, anchors)
    if not 0 < eta <= 1:
        raise ValueError('eta must lie in (0, 1]')
    b = _normalize(anchors)
    r = resolvent(w, tau)
    delta, edges = _response(w, b, r, tau, eta, linear)
    return b.T @ r @ b, delta, edges


def _upper(x):
    index = torch.triu_indices(x.shape[-1], x.shape[-1], device=x.device)
    return x[..., index[0], index[1]].flatten()


@torch.no_grad()
def describe(w, anchors, roles, times: Sequence[float] = DEFAULT_TIMES, eta=.5):
    """Return comparable descriptors. Roles are soft mediator labels, not features.

    rift = [occupancy, endpoint transfer, grouped finite-edge responses].
    allrole_transfer is a deliberately strong diffusion baseline with access to
    EVERY mediator role, not only endpoints. No model gets privileged graph input.
    """
    validate(w, anchors, roles)
    if not times or not 0 < eta <= 1:
        raise ValueError('provide positive times and eta in (0,1]')
    b, c = _normalize(anchors), _normalize(roles)
    occupancy = torch.cat([anchors.mean(0), roles.mean(0)])
    blocks = {name: [] for name in ('transfer', 'allrole_transfer', 'mediated',
                                    'linear', 'collapsed', 'roleless')}
    for tau in times:
        r = resolvent(w, float(tau))
        k = b.T @ r @ b
        all_k = c.T @ r @ c
        delta, edges = _response(w, b, r, float(tau), eta)
        linear, _ = _response(w, b, r, float(tau), eta, linear=True)
        membership = .5 * (roles[edges[0]] + roles[edges[1]])
        mediated = torch.einsum('eg,ers->grs', membership, delta)
        tangent = torch.einsum('eg,ers->grs', membership, linear)
        blocks['transfer'].append(_upper(k))
        blocks['allrole_transfer'].append(_upper(all_k))
        blocks['mediated'].append(_upper(mediated))
        blocks['linear'].append(_upper(tangent))
        blocks['collapsed'].append(_upper(delta.sum(0)))
        # Spectra are invariant to endpoint/mediator name permutations.
        spectra = torch.linalg.eigvalsh(mediated).flatten().sort().values
        blocks['roleless'].append(torch.cat([torch.linalg.eigvalsh(k), spectra]))
    z = {name: torch.cat(values) for name, values in blocks.items()}
    edge_pairs = roles.T @ w @ roles / w.sum().clamp_min(1e-12)
    cost = w.detach().cpu().double().numpy()
    cost = np.divide(1., cost, out=np.zeros_like(cost), where=cost > 0)
    distance = shortest_path(cost, directed=False)
    bounded = np.divide(distance, 1 + distance, out=np.ones_like(distance), where=np.isfinite(distance))
    bounded = torch.as_tensor(bounded, device=w.device, dtype=w.dtype)
    result = {
        'occupancy': occupancy,
        'direct': torch.cat([occupancy, _upper(edge_pairs)]),
        'shortest': torch.cat([occupancy, _upper(c.T @ bounded @ c)]),
        'transfer': torch.cat([occupancy, z['transfer']]),
        'allrole_transfer': torch.cat([occupancy, z['transfer'], z['allrole_transfer']]),
        'rift': torch.cat([occupancy, z['transfer'], z['mediated']]),
        'rift_linear': torch.cat([occupancy, z['transfer'], z['linear']]),
        'rift_no_mediator': torch.cat([occupancy, z['transfer'], z['collapsed']]),
        'rift_no_identity': torch.cat([occupancy.sort().values, z['roleless']]),
    }
    return {name: value.detach().cpu().double().numpy() for name, value in result.items()}


class NormalMemory:
    """Same fit-only diagonal scaling + 1-NN RMS distance for every descriptor.

    A relative variance floor prevents tiny-variance coordinates from dominating.
    Optional PCA is fitted ONLY on the normal memory, never on queries/calibration.
    """
    def __init__(self, normal: np.ndarray, pca_dim: int = 0):
        x = np.asarray(normal, dtype=np.float64)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError('need >=2 finite normal descriptor vectors')
        self.center = x.mean(0)
        scale = x.std(0)
        positive = scale[scale > 1e-10]
        floor = max(.1 * np.median(positive) if len(positive) else 1., 1e-8)
        self.scale = np.maximum(scale, floor)
        self.projection = None
        self.bank = (x - self.center) / self.scale
        if pca_dim:
            if pca_dim < 1:
                raise ValueError('pca_dim must be nonnegative')
            from sklearn.decomposition import PCA
            self.projection = PCA(n_components=min(pca_dim, *x.shape), svd_solver='full')
            self.bank = self.projection.fit_transform(self.bank)

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.center) or not np.isfinite(x).all():
            raise ValueError('query descriptors have invalid shape or values')
        z = (x - self.center) / self.scale
        return self.projection.transform(z) if self.projection is not None else z

    def score(self, x, return_nearest=False):
        z = self.transform(x)
        distances, indices = [], []
        for start in range(0, len(z), 256):
            d = cdist(z[start:start+256], self.bank, metric='sqeuclidean') / self.bank.shape[1]
            i = d.argmin(1)
            distances.extend(np.sqrt(d[np.arange(len(d)), i]))
            indices.extend(i)
        scores = np.asarray(distances)
        # Do not rank machine-roundoff as anomaly evidence on degenerate graphs.
        scores[scores < 1e-12] = 0.
        return (scores, np.asarray(indices)) if return_nearest else scores


def normal_pvalues(scores, calibration):
    calibration = np.sort(np.asarray(calibration, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    if not len(calibration) or not np.isfinite(calibration).all() or not np.isfinite(scores).all():
        raise ValueError('finite normal calibration scores are required')
    return (1 + len(calibration) - np.searchsorted(calibration, scores, side='left')) / (len(calibration) + 1)


def metrics(labels, scores, calibration):
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError('need nonempty matching binary labels and scores')
    p = normal_pvalues(scores, calibration)
    both = len(np.unique(labels)) == 2
    result = {'n': len(labels), 'normal': int((labels == 0).sum()), 'anomaly': int(labels.sum()),
              'auroc': float(roc_auc_score(labels, scores)) if both else None,
              'ap': float(average_precision_score(labels, scores)) if both else None}
    for alpha in (.05, .10):
        pred = p <= alpha
        tag = str(int(alpha * 100))
        result[f'fpr_at_{tag}pct_cal'] = float(pred[labels == 0].mean()) if (labels == 0).any() else None
        result[f'tpr_at_{tag}pct_cal'] = float(pred[labels == 1].mean()) if (labels == 1).any() else None
    return result
