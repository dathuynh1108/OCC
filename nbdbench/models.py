import time

import numpy as np
import osqp
import torch
from scipy import sparse

from .math import (
    diffusion_embedding,
    farthest_first,
    frame_distances,
    squared_distances,
)


class BubbleModel:
    def __init__(self, config):
        self.cfg = config

    @torch.no_grad()
    def fit(self, x, seed):
        c = self.cfg
        self.seed_indices = farthest_first(x, c["bubbles"], seed)
        neighborhoods = (
            squared_distances(x[self.seed_indices], x)
            .topk(min(c["neighbors"], len(x)), largest=False)
            .indices
        )
        self.neighborhood_indices = neighborhoods.cpu().numpy()
        centers, frames, eigenvalues, residuals = [], [], [], []
        for ids in neighborhoods:
            local = x[ids].double()
            center = local.mean(0)
            a = local - center
            # Exact local PCA via the smaller sample Gram matrix; no random PCA.
            values, vectors = torch.linalg.eigh(a @ a.T / (len(local) - 1))
            rank = min(c["rank"], len(local) - 1, x.shape[1] - 1)
            top = values[-rank:].flip(0).clamp_min(c["epsilon"])
            v = vectors[:, -rank:].flip(1)
            u = a.T @ v / torch.sqrt((len(local) - 1) * top)[None]
            # Rank-deficient local data cannot support an invented tangent frame.
            if not torch.allclose(
                u.T @ u,
                torch.eye(rank, device=x.device, dtype=torch.float64),
                atol=1e-5,
            ):
                raise RuntimeError(
                    "local PCA rank deficient; protocol requires a fresh rank decision"
                )
            sigma = (
                (a.square().sum() / (len(local) - 1) - top.sum()) / (x.shape[1] - rank)
            ).clamp_min(c["epsilon"])
            centers.append(center.float())
            frames.append(u.float())
            eigenvalues.append(top.float())
            residuals.append(sigma.float())
        self.centers = torch.stack(centers)
        self.u = torch.stack(frames)
        self.eigenvalues = torch.stack(eigenvalues)
        self.sigma = torch.stack(residuals)
        self.frame = frame_distances(self.u)
        all_ids = torch.arange(len(self.centers), device=x.device)[None].expand(
            len(self.centers), -1
        )
        energies = self.energy(self.centers, all_ids)
        symmetric_energy = (energies + energies.T) / 2
        distances = squared_distances(self.centers, self.centers)
        distances.fill_diagonal_(float("inf"))
        near = distances.topk(
            min(c["graph_neighbors"], len(self.centers) - 1), largest=False
        ).indices
        edge = torch.zeros_like(distances, dtype=torch.bool)
        edge.scatter_(1, near, True)
        edge = edge | edge.T

        def positive_median(a):
            a = a[edge & (a > c["epsilon"])]
            return float(a.median()) if len(a) else 1.0

        self.tau_center = positive_median(symmetric_energy)
        self.tau_frame = positive_median(self.frame)
        delta = (
            symmetric_energy / self.tau_center
            + c["graph_frame_weight"] * self.frame / self.tau_frame
        )
        w = torch.where(edge, torch.exp(-delta.double()), 0).cpu().numpy()
        np.fill_diagonal(w, c["self_loop"])
        self.w = w
        self.p, self.pi, self.graph_eigenvalues, phi = diffusion_embedding(
            w, c["diffusion_times"]
        )
        self.phi = torch.tensor(phi, device=x.device, dtype=torch.float32)
        self.fit_diagnostics = {
            "fit_patches": len(x),
            "descriptor_dimension": x.shape[1],
            "bubbles": len(self.centers),
            "rank": self.u.shape[-1],
            "minimum_edge_weight": float(w[w > 0].min()),
            "near_unit_eigenvalues": int(
                np.sum(np.abs(self.graph_eigenvalues) > 1 - 1e-8)
            ),
            "tau_center": self.tau_center,
            "tau_frame": self.tau_frame,
        }
        return self

    def energy(self, x, ids):
        delta = x[:, None, :] - self.centers[ids]
        projected = torch.einsum("bkd,bkdr->bkr", delta, self.u[ids])
        rank = self.u.shape[-1]
        tangential = (
            projected.square() / (self.eigenvalues[ids] + self.cfg["epsilon"])
        ).sum(-1) / rank
        residual = (delta.square().sum(-1) - projected.square().sum(-1)).clamp_min(0)
        return tangential + self.cfg["beta"] * residual / (
            (x.shape[-1] - rank) * (self.sigma[ids] + self.cfg["epsilon"])
        )

    @torch.no_grad()
    def components(self, x):
        ids = (
            squared_distances(x, self.centers)
            .topk(min(self.cfg["candidate_count"], len(self.centers)), largest=False)
            .indices
        )
        energy = self.energy(x, ids)
        t = self.cfg["temperature"]
        attachment = torch.softmax(-energy / t, dim=1)
        b = energy.min(1).values
        a = -t * (torch.logsumexp(-energy / t, 1) - np.log(energy.shape[1]))
        embedding = self.phi[:, ids, :]
        mean = (attachment[None, :, :, None] * embedding).sum(2)
        d = (
            (attachment[None, :, :, None] * (embedding - mean[:, :, None, :]).square())
            .sum((2, 3))
            .mean(0)
        )
        rf = self.frame[ids[:, :, None], ids[:, None, :]]
        f = 0.5 * (attachment[:, :, None] * attachment[:, None, :] * rf).sum((1, 2))
        return torch.stack([b, a, d, f], dim=1)

    def geometry_bytes(self):
        arrays = [
            self.centers,
            self.u,
            self.eigenvalues,
            self.sigma,
            self.frame,
            self.phi,
        ]
        return (
            sum(a.numel() * a.element_size() for a in arrays)
            + sum(a.nbytes for a in [self.w, self.p, self.pi, self.graph_eigenvalues])
            + 16
        )

    def state(self):
        names = ["centers", "u", "eigenvalues", "sigma", "frame", "phi", "seed_indices"]
        result = {name: getattr(self, name).cpu() for name in names}
        result.update(
            {
                name: getattr(self, name)
                for name in [
                    "w",
                    "p",
                    "pi",
                    "graph_eigenvalues",
                    "tau_center",
                    "tau_frame",
                    "fit_diagnostics",
                    "neighborhood_indices",
                ]
            }
        )
        return result

    def restore(self, state, device):
        for name, value in state.items():
            setattr(self, name, value.to(device) if torch.is_tensor(value) else value)
        return self


class KernelSVDD:
    @torch.no_grad()
    def fit(self, support, nu):
        start = time.time()
        self.support = support.double()
        distances = squared_distances(self.support, self.support)
        positive = distances[distances > 1e-12]
        self.gamma = 1.0 / float(positive.median())
        k = torch.exp(-self.gamma * distances).cpu().numpy()
        n = len(k)
        bound = 1 / (nu * n)
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(2 * k),
            q=-np.diag(k).copy(),
            A=sparse.vstack([np.ones((1, n)), sparse.eye(n)], format="csc"),
            l=np.r_[1.0, np.zeros(n)],
            u=np.r_[1.0, np.full(n, bound)],
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=100000,
            polishing=True,
            verbose=False,
        )
        result = solver.solve()
        if result.info.status != "solved":
            raise RuntimeError(f"SVDD QP failed: {result.info.status}")
        alpha = result.x
        if (
            abs(alpha.sum() - 1) > 1e-5
            or alpha.min() < -1e-5
            or alpha.max() > bound + 1e-5
        ):
            raise RuntimeError("SVDD dual constraints failed")
        self.alpha = torch.tensor(alpha, device=support.device, dtype=torch.float64)
        self.center_norm = float(alpha @ k @ alpha)
        distance = np.diag(k) - 2 * k @ alpha + self.center_norm
        free = (alpha > 1e-5) & (alpha < bound - 1e-5)
        if not np.any(free):
            raise RuntimeError("SVDD has no free support vector for radius")
        self.radius_squared = float(np.median(distance[free]))
        self.diagnostics = {
            "status": result.info.status,
            "iterations": result.info.iter,
            "primal_residual": result.info.prim_res,
            "dual_residual": result.info.dual_res,
            "alpha_sum": float(alpha.sum()),
            "support_training_count": n,
            "free_support_vectors": int(free.sum()),
            "seconds": time.time() - start,
        }
        return self

    @torch.no_grad()
    def score(self, x):
        k = torch.exp(-self.gamma * squared_distances(x.double(), self.support))
        return 1 - 2 * (k @ self.alpha) + self.center_norm - self.radius_squared

    def state(self):
        return {
            k: v.cpu() if torch.is_tensor(v) else v for k, v in self.__dict__.items()
        }

    def restore(self, state, device):
        self.__dict__.update(
            {k: v.to(device) if torch.is_tensor(v) else v for k, v in state.items()}
        )
        return self
