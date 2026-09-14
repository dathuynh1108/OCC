import math

import numpy as np
import pytest
import torch

from nbdbench.math import empirical_tail as production_tail
from nbdbench.math import squared_distances
from nbdbench.models import BubbleModel
from reproduction.nbd_forensics import reference


def test_reference_distance_matches_explicit_and_production_near_ties():
    x = np.array([[0.0, 1.0], [1e-7, 1.0], [2.0, -1.0]], dtype=np.float32)
    expected = reference.squared_distance(x, x)
    actual = squared_distances(torch.from_numpy(x).double(), torch.from_numpy(x).double()).numpy()
    assert np.allclose(actual, expected, atol=1e-12, rtol=0)
    assert np.all(expected >= 0)
    assert np.array_equal(np.argsort(expected[0]), np.argsort(actual[0]))


def test_reference_empirical_tail_has_exact_ge_tie_behavior():
    calibration = np.array([0.0, 1.0, 1.0, 3.0])
    values = np.array([[0.0, 1.0, 2.0, 4.0]])
    expected = reference.empirical_tail(values, calibration)
    assert np.allclose(expected, production_tail(values, calibration))
    assert np.allclose(expected, reference.empirical_tail_sorted(values, np.sort(calibration)))
    # At score 1 there are three calibration values >= 1; add-one makes 4/5.
    assert expected[0, 1] == pytest.approx(-math.log(4 / 5))
    assert expected[0, -1] == pytest.approx(math.log(5))


def test_pooling_after_tail_need_not_preserve_raw_pooling_order():
    calibration = np.arange(11, dtype=float)
    raw_first = np.array([[20.0, 0.0]])
    raw_second = np.array([[9.0, 9.0]])
    assert reference.top_mean(raw_first, 1.0)[0] > reference.top_mean(raw_second, 1.0)[0]
    tail_first = reference.top_mean(reference.empirical_tail(raw_first, calibration), 1.0)[0]
    tail_second = reference.top_mean(reference.empirical_tail(raw_second, calibration), 1.0)[0]
    assert tail_second > tail_first


def test_energy_reference_matches_production_with_fixed_geometry():
    config = {"bubbles": 5, "neighbors": 16, "rank": 2, "candidate_count": 3, "graph_neighbors": 3, "epsilon": 1e-6, "beta": 1.0, "temperature": 1.0, "graph_frame_weight": 1.0, "self_loop": 1.0, "diffusion_times": [1, 3, 5]}
    torch.manual_seed(19)
    model = BubbleModel(config).fit(torch.randn(32, 6), seed=2)
    query = torch.randn(1, 6)
    ids = torch.tensor([[0]])
    actual = model.energy(query, ids).item()
    expected = reference.bubble_energy(
        query[0].numpy(), model.centers[0].numpy(), model.u[0].numpy(), model.eigenvalues[0].numpy(), float(model.sigma[0]), config["beta"], config["epsilon"]
    )
    assert actual == pytest.approx(expected, abs=2e-6)


def test_independent_metrics_handle_ties_without_trapezoidal_pr_auc():
    labels = np.array([0, 1, 1, 0])
    scores = np.array([0.5, 0.5, 1.0, 0.0])
    assert reference.auroc_pairwise(labels, scores) == pytest.approx(0.875)
    assert reference.average_precision_grouped(labels, scores) == pytest.approx(5 / 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_runtime_executes_production_bubble_component_path():
    config = {"bubbles": 5, "neighbors": 16, "rank": 2, "candidate_count": 3, "graph_neighbors": 3, "epsilon": 1e-6, "beta": 1.0, "temperature": 1.0, "graph_frame_weight": 1.0, "self_loop": 1.0, "diffusion_times": [1, 3, 5]}
    torch.manual_seed(23)
    model = BubbleModel(config).fit(torch.randn(32, 6, device="cuda"), seed=0)
    components = model.components(torch.randn(9, 6, device="cuda"))
    assert components.device.type == "cuda"
    assert torch.isfinite(components).all()
