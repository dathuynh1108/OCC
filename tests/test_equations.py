import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nbdbench.data import EXPECTED, split_normal_indices
from nbdbench.heads import DeepHead, DROCCHead, fit_deep, fit_drocc
from nbdbench.math import (
    annulus_project,
    diffusion_embedding,
    empirical_tail,
    farthest_first,
    frame_distances,
    image_score,
    normal_threshold,
)
from nbdbench.run import calibrated_variants, score_method_names, write_package_freeze
from nbdbench.lean_report import comparison_pairs


def test_write_package_freeze_supports_uv_managed_environment(tmp_path):
    destination = tmp_path / "packages.txt"

    write_package_freeze(destination)

    packages = destination.read_text(encoding="utf-8")
    assert any(line.startswith("torch") for line in packages.splitlines())


def test_lean_score_protocol_emits_only_a_plus_d_and_b_plus_d():
    raw = np.zeros((1, 2, 9), dtype=np.float64)
    calibration = np.zeros((2, 2, 9), dtype=np.float64)
    raw[..., 5] = [1.0, 2.0]  # B
    raw[..., 6] = [3.0, 4.0]  # A
    raw[..., 7] = [5.0, 6.0]  # D
    calibration[..., 5] = [0.0, 1.0]
    calibration[..., 6] = [0.0, 3.0]
    calibration[..., 7] = [0.0, 5.0]
    cfg = {"score_protocol": "lean_ad_bd"}

    variants = calibrated_variants(raw, calibration, cfg)

    a = empirical_tail(raw[..., 6], calibration[..., 6].reshape(-1))
    b = empirical_tail(raw[..., 5], calibration[..., 5].reshape(-1))
    d = empirical_tail(raw[..., 7], calibration[..., 7].reshape(-1))
    assert score_method_names(cfg)[-4:] == ["Bubble_A", "Bubble_B", "Bubble_A+D", "Bubble_B+D"]
    assert variants.shape == (1, 2, 9)
    np.testing.assert_allclose(variants[..., 5], a)
    np.testing.assert_allclose(variants[..., 6], b)
    np.testing.assert_allclose(variants[..., 7], a + d)
    np.testing.assert_allclose(variants[..., 8], b + d)


def test_lean_report_compares_each_final_score_to_its_own_component():
    assert comparison_pairs({"score_protocol": "lean_ad_bd"}) == [
        ("Bubble_A+D", "Bubble_A"),
        ("Bubble_B+D", "Bubble_B"),
    ]
from nbdbench.models import BubbleModel, KernelSVDD
from nbdbench.run import METHODS, calibrated_variants


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/full.json").read_text())


def test_original_dataset_full_counts():
    assert len(EXPECTED) == 15
    assert np.array(list(EXPECTED.values())).sum(0).tolist() == [3629, 467, 1258]


def test_splits_use_every_normal_image_once_and_no_test(config):
    for count, _, _ in EXPECTED.values():
        for seed in config["seeds"]:
            result = split_normal_indices(count, seed, config)
            joined = np.concatenate(list(result.values()))
            assert len(set(joined)) == count
            assert sorted(joined) == list(range(count))
            assert all(len(part) for part in result.values())
            assert np.array_equal(
                result["fit"], split_normal_indices(count, seed, config)["fit"]
            )


def test_empirical_tail_includes_ties_and_add_one():
    actual = empirical_tail(
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 2.0, 3.0])
    )
    assert np.allclose(actual, -np.log(np.array([1.0, 1.0, 0.8, 0.4, 0.2])))


def test_threshold_finite_sample_limit_and_top_fraction():
    assert normal_threshold(np.arange(12), 0.05) == float("inf")
    assert normal_threshold(np.arange(19), 0.05) == 18
    assert np.allclose(image_score(np.arange(100)[None], 0.02), [98.5])


def test_diffusion_equals_transition_distance():
    w = np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.4], [0.1, 0.4, 1.0]])
    p, pi, _, phi = diffusion_embedding(w, [1, 3, 5])
    assert np.allclose(pi[:, None] * p, pi[None, :] * p.T)
    for t, embedding in zip([1, 3, 5], phi):
        pt = np.linalg.matrix_power(p, t)
        direct = ((pt[:, None] - pt[None]) ** 2 / pi).sum(-1)
        spectral = ((embedding[:, None] - embedding[None]) ** 2).sum(-1)
        assert np.allclose(direct, spectral, atol=1e-12)
        weights = np.array([0.2, 0.3, 0.5])
        pairwise = 0.5 * (weights[:, None] * weights[None] * direct).sum()
        mean = weights @ embedding
        variance = (weights[:, None] * (embedding - mean) ** 2).sum()
        assert np.isclose(pairwise, variance)


def test_disconnected_graph_keeps_extra_stationary_modes():
    p, pi, values, phi = diffusion_embedding(np.eye(3), [1])
    assert len(values) == 2
    direct = ((p[:, None] - p[None]) ** 2 / pi).sum(-1)
    spectral = ((phi[0, :, None] - phi[0, None]) ** 2).sum(-1)
    assert np.allclose(direct, spectral)


def test_frame_distance_basis_sign_and_rotation_invariant():
    torch.manual_seed(1)
    q, _ = torch.linalg.qr(torch.randn(9, 3, dtype=torch.float64))
    r, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    u = torch.stack([q, -q, q @ r])
    assert torch.allclose(
        frame_distances(u), torch.zeros(3, 3, dtype=torch.float64), atol=1e-12
    )


def test_float32_frame_distance_matches_explicit_projector():
    torch.manual_seed(128)
    frames = torch.stack([torch.linalg.qr(torch.randn(48, 16))[0] for _ in range(4)])
    actual = frame_distances(frames)
    projectors = frames.double() @ frames.double().transpose(1, 2)
    expected = (projectors[:, None] - projectors[None]).square().sum((-1, -2))
    assert torch.equal(actual.diag(), torch.zeros(4))
    assert torch.allclose(actual.double(), expected, atol=2e-6, rtol=0)


def test_farthest_first_unique_even_with_duplicates():
    ids = farthest_first(torch.zeros(12, 4), 12, 0)
    assert len(ids.unique()) == 12
    assert torch.equal(ids, farthest_first(torch.zeros(12, 4), 12, 0))


def test_bubble_energy_matches_explicit_projector_and_replay(config):
    torch.manual_seed(4)
    config.update(bubbles=6, neighbors=16, rank=2, candidate_count=3, graph_neighbors=3)
    x = torch.randn(100, 8)
    model = BubbleModel(config).fit(x, 0)
    query = x[:5]
    ids = torch.arange(6)[None].expand(5, -1)
    energy = model.energy(query, ids)
    for i in range(5):
        for j in range(6):
            delta = query[i] - model.centers[j]
            a = model.u[j].T @ delta
            residual = delta - model.u[j] @ a
            expected = (
                a.square() / (model.eigenvalues[j] + config["epsilon"])
            ).sum() / 2
            expected += residual.square().sum() / (
                6 * (model.sigma[j] + config["epsilon"])
            )
            assert torch.isclose(energy[i, j], expected, atol=1e-5)
    restored = BubbleModel(config).restore(model.state(), "cpu")
    assert torch.equal(model.components(query), restored.components(query))
    assert model.geometry_bytes() > model.centers.numel() * 4


def test_kernel_svdd_qp_constraints_and_score_order():
    torch.manual_seed(10)
    x = torch.randn(80, 3) * 0.1
    model = KernelSVDD().fit(x, 0.1)
    assert model.diagnostics["status"] == "solved"
    assert abs(float(model.alpha.sum()) - 1) < 1e-6
    assert model.alpha.min() >= -1e-6
    assert model.alpha.max() <= 1 / (0.1 * len(x)) + 1e-6
    assert (
        model.score(torch.ones(1, 3) * 10).item()
        > model.score(torch.zeros(1, 3)).item()
    )


def test_deep_head_has_no_bias():
    model = DeepHead(8, 5, 3)
    assert all(
        layer.bias is None
        for layer in model.modules()
        if isinstance(layer, torch.nn.Linear)
    )
    assert not model.center.requires_grad


def test_drocc_analytic_direction_matches_autograd():
    torch.manual_seed(11)
    model = DROCCHead(12, 7).double()
    x = torch.randn(15, 12, dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(torch.nn.functional.softplus(model(x)).mean(), x)[0]
    expected = gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    assert torch.allclose(
        model.input_gradient_direction(x.detach()), expected, atol=1e-10
    )


def test_drocc_annulus_constraints_after_all_steps(config):
    torch.manual_seed(5)
    x = torch.randn(30, 8)
    model = DROCCHead(8, 10)
    adv = model.adversarial(x, 2.0, config)
    distance = (adv - x).norm(dim=1)
    assert torch.all(distance >= 2.0 - 1e-5)
    assert torch.all(distance <= 4.0 + 1e-5)
    assert torch.allclose(
        annulus_project(torch.zeros_like(x), 2.0, 2.0).norm(dim=1),
        torch.full((30,), 2.0),
    )


def test_ten_variants_and_no_test_in_calibration(config):
    rng = np.random.default_rng(1)
    cal = rng.normal(size=(4, 7, 9))
    test = rng.normal(size=(5, 7, 9))
    scores = calibrated_variants(test, cal, config)
    assert scores.shape == (5, 7, len(METHODS))
    altered = np.concatenate([test, np.full_like(test, 1e6)])
    assert np.array_equal(calibrated_variants(altered, cal, config)[:5], scores)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA training smoke requires GPU"
)
def test_cuda_training_full_epochs_and_saved_resume(config, tmp_path):
    config.update(
        ae_epochs=2,
        deep_epochs=2,
        drocc_epochs=3,
        drocc_warmup=1,
        drocc_ascent_steps=3,
        head_batch=32,
        head_hidden=8,
        head_output=3,
        radius_query_count=16,
    )
    x = torch.randn(70, 8, device="cuda")
    head, diagnostic = fit_deep(x, config, tmp_path, 0, "fixture-only")
    drocc, dd = fit_drocc(x, config, tmp_path, 0, "fixture-only")
    expected_deep = head.score(x).detach().clone()
    expected_drocc = drocc.score(x).detach().clone()
    head2, _ = fit_deep(x, config, tmp_path, 0, "fixture-only")
    drocc2, _ = fit_drocc(x, config, tmp_path, 0, "fixture-only")
    assert torch.equal(expected_deep, head2.score(x))
    assert torch.equal(expected_drocc, drocc2.score(x))
    for name, count in [("autoencoder", 2), ("deep_svdd", 2), ("drocc", 3)]:
        history = json.loads((tmp_path / f"{name}.history.json").read_text())
        assert len(history) == count
        assert all(
            row["seen_patches"] == 70 and row["optimizer_steps"] == 3 for row in history
        )
    assert diagnostic["svdd_epochs_completed"] == 2
    assert dd["epochs_completed"] == 3


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA workflow regression requires GPU"
)
def test_complete_saved_prediction_workflow_fixture(config, tmp_path):
    from nbdbench.run import run_one

    config.update(
        ae_epochs=1,
        deep_epochs=1,
        drocc_epochs=2,
        drocc_warmup=1,
        drocc_ascent_steps=2,
        head_batch=32,
        head_hidden=8,
        head_output=3,
        radius_query_count=16,
        bubbles=6,
        neighbors=16,
        rank=2,
        candidate_count=3,
        graph_neighbors=3,
        svdd_support_budget=64,
    )
    rng = np.random.default_rng(22)
    features = rng.normal(size=(102, 2, 8)).astype(np.float32)
    paths = [f"toothbrush/train/good/{i:03d}.png" for i in range(60)]
    paths += [f"toothbrush/test/good/{i:03d}.png" for i in range(12)]
    paths += [f"toothbrush/test/defective/{i:03d}.png" for i in range(30)]
    output = tmp_path / "fixture-only"
    run_one(
        features,
        paths,
        config,
        "toothbrush",
        0,
        output,
        "synthetic-fixture-not-benchmark",
        "cuda",
    )
    verification = json.loads((output / "verification.json").read_text())
    assert verification["status"] == "passed"
    assert len(verification["methods"]) == 10
    assert verification["test_images"] == 42
    run_one(
        None,
        None,
        config,
        "toothbrush",
        0,
        output,
        "synthetic-fixture-not-benchmark",
        "cuda",
    )
