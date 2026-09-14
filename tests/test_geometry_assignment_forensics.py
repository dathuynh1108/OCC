import math

import numpy as np
import pytest
import torch

from nbdbench.math import squared_distances
from nbdbench.models import BubbleModel

from reproduction.nbd_forensics.geometry_assignment import (
    ENERGY_EQ_ATOL,
    _production_scores,
    _raw_replay_errors,
    array_sha256,
    check_descriptor_archive,
    check_score_invariants,
    energy_terms,
    fit_ownership,
    select_candidates,
)


def test_fit_ownership_round_trip_and_neighborhood_diversity():
    bubble = {
        "neighborhood_indices": np.array([[0, 3, 4, 7], [1, 2, 5, 6]]),
        "seed_indices": np.array([0, 7]),
    }
    ownership, diversity = fit_ownership(
        "toy",
        0,
        {"fit": ["toy/train/good/a.png", "toy/train/good/b.png"]},
        bubble,
        [2, 4, 3],
    )
    assert len(ownership) == 8
    assert ownership[3] == {
        "category": "toy",
        "seed": 0,
        "fit_patch_index": 3,
        "fit_image_slot": 0,
        "patch_offset": 3,
        "image_id": "toy/train/good/a.png",
        "patches_per_image": 4,
    }
    assert ownership[4]["image_id"] == "toy/train/good/b.png"
    assert diversity[0]["unique_images"] == 2
    assert diversity[0]["neighborhood_duplicate_index_count"] == 0


def test_fit_ownership_rejects_out_of_range_neighborhood_index():
    with pytest.raises(ValueError, match="outside flattened FIT range"):
        fit_ownership(
            "toy",
            0,
            {"fit": ["toy/train/good/a.png", "toy/train/good/b.png"]},
            {"neighborhood_indices": np.array([[8]]), "seed_indices": np.array([0])},
            [2, 4, 3],
        )


def test_energy_terms_match_direct_covariance_inverse_and_logdets():
    query = np.array([1.5, -2.0, 0.25, 3.0])
    center = np.zeros(4)
    frame = np.eye(4)[:, :2]
    eigenvalues = np.array([2.0, 5.0])
    sigma, beta, epsilon = 3.0, 2.0, 1e-6
    terms = energy_terms(query, center, frame, eigenvalues, sigma, beta, epsilon)
    ell, residual = eigenvalues + epsilon, sigma + epsilon
    c0 = frame @ np.diag(ell) @ frame.T + residual * (np.eye(4) - frame @ frame.T)
    ceff = (
        frame @ np.diag(2 * ell) @ frame.T
        + ((4 - 2) / beta) * residual * (np.eye(4) - frame @ frame.T)
    )
    assert terms["q_total"] == pytest.approx(query @ np.linalg.inv(c0) @ query)
    assert terms["energy_total"] == pytest.approx(query @ np.linalg.inv(ceff) @ query)
    assert terms["logdet_c0"] == pytest.approx(np.linalg.slogdet(c0)[1])
    assert terms["logdet_ceff"] == pytest.approx(np.linalg.slogdet(ceff)[1])
    assert terms["nll0"] == pytest.approx(
        0.5 * (terms["q_total"] + terms["logdet_c0"] + 4 * math.log(2 * math.pi))
    )


def test_candidate_tie_rule_and_energy_search_invariants():
    distances = np.array([[1.0, 1.0, 2.0], [3.0, 2.0, 2.0]])
    nearest, candidates = select_candidates(distances, candidate_count=2)
    assert nearest.tolist() == [0, 1]
    assert candidates.tolist() == [[0, 1], [1, 2]]
    check_score_invariants(
        np.array([4.0, 7.0]), np.array([3.0, 7.0]), np.array([2.0, 5.0])
    )
    with pytest.raises(AssertionError, match="S3 <= S2 <= S1"):
        check_score_invariants(np.array([1.0]), np.array([1.0 + 2 * ENERGY_EQ_ATOL]), np.array([0.0]))


def test_hash_bound_descriptor_archive_requires_exact_source_order(tmp_path):
    descriptors = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    archive = tmp_path / "descriptors.npz"
    paths = np.array(["toy/train/good/0.png", "toy/train/good/1.png", "toy/test/good/0.png"])
    np.savez_compressed(archive, all_descriptors=descriptors, paths=paths)
    splits = {
        "fit": ["toy/train/good/0.png"],
        "score_calibration": ["toy/train/good/1.png"],
        "threshold_calibration": [],
        "test": ["toy/test/good/0.png"],
    }
    full, actual_paths, by_split = check_descriptor_archive(archive, array_sha256(descriptors), splits)
    assert np.array_equal(full, descriptors)
    assert actual_paths == paths.tolist()
    assert np.array_equal(by_split["test"], descriptors[2:])
    with pytest.raises(ValueError, match="descriptor bytes"):
        check_descriptor_archive(archive, "0" * 64, splits)


def test_primary_s0_and_s2_replay_the_source_candidate_path_in_chunks(tmp_path):
    config = {
        "bubbles": 4,
        "neighbors": 8,
        "rank": 2,
        "candidate_count": 2,
        "score_batch": 512,
        "graph_neighbors": 2,
        "epsilon": 1e-6,
        "beta": 1.0,
        "temperature": 1.0,
        "graph_frame_weight": 1.0,
        "self_loop": 1.0,
        "diffusion_times": [1],
    }
    torch.manual_seed(7)
    model = BubbleModel(config).fit(torch.randn(12, 4), seed=3)
    descriptors = torch.randn(2, 2, 4).numpy().astype(np.float32)
    actual = _production_scores(descriptors, model.state(), config, chunk_size=1)
    flat = torch.from_numpy(descriptors.reshape(-1, 4))
    ids = squared_distances(flat, model.centers).topk(2, largest=False).indices
    expected_s0 = squared_distances(flat, model.centers).min(1).values.numpy()
    expected_s2 = model.components(flat)[:, 0].numpy()
    assert np.allclose(actual["s0"], expected_s0)
    assert np.allclose(actual["s2"], expected_s2)
    score_sets = {"score_calibration": actual, "threshold_calibration": actual, "test": actual}
    raw = np.zeros((2, 2, 9), dtype=np.float64)
    raw[..., 0] = expected_s0.reshape(2, 2)
    raw[..., 5] = expected_s2.reshape(2, 2)
    archive = tmp_path / "scores.npz"
    np.savez_compressed(archive, calibration_raw=raw, threshold_raw=raw, test_raw=raw)
    assert max(_raw_replay_errors(score_sets, archive).values()) == pytest.approx(0.0)
