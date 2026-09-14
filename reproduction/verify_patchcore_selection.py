"""Source end-to-end fit versus cached fit must choose identical coreset rows."""

import numpy as np
import torch
from .common import ROOT, rng_state, seed_everything, write_json
from .patchcore import AuthorPatchCore, datasets
from .mvtec import selection_rng_from_source


@torch.no_grad()
def main():
    train, _ = datasets(ROOT / "data/mvtec_ad", "bottle")
    subset = torch.utils.data.Subset(train, [0, 1, 2])
    seed_everything(17)
    direct = AuthorPatchCore()
    direct.model.fit(
        torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    )
    memory = direct.model.anomaly_scorer.detection_features.copy()
    indices = direct.sampler.selected_indices.copy()
    expected_rng = rng_state()
    # Source iteration is deterministic apart from its iterator base seed.
    features = np.concatenate(
        [direct.descriptors(train[i]["image"][None]) for i in [0, 1, 2]]
    )
    del direct
    seed_everything(17)
    cached = AuthorPatchCore()
    selection_rng_from_source(cached, ROOT / "data/mvtec_ad", "bottle")
    actual, actual_indices = cached.fit_memory(features)
    actual_rng = rng_state()
    np.testing.assert_array_equal(memory, actual)
    np.testing.assert_array_equal(indices, actual_indices)
    np.testing.assert_array_equal(expected_rng["numpy"][1], actual_rng["numpy"][1])
    assert expected_rng["numpy"][2:] == actual_rng["numpy"][2:]
    assert torch.equal(expected_rng["torch"], actual_rng["torch"])
    assert all(
        torch.equal(a, b) for a, b in zip(expected_rng["cuda"], actual_rng["cuda"])
    )
    write_json(
        ROOT / "artifacts/reproduction/fixtures/patchcore_selection_parity.json",
        {
            "passed": True,
            "real_train_images": 3,
            "selected_descriptors": len(memory),
            "memory_max_abs_error": 0,
            "selected_indices_identical": True,
            "RNG_after_identical": True,
        },
    )


if __name__ == "__main__":
    main()
