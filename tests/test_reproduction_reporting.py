import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from reproduction.common import sha256
from reproduction.pending_runs import completed_result


def test_full_run_admission_rejects_short_run_and_changed_predictions(tmp_path):
    result_path = tmp_path / "result.json"
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("sample_id,score\n0,0.25\n")
    config = tmp_path / "config.json"
    config.write_text('{"svdd_epochs":150}')
    result = {
        "complete": True, "smoke": False, "svdd_epochs": 12,
        "prediction_sha256": sha256(predictions),
    }
    result_path.write_text(json.dumps(result))
    args = (result_path, {"svdd_epochs": 150}, "predictions.csv", (config, {"svdd_epochs": 150}))
    with pytest.raises(ValueError, match="Result differs"):
        completed_result(*args)
    result["svdd_epochs"] = 150
    result_path.write_text(json.dumps(result))
    assert completed_result(*args)
    predictions.write_text("sample_id,score\n0,0.9\n")
    with pytest.raises(ValueError, match="Prediction hash mismatch"):
        completed_result(*args)


def test_default_mvtec_run_never_reads_or_executes_nbd_track(tmp_path, monkeypatch):
    from reproduction import mvtec

    fixture = tmp_path / "artifacts/reproduction/fixtures/patchcore_parity.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"passed":true}')
    monkeypatch.setattr(mvtec, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["mvtec", "--smoke"])
    monkeypatch.setattr(mvtec, "load_plan", lambda: {
        "native": {"patchcore": {"categories": ["toothbrush"], "seeds": [0]}}
    })
    monkeypatch.setattr(mvtec, "seed_everything", lambda seed: None)
    monkeypatch.setattr(mvtec, "inspect_dataset", lambda *args: {})
    monkeypatch.setattr(mvtec, "selection_rng_from_source", lambda *args: None)
    monkeypatch.setattr(mvtec, "sha256", lambda *args: "plan")
    for name in ("reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(mvtec.torch.cuda, name, lambda: None)
    monkeypatch.setattr(mvtec.torch.cuda, "max_memory_allocated", lambda: 0)

    def forbidden(*args):
        pytest.fail("Native reproduction reached the controlled NBD track")

    for name in ("run_one", "validate_historical_splits", "controlled_patchcore"):
        monkeypatch.setattr(mvtec, name, forbidden)
    calls = []
    monkeypatch.setattr(mvtec, "native", lambda *args: calls.append(args))
    monkeypatch.setitem(sys.modules, "reproduction.patchcore", SimpleNamespace(
        AuthorPatchCore=lambda: SimpleNamespace(before_probe="raw", after_probe="effective"),
        extract_category=lambda *args: (
            np.zeros((3, 1, 1)), ["train", "test1", "test2"],
            np.zeros((2, 1, 1)), np.array([0, 1]), 1,
        ),
    ))
    mvtec.main()
    assert len(calls) == 1
    assert calls[0][8] == tmp_path / "artifacts/reproduction/smoke/mvtec/native/toothbrush/seed-0"
    assert not (tmp_path / "artifacts/reproduction/smoke/mvtec/controlled").exists()


def test_mvtec_full_schedule_uses_immutable_full_plan_and_new_identity():
    from reproduction.mvtec_image_baselines import output_for_schedule, schedule_config

    light = schedule_config("light")
    full = schedule_config("full")
    assert light["plan_path"].name == "run_plan_budgeted.json"
    assert full["plan_path"].name == "run_plan.json"
    assert full["plan"]["native"]["deep_svdd"]["ae"]["cifar10"]["epochs"] == 350
    assert full["plan"]["native"]["deep_svdd"]["svdd_epochs"] == 150
    assert full["plan"]["native"]["drocc"]["epochs"] == 100
    assert output_for_schedule("full") != output_for_schedule("light")
