"""Repair only CSV-parser rounding in completed shallow metric exports.

Preserve original metrics in each JSON. Scores, checkpoints and gamma selection
are untouched. Exact float64 CSV reading must match checkpoint replay first.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from .common import ROOT, write_json
from .shallow import raw_dataset


def main():
    threadpool_limits(4)
    root = ROOT / "results/native/shallow"
    corrected = []
    datasets = {}
    for path in sorted(root.rglob("result.json")):
        result = json.loads(path.read_text())
        if result.get("csv_float_parser") == "round_trip":
            continue
        table = pd.read_csv(
            path.with_name("predictions.csv"), float_precision="round_trip"
        )
        auc = roc_auc_score(table.anomaly_label, table.score)
        ap = average_precision_score(table.anomaly_label, table.score)
        # Check exact serialization against the stored estimator; each corrected
        # export gets a fresh saved-model replay on all its9000 evaluated images.
        if result["dataset"] not in datasets:
            _, _, tx, _ = raw_dataset(result["dataset"])
            datasets[result["dataset"]] = tx
        saved = joblib.load(path.with_name("model.joblib"))
        tx = datasets[result["dataset"]].copy()
        tx /= np.float32(255)
        tx -= saved["scale_min"]
        tx /= saved["scale_max"] - saved["scale_min"]
        features = saved["pca"].transform(tx.reshape(len(tx), -1))[saved["test_ids"]]
        scores = -saved["model"].decision_function(features)
        error = float(np.max(np.abs(scores - table.score.to_numpy())))
        assert error <= 1e-6, (str(path), error)
        old = {k: result[k] for k in ["auroc", "average_precision"]}
        result.update(
            auroc=auc,
            average_precision=ap,
            csv_float_parser="round_trip",
            csv_metrics_before_roundtrip_fix=old,
            csv_checkpoint_replay_max_abs_error=error,
        )
        write_json(path, result)
        corrected.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "old": old,
                "new": {"auroc": auc, "average_precision": ap},
                "checkpoint_replay_max_abs_error": error,
            }
        )
    evidence = ROOT / "artifacts/reproduction/fixtures/csv_roundtrip_repairs.json"
    previous = json.loads(evidence.read_text()) if evidence.exists() else []
    write_json(evidence, previous + corrected)
    print("Verified exports:", len(corrected))


if __name__ == "__main__":
    main()
