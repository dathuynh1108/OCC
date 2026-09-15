"""Light MVTec comparison with author image CNNs and the source RBF branch.

The input adapter targets a new dataset. This is not a claim that these papers
published this MVTec experiment. Network/loss/training operations are imported
from the source-audited native runners, without a pretrained feature head.
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    ROOT, environment, restore_rng, rng_state, seed_everything, sha256,
    verify_source, write_json,
)

CACHE = ROOT / "artifacts/reproduction/mvtec-images32"
OUTPUT = ROOT / "results/mvtec-native-light-4090-20260915"
FULL_OUTPUT = ROOT / "results/mvtec-native-full-4090-20260915"

SCHEDULES = {
    "light": {
        "plan": ROOT / "run_plan_budgeted.json",
        "output": OUTPUT,
        "deep_target": "MVTec_author_CIFAR_CNN_AE5_SVDD12",
        "drocc_target": "MVTec_author_CIFAR_CNN_CLI_defaults_E5",
        "drocc_selection": "author_test_selected",
    },
    "full": {
        "plan": ROOT / "run_plan.json",
        "output": FULL_OUTPUT,
        "deep_target": "MVTec_author_CIFAR_CNN_AE350_SVDD150",
        "drocc_target": "MVTec_author_CIFAR_CNN_CLI_defaults_E100",
        "drocc_selection": "fixed_final_epoch",
    },
}


def schedule_config(schedule):
    """Return the immutable plan and identity for a named MVTec schedule."""
    settings = SCHEDULES[schedule]
    plan_path = settings["plan"]
    return {**settings, "plan_path": plan_path, "plan": json.loads(plan_path.read_text())}


def output_for_schedule(schedule):
    return SCHEDULES[schedule]["output"]


def protocol(schedule="light"):
    settings = schedule_config(schedule)
    plan = settings["plan"]
    deep = plan["native"]["deep_svdd"]
    drocc_epochs = plan["native"]["drocc"]["epochs"]
    return {
        "scope": f"MVTec image-method comparison, {schedule} schedule, seed 0",
        "schedule": schedule,
        "run_plan": str(settings["plan_path"].relative_to(ROOT)),
        "run_plan_sha256": sha256(settings["plan_path"]),
        "dataset": "mvtec", "seed": 0,
        "data_manifest_sha256": sha256(CACHE / "manifest.json"),
        "source_lock_sha256": sha256(ROOT / "source_lock.json"),
        "deep": {
            **deep,
            "target": settings["deep_target"],
            "ae_settings": deep["ae"]["cifar10"],
            "dataset": "mvtec", "seed": 0,
            "preprocessing": "source L1 GCN; scalar min/max fitted on MVTec normal train only",
        },
        "drocc": {
            "target": settings["drocc_target"],
            "dataset": "mvtec", "seed": 0, "smoke": False,
            "epochs": drocc_epochs, "batch_size": 128, "optimizer": "adam", "lr": .001,
            "radius": .2, "gamma": 2., "mu": 1., "only_ce_epochs": 0,
            "ascent_num_steps": 50, "ascent_step_size": .001,
            "projection_interval": 10,
            "preprocessing": "author CIFAR mean/std, applied to 32x32 MVTec RGB",
            "parameters_source": "main_cifar.py CLI defaults; train() effective ascent default",
            "primary_selection": settings["drocc_selection"],
            "additional_selection": "fixed_final_epoch",
            "lr_schedule": "source thresholds at 40% and 80% of total epochs",
        },
        "shallow": {
            "target": "MVTec_RBF_SVDD_source_non_gridsearch",
            "nu": [.01, .1], "pca_variance": .95,
            "gamma": "1 / max_pairwise_normal_train_distance_squared",
            "source": "deep_svdd_theano/src/svm.py:188-192, GridSearch=False",
            "preprocessing": "source divide255, train min/max, train-only PCA95",
        },
        "evaluation": "all 15 categories, official full test; author test-selected DROCC export is diagnostic only",
        "deep_selection": "fixed_final_epoch",
    }


def load_images(category):
    manifest = json.loads((CACHE / "manifest.json").read_text())
    record = next(r for r in manifest["categories"] if r["category"] == category)
    path = CACHE / f"{category}.npz"
    assert sha256(path) == record["sha256"]
    with np.load(path, allow_pickle=False) as saved:
        raw, labels, paths = saved["pixels"], saved["labels"], saved["paths"].tolist()
        count = int(saved["train_count"])
    assert raw.shape == (len(paths), 3, 32, 32)
    assert count == record["train_count"] and len(paths) - count == record["test_count"]
    assert np.all(labels[:count] == 0) and set(labels[count:]) == {0, 1}
    return raw, labels, paths, count


class ImageData:
    def __init__(self, pixels, labels, count):
        self.train_set = TensorDataset(pixels[:count], torch.zeros(count), torch.arange(count))
        self.test_set = TensorDataset(
            pixels[count:], torch.as_tensor(labels[count:]), torch.arange(len(labels) - count)
        )
        self.original_labels = labels[count:].tolist()

    def loaders(self, batch_size, shuffle_train=True, shuffle_test=False, num_workers=0):
        return (
            DataLoader(self.train_set, batch_size, shuffle=shuffle_train, num_workers=num_workers),
            DataLoader(self.test_set, batch_size, shuffle=shuffle_test, num_workers=num_workers),
        )


def save_scores(out, paths, labels, scores, metadata):
    out.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores, dtype=float)
    assert len(paths) == len(labels) == len(scores) and np.isfinite(scores).all()
    assert len(set(paths)) == len(paths)
    prediction = out / "predictions.csv"
    with prediction.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "anomaly_label", "score"])
        writer.writerows(zip(paths, labels, scores))
    with prediction.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    y = [int(r["anomaly_label"]) for r in rows]
    s = [float(r["score"]) for r in rows]
    result = {
        **metadata, "dataset": "mvtec", "seed": 0, "test_count": len(y),
        "auroc": roc_auc_score(y, s), "average_precision": average_precision_score(y, s),
        "predictions_sha256": sha256(prediction), "complete": True,
    }
    write_json(out / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def run_deep(raw, labels, paths, count, out, cfg):
    from .deep_svdd import (
        AETrainer, build_autoencoder, build_network, train_phase, predict,
    )
    from datasets.preprocessing import global_contrast_normalization

    verify_source("deep_svdd_torch")
    seed_everything(0)
    pixels = torch.from_numpy(raw.copy()).float().div_(255)
    pixels = torch.stack([global_contrast_normalization(x, scale="l1") for x in pixels])
    low, high = pixels[:count].min(), pixels[:count].max()
    assert torch.isfinite(pixels).all() and high > low
    pixels = (pixels - low) / (high - low)
    write_json(out / "preprocessing.json", {"train_min": float(low), "train_max": float(high)})
    data = ImageData(pixels, labels, count)
    net = build_network("cifar10_LeNet")
    ae = build_autoencoder("cifar10_LeNet")
    ae, _, _, _ = train_phase(ae, data, cfg, "ae", "shared_initialization", out / "ae")
    saved_ae = torch.load(out / "ae/latest.pt", map_location="cuda", weights_only=False)
    restore_rng(saved_ae["rng"])
    AETrainer(batch_size=cfg["batch_size"], device="cuda").test(data, ae)
    start_rng = rng_state()
    state = net.state_dict()
    state.update({k: v for k, v in ae.state_dict().items() if k in state})
    net.load_state_dict(state)
    initial = copy.deepcopy(net.state_dict())
    for objective in cfg["objectives"]:
        directory = out / objective
        net.load_state_dict(initial)
        restore_rng(start_rng)
        net, center, radius, _ = train_phase(net, data, cfg, "svdd", objective, directory)
        rows = predict(net, data, center, radius, objective)
        saved = torch.load(directory / "latest.pt", map_location="cuda", weights_only=False)
        replay = build_network("cifar10_LeNet").cuda()
        replay.load_state_dict(saved["net"])
        replay_rows = predict(replay, data, saved["center"], saved["radius"], objective)
        error = max(abs(a["score"] - b["score"]) for a, b in zip(rows, replay_rows))
        assert error <= 1e-6
        save_scores(directory, paths[count:], labels[count:], [r["score"] for r in rows], {
            "method": f"DeepSVDD_{objective}", "category": out.parent.name,
            "selection": "fixed_final_epoch", "checkpoint_sha256": sha256(directory / "latest.pt"),
            "checkpoint_replay_max_abs_error": error,
        })


def run_drocc(raw, labels, paths, count, out, cfg):
    from .drocc import make_model_optimizer, train, predict
    from torchvision.transforms.functional import normalize

    verify_source("edgeml")
    seed_everything(0)
    pixels = normalize(torch.from_numpy(raw.copy()).float() / 255,
                       [.4914, .4822, .4465], [.247, .243, .261])
    data = ImageData(pixels, labels, count)
    train_data = TensorDataset(pixels[:count], torch.ones(count), torch.arange(count))
    test_data = TensorDataset(pixels[count:], 1 - torch.as_tensor(labels[count:]),
                              torch.arange(len(labels) - count))
    train_loader = DataLoader(train_data, cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test_data, cfg["batch_size"], shuffle=False)
    model, optimizer = make_model_optimizer(cfg)
    train(model, optimizer, train_loader, test_loader, data.original_labels, cfg, out)
    for selection, checkpoint in (("fixed_final_epoch", "latest.pt"), ("author_test_selected", "test_selected.pt")):
        saved = torch.load(out / checkpoint, map_location="cuda", weights_only=False)
        model.load_state_dict(saved["model"])
        rows = predict(model, test_loader, data.original_labels)
        replay, _ = make_model_optimizer(cfg)
        replay.load_state_dict(saved["model"])
        replay_rows = predict(replay, test_loader, data.original_labels)
        error = max(abs(a["score"] - b["score"]) for a, b in zip(rows, replay_rows))
        assert error <= 1e-6
        save_scores(out / selection, paths[count:], labels[count:], [r["score"] for r in rows], {
            "method": "DROCC", "category": out.parent.name, "selection": selection,
            "selected_epoch": saved["epoch"], "checkpoint_sha256": sha256(out / checkpoint),
            "checkpoint_replay_max_abs_error": error,
        })


def run_shallow(raw, labels, paths, count, out, cfg):
    import joblib
    from sklearn.metrics import pairwise_distances
    from sklearn.svm import OneClassSVM
    from .shallow import prepare, svdd_diagnostics

    verify_source("deep_svdd_theano")
    train_x, test_x, pca, low, high = prepare(
        raw[:count].astype(np.float32), raw[count:].astype(np.float32), np.arange(count)
    )
    max_distance = float(np.max(pairwise_distances(train_x)))
    assert max_distance > 0
    gamma = 1 / max_distance ** 2
    for nu in cfg["nu"]:
        model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(train_x)
        scores = -model.decision_function(test_x)
        diagnostics = svdd_diagnostics(model, train_x, gamma, nu)
        directory = out / f"nu_{nu}"
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "pca": pca, "minimum": low, "maximum": high}, directory / "model.joblib")
        replay = joblib.load(directory / "model.joblib")["model"]
        error = float(np.max(np.abs(scores + replay.decision_function(test_x))))
        assert error <= 1e-7
        save_scores(directory, paths[count:], labels[count:], scores, {
            "method": f"RBF_SVDD_nu_{nu}", "category": out.parent.name,
            "gamma": gamma, "nu": nu, "pca_dimensions": train_x.shape[1],
            "selection": "source_non_gridsearch_train_only", "diagnostics": diagnostics,
            "checkpoint_sha256": sha256(directory / "model.joblib"),
            "checkpoint_replay_max_abs_error": error,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("deep", "drocc", "shallow"), required=True)
    parser.add_argument("--category")
    parser.add_argument("--schedule", choices=tuple(SCHEDULES), default="light")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or output_for_schedule(args.schedule)
    plan = protocol(args.schedule)
    if args.schedule == "light":
        assert plan["deep"]["ae_settings"]["epochs"] == 5 and plan["deep"]["svdd_epochs"] == 12
        assert plan["drocc"]["epochs"] == 5
    else:
        assert plan["deep"]["ae_settings"]["epochs"] == 350 and plan["deep"]["svdd_epochs"] == 150
        assert plan["drocc"]["epochs"] == 100
    lock = output / "protocol.json"
    if lock.exists():
        assert json.loads(lock.read_text()) == plan
    else:
        write_json(lock, plan)
    manifest = json.loads((CACHE / "manifest.json").read_text())
    categories = [r["category"] for r in manifest["categories"]]
    if args.category:
        assert args.category in categories
        categories = [args.category]
    seed_everything(0)
    write_json(output / f"environment_{args.method}.json", {
        **environment(), "schedule": args.schedule, "protocol_sha256": sha256(lock),
    })
    for category in categories:
        directory = output / category / args.method
        complete = directory / "COMPLETE.json"
        if complete.exists():
            saved = json.loads(complete.read_text())
            for name, digest in saved["artifacts"].items():
                assert sha256(directory / name) == digest
            continue
        raw, labels, paths, count = load_images(category)
        write_json(directory / "splits.json", {"train": paths[:count], "test": paths[count:]})
        cfg = {**plan[args.method], "category": category}
        write_json(directory / "config.json", cfg)
        start = time.perf_counter()
        {"deep": run_deep, "drocc": run_drocc, "shallow": run_shallow}[args.method](
            raw, labels, paths, count, directory, cfg
        )
        artifacts = {
            str(p.relative_to(directory)): sha256(p)
            for pattern in ("**/result.json", "**/predictions.csv", "**/*.pt", "**/*.joblib")
            for p in directory.glob(pattern)
        }
        write_json(complete, {"artifacts": artifacts, "seconds": time.perf_counter() - start})
        print("COMPLETE", args.method, category, flush=True)


if __name__ == "__main__":
    main()
