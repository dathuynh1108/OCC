"""Native CIFAR DROCC, using the pinned author's model and adversarial search."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    ROOT,
    environment,
    load_plan,
    restore_rng,
    rng_state,
    save_checkpoint,
    seed_everything,
    sha256,
    verify_source,
    write_json,
)

sys.path[:0] = [
    str(ROOT / "vendor/edgeml/examples/pytorch/DROCC"),
    str(ROOT / "vendor/edgeml/pytorch"),
]
from main_cifar import CIFAR10_LeNet, adjust_learning_rate
from data_process_scripts.process_cifar import CIFAR10_Dataset
from edgeml_pytorch.trainer.drocc_trainer import DROCCTrainer


def make_model_optimizer(config):
    model = torch.nn.DataParallel(CIFAR10_LeNet()).cuda()
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0)
    return model, optimizer


def load_data(root, normal_class, evidence):
    author = CIFAR10_Dataset(str(root), normal_class)
    cache = root / "drocc_cache" / f"class_{normal_class}.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
    else:
        saved = {"original_labels": list(author.test_set.targets)}
        for split in ("train", "test"):
            source = getattr(author, split + "_set")
            rows = [source[i] for i in range(len(source))]
            saved[split] = (
                torch.stack([r[0] for r in rows]),
                torch.tensor([r[1] for r in rows]),
                torch.tensor([r[2] for r in rows]),
            )
        save_checkpoint(cache, saved)
    train, test = TensorDataset(*saved["train"]), TensorDataset(*saved["test"])
    checks = {}
    for split, data in [("train", train), ("test", test)]:
        original = getattr(author, split + "_set")
        errors = [
            float((original[i][0] - data[i][0]).abs().max())
            for i in [0, len(data) // 2, len(data) - 1]
        ]
        assert max(errors) == 0
        checks[split] = {
            "count": len(data),
            "max_abs_error": max(errors),
            "ids": saved[split][2].tolist(),
        }
    write_json(evidence, {**checks, "cache_sha256": sha256(cache), "normal_label": 1})
    return train, test, saved["original_labels"]


@torch.no_grad()
def predict(model, loader, original_labels):
    model.eval()
    rows = []
    for data, target, indices in loader:
        logits = model(data.cuda().float()).squeeze(1).cpu().tolist()
        for i, normal, logit in zip(indices.tolist(), target.tolist(), logits):
            rows.append(
                {
                    "sample_id": i,
                    "original_label": original_labels[i],
                    "normal_label": int(normal),
                    "anomaly_label": 1 - int(normal),
                    "normal_logit": logit,
                    "score": -logit,
                }
            )
    return rows


def train(
    model,
    optimizer,
    train_loader,
    test_loader,
    original_labels,
    config,
    out,
    stop_after=None,
):
    out.mkdir(parents=True, exist_ok=True)
    trainer = DROCCTrainer(
        model,
        optimizer,
        config["mu"],
        config["radius"],
        config["gamma"],
        torch.device("cuda"),
    )
    trainer.ascent_num_steps = config["ascent_num_steps"]
    trainer.ascent_step_size = config["ascent_step_size"]
    history, start_epoch, best_auc, best_epoch = [], 0, -float("inf"), None
    latest = out / "latest.pt"
    if latest.exists():
        state = torch.load(latest, weights_only=False, map_location="cuda")
        assert state["config"] == config
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        history, start_epoch = state["history"], state["epoch"]
        best_auc, best_epoch = state["best_auc"], state["best_epoch"]
    for epoch in range(start_epoch, config["epochs"]):
        tick = time.perf_counter()
        model.train()
        adjust_learning_rate(
            epoch, config["epochs"], config["only_ce_epochs"], config["lr"], optimizer
        )
        ce_total, adv_total, steps, count, grad_total = 0.0, 0.0, 0, 0, 0.0
        for data, target, _ in train_loader:
            data, target = data.cuda().float(), target.cuda().float().squeeze()
            optimizer.zero_grad()
            logits = model(data).squeeze(1)
            ce_loss = F.binary_cross_entropy_with_logits(logits, target)
            adv_loss = trainer.one_class_adv_loss(data[target == 1])
            loss = ce_loss + adv_loss * config["mu"]
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite DROCC loss epoch={epoch}, batch={steps}"
                )
            loss.backward()
            grad_norm = (
                torch.stack(
                    [
                        p.grad.square().sum()
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                )
                .sum()
                .sqrt()
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Nonfinite DROCC gradient epoch={epoch}, batch={steps}"
                )
            optimizer.step()
            ce_total += ce_loss.item()
            adv_total += adv_loss.item()
            grad_total += grad_norm.item()
            steps += 1
            count += len(data)
        rows = predict(model, test_loader, original_labels)
        # Exact source polarity for historical selection, avoiding a rounding tie change.
        auc = roc_auc_score(
            [r["normal_label"] for r in rows], [r["normal_logit"] for r in rows]
        )
        is_best = auc > best_auc
        if is_best:
            best_auc, best_epoch = auc, epoch + 1
        torch.cuda.synchronize()
        row = {
            "epoch": epoch + 1,
            "ce_loss": ce_total / steps,
            "adv_loss": adv_total / steps,
            "gradient_norm": grad_total / steps,
            "lr": optimizer.param_groups[0]["lr"],
            "optimizer_steps": steps,
            "examples": count,
            "test_selected_diagnostic_auroc": auc,
            "seconds": time.perf_counter() - tick,
        }
        history.append(row)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(),
            "epoch": epoch + 1,
            "history": history,
            "config": config,
            "best_auc": best_auc,
            "best_epoch": best_epoch,
        }
        save_checkpoint(latest, state)
        if is_best:
            save_checkpoint(out / "test_selected.pt", state)
        write_json(out / "history.json", history)
        print(json.dumps({"run": str(out.relative_to(ROOT)), **row}), flush=True)
        if stop_after is not None and epoch + 1 >= stop_after:
            break
    return model, history


def export(out, data, labels, config):
    summaries = []
    for selection, name in [
        ("fixed_final_epoch", "latest.pt"),
        ("test_selected", "test_selected.pt"),
    ]:
        state = torch.load(out / name, weights_only=False, map_location="cuda")
        assert selection == "test_selected" or state["epoch"] == config["epochs"]
        model, _ = make_model_optimizer(config)
        model.load_state_dict(state["model"])
        loader = DataLoader(data, batch_size=config["batch_size"], shuffle=False)
        rows = predict(model, loader, labels)
        # Persist/reload checkpoint and score a second fresh model.
        replay, _ = make_model_optimizer(config)
        replay.load_state_dict(
            torch.load(out / name, weights_only=False, map_location="cuda")["model"]
        )
        replay_rows = predict(replay, loader, labels)
        error = max(abs(a["score"] - b["score"]) for a, b in zip(rows, replay_rows))
        assert error <= 1e-6
        csv_path = out / f"{selection}_predictions.csv"
        with csv_path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        with csv_path.open() as f:
            imported = list(csv.DictReader(f))
        y, scores = (
            [int(r["anomaly_label"]) for r in imported],
            [float(r["score"]) for r in imported],
        )
        result = {
            "target": config["target"],
            "dataset": "cifar10",
            "normal_class": config["normal_class"],
            "seed": config["seed"],
            "selection": selection,
            "selected_epoch": state["epoch"],
            "trained_epochs": config["epochs"],
            "test_count": len(rows),
            "auroc": roc_auc_score(y, scores),
            "average_precision": average_precision_score(y, scores),
            "checkpoint_replay_max_abs_error": error,
            "checkpoint_sha256": sha256(out / name),
            "prediction_sha256": sha256(csv_path),
            "smoke": config["smoke"],
            "complete": True,
        }
        write_json(out / f"{selection}_result.json", result)
        summaries.append(result)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal-class", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    verify_source("edgeml")
    cfg = copy.deepcopy(load_plan()["native"]["drocc"])
    cfg.update(cfg["class_params"][args.normal_class])
    cfg.update(seed=args.seed, smoke=args.smoke)
    if args.smoke:
        cfg["epochs"] = 1
    seed_everything(args.seed)
    base = ROOT / (
        "artifacts/reproduction/smoke/drocc"
        if args.smoke
        else f"results/native/drocc/{cfg['target']}"
    )
    out = base / f"class_{args.normal_class}" / f"seed_{args.seed}"
    data, test, labels = load_data(
        ROOT / "data/native", args.normal_class, out / "data_evidence.json"
    )
    model, optimizer = make_model_optimizer(cfg)
    write_json(
        out / "config.json",
        {
            **cfg,
            "runtime": environment(),
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    train_loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True)
    test_loader = DataLoader(test, batch_size=cfg["batch_size"], shuffle=False)
    model, history = train(
        model, optimizer, train_loader, test_loader, labels, cfg, out
    )
    print(json.dumps(export(out, test, labels, cfg)), flush=True)


if __name__ == "__main__":
    main()
