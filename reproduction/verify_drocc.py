"""Compare native DROCC losses, gradients and BN/optimizer behavior to upstream."""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    ROOT,
    load_plan,
    rng_state,
    restore_rng,
    seed_everything,
    write_json,
    verify_source,
)
from .drocc import DROCCTrainer, adjust_learning_rate, make_model_optimizer, train


def compare_state(a, b):
    errors = {k: float((a[k].double() - b[k].double()).abs().max()) for k in a}
    assert max(errors.values()) <= 1e-7, errors
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/reproduction/fixtures/drocc",
        help="Fresh fixture directory; this command refuses to reuse it.",
    )
    args = parser.parse_args(argv)
    verify_source("edgeml")
    seed_everything(17)
    x = torch.randn(4, 3, 32, 32)
    train_data = TensorDataset(x, torch.ones(4), torch.arange(4))
    test_data = TensorDataset(x, torch.tensor([1, 0, 0, 1]), torch.arange(4))
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
    root = args.output.resolve()
    assert not root.exists(), "Refuse stale parity fixture output"
    results = []
    for normal_class in [
        0,
        1,
    ]:  # Exercise both Adam and SGD with the full 50-step search.
        cfg = copy.deepcopy(load_plan()["native"]["drocc"])
        cfg.update(cfg["class_params"][normal_class], epochs=1)
        ref, ropt = make_model_optimizer(cfg)
        candidate, copt = make_model_optimizer(cfg)
        candidate.load_state_dict(ref.state_dict())
        state = rng_state()
        losses, gradients = [], []
        original_bce = F.binary_cross_entropy_with_logits

        def capture_bce(*args, **kwargs):
            value = original_bce(*args, **kwargs)
            losses.append(float(value.detach()))
            return value

        def capture_grad(optimizer, args, kwargs):
            gradients.append(
                [
                    p.grad.detach().clone()
                    for g in optimizer.param_groups
                    for p in g["params"]
                ]
            )

        F.binary_cross_entropy_with_logits = capture_bce
        hook = ropt.register_step_pre_hook(capture_grad)
        source = DROCCTrainer(
            ref, ropt, cfg["mu"], cfg["radius"], cfg["gamma"], torch.device("cuda")
        )
        source.train(
            train_loader,
            test_loader,
            cfg["lr"],
            adjust_learning_rate,
            1,
            only_ce_epochs=0,
            ascent_step_size=cfg["ascent_step_size"],
            ascent_num_steps=50,
            metric="AUC",
        )
        hook.remove()
        ref_losses, ref_gradients = losses[:], gradients[:]
        losses.clear()
        gradients.clear()
        restore_rng(state)
        hook = copt.register_step_pre_hook(capture_grad)
        candidate, _ = train(
            candidate,
            copt,
            train_loader,
            test_loader,
            [0, 1, 1, 0],
            cfg,
            root / str(normal_class),
        )
        hook.remove()
        F.binary_cross_entropy_with_logits = original_bce
        loss_error = float(np.max(np.abs(np.asarray(ref_losses) - np.asarray(losses))))
        grad_error = max(
            float((a - b).abs().max())
            for left, right in zip(ref_gradients, gradients)
            for a, b in zip(left, right)
        )
        assert len(losses) == 52 and loss_error <= 1e-7 and grad_error <= 1e-7
        errors = compare_state(source.model.state_dict(), candidate.state_dict())
        cfg["epochs"] = 2
        seed_everything(333)
        full, fullopt = make_model_optimizer(cfg)
        split, splitopt = make_model_optimizer(cfg)
        split.load_state_dict(full.state_dict())
        state = rng_state()
        full, _ = train(
            full,
            fullopt,
            train_loader,
            test_loader,
            [0, 1, 1, 0],
            cfg,
            root / f"{normal_class}_full",
        )
        restore_rng(state)
        split, _ = train(
            split,
            splitopt,
            train_loader,
            test_loader,
            [0, 1, 1, 0],
            cfg,
            root / f"{normal_class}_resume",
            stop_after=1,
        )
        split, _ = train(
            split,
            splitopt,
            train_loader,
            test_loader,
            [0, 1, 1, 0],
            cfg,
            root / f"{normal_class}_resume",
        )
        resume = compare_state(full.state_dict(), split.state_dict())
        results.append(
            {
                "optimizer": cfg["optimizer"],
                "ascent_steps": 50,
                "BCE_comparisons": len(losses),
                "loss_max_abs_error": loss_error,
                "gradient_max_abs_error": grad_error,
                "parameter_BN_max_abs_error": max(errors.values()),
                "resume_max_abs_error": max(resume.values()),
                "tolerance": 1e-7,
                "passed": True,
            }
        )
    write_json(root.parent / "drocc_parity.json", results)
    print("PASS", results)


if __name__ == "__main__":
    main()
