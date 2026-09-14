"""Build final tables only from every completed, verified run."""

import argparse
import json
from itertools import chain
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import sha256_file
from .run import METHODS, write_json


def collect_verified(output):
    lock = json.loads((output / "protocol_lock.json").read_text())
    cfg = lock["config"]
    frames, predictions, verifications, counts, histories = [], [], [], [], []
    for category in cfg["categories"]:
        for seed in cfg["seeds"]:
            directory = output / category / f"seed-{seed}"
            complete = json.loads((directory / "COMPLETE.json").read_text())
            for name, digest in complete["artifacts"].items():
                if sha256_file(directory / name) != digest:
                    raise RuntimeError(f"artifact hash mismatch: {directory / name}")
            verified = json.loads((directory / "verification.json").read_text())
            if verified["status"] != "passed":
                raise RuntimeError("unverified run")
            verifications.append({"category": category, "seed": seed, **verified})
            metrics = pd.read_csv(directory / "metrics.csv")
            pred = pd.read_csv(directory / "predictions.csv")
            if sorted(metrics.method.tolist()) != sorted(METHODS):
                raise RuntimeError("missing or duplicated score variant")
            for method in METHODS:
                row = metrics[metrics.method == method].iloc[0]
                p = pred[pred.method == method]
                if p.path.duplicated().any():
                    raise RuntimeError("duplicate image predictions")
                if not np.isclose(
                    roc_auc_score(p.label, p.score), row.auroc, atol=1e-12, rtol=0
                ):
                    raise RuntimeError("saved AUROC mismatch")
                if not np.isclose(
                    average_precision_score(p.label, p.score),
                    row.average_precision,
                    atol=1e-12,
                    rtol=0,
                ):
                    raise RuntimeError("saved AP mismatch")
            frames.append(metrics)
            predictions.append(pred)
            split = json.loads((directory / "splits.json").read_text())
            if len(set(chain.from_iterable(split.values()))) != sum(
                map(len, split.values())
            ):
                raise RuntimeError("image split leakage")
            counts.append(
                {
                    "category": category,
                    "seed": seed,
                    **{k: len(v) for k, v in split.items()},
                }
            )
            for stage, epoch_count in [
                ("autoencoder", cfg["ae_epochs"]),
                ("deep_svdd", cfg["deep_epochs"]),
                ("drocc", cfg["drocc_epochs"]),
            ]:
                history = json.loads((directory / f"{stage}.history.json").read_text())
                if [h["epoch"] for h in history] != list(range(1, epoch_count + 1)):
                    raise RuntimeError("missing training epoch")
                expected_patches = len(split["fit"]) * 784
                if any(h["seen_patches"] != expected_patches for h in history):
                    raise RuntimeError("partial training epoch")
                histories.extend(
                    {"category": category, "seed": seed, "stage": stage, **h}
                    for h in history
                )
    return (
        cfg,
        pd.concat(frames),
        pd.concat(predictions),
        verifications,
        pd.DataFrame(counts),
        pd.DataFrame(histories),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    _cfg, metrics, predictions, verifications, counts, histories = collect_verified(out)
    metrics.to_csv(out / "per_category_per_seed.csv", index=False)
    predictions.to_csv(out / "predictions_all.csv", index=False)
    counts.to_csv(out / "split_counts.csv", index=False)
    histories.to_csv(out / "training_histories.csv", index=False)
    seed_macro = (
        metrics.groupby(["method", "seed"], sort=False)[
            ["auroc", "average_precision", "test_fpr", "test_tpr"]
        ]
        .mean()
        .reset_index()
    )
    seed_macro.to_csv(out / "macro_by_seed.csv", index=False)
    summary = (
        seed_macro.groupby("method", sort=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            ap_mean=("average_precision", "mean"),
            ap_sd=("average_precision", "std"),
            fpr_mean=("test_fpr", "mean"),
            tpr_mean=("test_tpr", "mean"),
        )
        .reindex(METHODS)
        .reset_index()
    )
    summary.to_csv(out / "summary.csv", index=False)
    category_summary = (
        metrics.groupby(["category", "method"])
        .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"))
        .reset_index()
    )
    category_summary.to_csv(out / "category_summary.csv", index=False)
    pivot = metrics.pivot(index=["category", "seed"], columns="method", values="auroc")
    delta = pd.DataFrame(
        {
            "NBD_minus_same_centers": pivot.NBD - pivot.PatchScore_same_centers,
            "NBD_minus_byte_budget": pivot.NBD - pivot.PatchScore_byte_budget,
        }
    ).reset_index()
    delta.to_csv(out / "paired_deltas.csv", index=False)
    delta_seed = delta.groupby("seed")[
        ["NBD_minus_same_centers", "NBD_minus_byte_budget"]
    ].mean()
    delta_seed.to_csv(out / "paired_macro_deltas_by_seed.csv")
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#355c7d"] * 9 + ["#db743b"]
    ax.barh(
        summary.method,
        summary.auroc_mean * 100,
        xerr=summary.auroc_sd * 100,
        color=colors,
        capsize=3,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Image AUROC (%) — category macro mean ± seed sample SD")
    ax.set_title("Full MVTec AD · shared frozen Wide ResNet-50-2 · 3 seeds")
    fig.tight_layout()
    fig.savefig(plot_dir / "macro_auroc.png", dpi=180)
    plt.close(fig)
    heat = category_summary.pivot(
        index="category", columns="method", values="auroc_mean"
    ).reindex(columns=METHODS)
    fig, ax = plt.subplots(figsize=(13, 7))
    plot = ax.imshow(
        heat.values * 100, vmin=50, vmax=100, cmap="viridis", aspect="auto"
    )
    ax.set_xticks(range(len(METHODS)), METHODS, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(
                j,
                i,
                f"{heat.iloc[i, j] * 100:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if heat.iloc[i, j] < 0.8 else "black",
            )
    fig.colorbar(plot, ax=ax, label="Image AUROC (%)")
    fig.tight_layout()
    fig.savefig(plot_dir / "category_auroc.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    dm = (
        delta.groupby("category")[
            ["NBD_minus_same_centers", "NBD_minus_byte_budget"]
        ].mean()
        * 100
    )
    dm.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("NBD minus control (AUROC percentage points)")
    fig.tight_layout()
    fig.savefig(plot_dir / "paired_deltas.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, stage in zip(axes, ["autoencoder", "deep_svdd", "drocc"]):
        grouped = histories[histories.stage == stage].groupby("epoch").loss
        mean, sd = grouped.mean(), grouped.std()
        ax.plot(mean.index, mean.values)
        ax.fill_between(mean.index, (mean - sd).values, (mean + sd).values, alpha=0.2)
        ax.set_title(stage)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss, mean ± SD over runs")
    fig.tight_layout()
    fig.savefig(plot_dir / "training_losses.png", dpi=180)
    plt.close(fig)
    lines = [
        "| Method | Image AUROC (%) | AP (%) | Test FPR (%) | Test TPR (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    latex = [
        "\\begin{tabular}{lrr}",
        "\\hline",
        "Method & Image AUROC (\\%) & AP (\\%) \\\\",
        "\\hline",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.method} | {100 * row.auroc_mean:.2f} ± {100 * row.auroc_sd:.2f} | {100 * row.ap_mean:.2f} ± {100 * row.ap_sd:.2f} | {100 * row.fpr_mean:.2f} | {100 * row.tpr_mean:.2f} |"
        )
        name = row.method.replace("_", "\\_")
        latex.append(
            f"{name} & ${100 * row.auroc_mean:.2f} \\pm {100 * row.auroc_sd:.2f}$ & ${100 * row.ap_mean:.2f} \\pm {100 * row.ap_sd:.2f}$ \\\\"
        )
    latex += ["\\hline", "\\end{tabular}"]
    (out / "results_table.tex").write_text("\n".join(latex) + "\n")
    nbd = summary[summary.method == "NBD"].iloc[0]
    lock = json.loads((out / "protocol_lock.json").read_text())
    environment = json.loads((out / "environment.json").read_text())
    backbone = json.loads((out / "backbone.json").read_text())
    delta_text = "\n".join(
        f"- {name}: {100 * delta_seed[name].mean():+.2f} ± {100 * delta_seed[name].std():.2f} AUROC percentage points."
        for name in delta_seed
    )
    report = f"""# Final NBD benchmark report

**Status: complete, with all {len(verifications)} category/seed runs and {len(metrics)} metric rows verified.**

NBD image AUROC: **{nbd.auroc_mean * 100:.2f} ± {nbd.auroc_sd * 100:.2f}%**.
This is a controlled shared-CNN-feature comparison on full MVTec AD, not a
reproduction of the official PatchCore, Deep SVDD or DROCC benchmark tables.
The previous WDBC/Wine/Digits pilot is separate historical evidence.

## Results

{chr(10).join(lines)}

For each seed, first average image AUROC equally over all 15 categories.
Then report the mean and sample SD (ddof=1) of the three seed macro means.
AP uses anomaly=1. Test FPR/TPR are macro means at alpha=0.05 normal-only thresholds.
All ten variants are reported; the NBD variant is fixed B+A+D+F, not the best ablation.

## Paired comparisons

{delta_text}

These are paired by category and seed. See `paired_deltas.csv` for all 45 pairs.

## Data, training and scope

- MVTec AD: 3629 original training normals; 467 normal + 1258 anomalous test images; all 15 categories.
- Every training image belongs to exactly one of fit, score calibration, threshold calibration. Seeds: 0, 1, 2. Full split paths and file hashes are saved.
- Frozen Wide ResNet-50-2 IMAGENET1K_V1, layer2+layer3, 1536-D, 28x28 patch grid, resize 256/crop 224. No global PCA. All methods use identical cached descriptors within each run.
- Deep SVDD: bias-free 1536→64→16 head, 50 complete AE epochs, 100 complete SVDD epochs, fixed nonzero center. Output variance is diagnostic, not a score-selection criterion.
- DROCC: 1536→64→1 head, 100 complete epochs, 10 normal-only warmup epochs, 50 projected ascent steps per adversarial batch. Every fit patch is used in every epoch.
- RBF SVDD: actual dual QP on a declared 2048-point normal coreset. This cap is an adaptation; do not describe it as full-kernel training on every patch.
- NBD: 128 local-PCA bubbles of rank 16, 128-neighbor fits, bidirectional energy graph, projector discrepancy, all nonconstant diffusion modes, times 1/3/5. Full config is locked before any test prediction.
- Same-center control uses the exact NBD centers. Byte-budget control uses at most NBD geometric/graph array bytes; shared backbone and component calibration are excluded from that particular comparison.
- Top-1% mean image aggregation is shared by all variants. Do not retain the old slide's max-PatchScore equation for this table.
- Thresholds use held-out normal images only. Small calibration sets can require +inf at alpha=0.05 (notably toothbrush); retain that fact and the measured FPR/TPR. No test-selected threshold.
- Saved patch maps are not pixel-AUROC/AUPRO measurements. Do not present an evaluated segmentation claim.

## Reproducibility and verification

GPU: {environment["gpu"]}; Python {environment["python"]}; PyTorch {environment["torch"]}; torchvision {environment["torchvision"]}; CUDA {environment["cuda_runtime"]}.

Config SHA-256: `{lock["config_sha256"]}`.
Protocol identity (config/source/data/backbone): `{lock["identity"]}`.
Backbone checkpoint SHA-256: `{backbone["checkpoint_sha256"]}`.

Each completed run includes model/optimizer/RNG checkpoints, epoch histories,
split paths, raw patch scores, image predictions, thresholds, QP diagnostics,
Deep SVDD variance and SHA-256 manifests. Every calibration/threshold/test patch
was rescored from reloaded checkpoints. AUROCs were independently recomputed
from saved prediction CSV files. See `verification.json` and per-run records.

The full float32 descriptor cache lives in RAM one category at a time to respect
GPU-host disk limits. Its hash, ordered input paths and extraction code are saved;
it can be regenerated from original images and the pinned backbone checkpoint.

## Files for slides

`summary.csv`, `category_summary.csv`, `per_category_per_seed.csv`, `split_counts.csv`,
`paired_deltas.csv`, `results_table.tex`, `plots/`, `SLIDE_UPDATE_HANDOFF.md`.

Primary sources and precise algorithm adaptations are documented in the repository README.
"""
    (out / "REPORT.md").write_text(report)
    handoff = f"""# Handoff cho GPT cập nhật slide

## Kết quả được phép dùng

Đã chạy xong full MVTec AD: 15 category × 3 seed, 450 dòng metric, mọi model dùng cùng feature/split.
NBD image AUROC: **{100 * nbd.auroc_mean:.2f} ± {100 * nbd.auroc_sd:.2f}%**.
Dùng nguyên bảng `summary.csv` và `results_table.tex`, không chọn riêng seed/category tốt.

{chr(10).join(lines)}

{delta_text}

## Sửa slide review.pdf

- Trang 3: backbone của thí nghiệm mới là Wide ResNet-50-2 IMAGENET1K_V1 frozen, layer2+layer3, 1536 chiều. ResNet-18 chỉ thuộc các notebook trước đó. PatchScore vẫn là baseline nearest-memory, không gọi là full PatchCore. Đổi image max thành top-1% mean trong bảng so sánh mới.
- Trang 4–6: ghi rõ Deep SVDD-head và DROCC-head trên shared frozen CNN features. AE 50 + SVDD 100 epoch; DROCC 100 epoch/10 warmup/50 ascent. SVDD giải dual QP RBF trên coreset 2048 patch normal; ghi rõ adaptation.
- Trang 7–8: giữ WDBC/Wine/Digits nếu cần lịch sử, gắn nhãn pilot cũ, không trộn điểm với MVTec. Thêm protocol mới: 3629 train normal, 467 test normal, 1258 test anomaly; split train normal 60/20/20; seeds 0/1/2.
- Trang 9–16: NBD đã được đánh giá thực nghiệm theo công thức B+A+D+F. Thêm ablation B, BA, BAD, BAF, full NBD; giữ mọi kết quả dù kém baseline. Không đổi công thức để chạy theo điểm test.
- Trang 16: phân biệt component normal-tail scaling và image threshold calibration; score không phải xác suất anomaly. Threshold alpha=0.05 có thể vô hạn khi thiếu normal calibration, gồm toothbrush. SD là sample SD của ba category-macro seed means.
- Trang 17: đổi trạng thái NBD từ proposal/not benchmarked sang measured on full MVTec AD under controlled shared-CNN protocol. Không tuyên bố official-paper benchmark reproduction hoặc pixel-level performance.

## Hình nên chèn

1. `plots/macro_auroc.png`: bảng xếp hạng có error bar và đủ 10 variants.
2. `plots/category_auroc.png`: kết quả đủ 15 categories.
3. `plots/paired_deltas.png`: chênh lệch NBD với same-center và byte-budget controls.
4. `plots/training_losses.png`: lịch sử train thật. Đọc thêm per-run Deep SVDD variance diagnostics trước khi nói về collapse.

## Nguồn bằng chứng

Đọc `REPORT.md`, `verification.json`, `per_category_per_seed.csv`, `split_counts.csv`,
`predictions_all.csv`, `protocol_lock.json`, `backbone.json`. Code ở repo OCC,
config `configs/full.json`, lệnh chạy lại `scripts/run_full.sh`.
Không tái sử dụng các số pilot cũ làm kết quả mới; không tự điền số còn thiếu.
"""
    (out / "SLIDE_UPDATE_HANDOFF.md").write_text(handoff)
    write_json(
        out / "verification.json",
        {
            "status": "passed",
            "runs": len(verifications),
            "metric_rows": len(metrics),
            "all_artifact_hashes_checked": True,
            "all_csv_auroc_and_ap_recomputed": True,
            "every_epoch_uses_all_fit_patches": True,
            "maximum_checkpoint_replay_error": max(
                v["checkpoint_replay_max_absolute_error"] for v in verifications
            ),
        },
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
