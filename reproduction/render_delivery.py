"""Generate concise Beamer slides, plots and handoff from independently audited CSVs."""

from __future__ import annotations
import json
import re
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reproduction-2026-09-14"
DECK = ROOT / "slides/paper-faithful-review"
ASSETS = DECK / "assets"
LABELS = {
    "PatchScore_same_centers": "PatchScore: same centers",
    "PatchScore_byte_budget": "PatchScore: byte budget",
    "RBF_SVDD": "RBF SVDD (feature adaptation)",
    "DeepSVDD_head": "Deep SVDD head (15 epochs)",
    "DROCC_head": "DROCC head (15 epochs)",
    "Bubble_B": "B",
    "Bubble_BA": "B+A",
    "Bubble_BAD": "B+A+D",
    "Bubble_BAF": "B+A+F",
    "NBD": "NBD: B+A+D+F",
    "PatchCore_same_fit_top1pct": "PatchCore: same FIT, top 1%",
}
NAVY = "#16324f"
TEAL = "#008b8b"
ORANGE = "#d87831"


def pct(x):
    return f"{float(x) * 100:.2f}"


def stat(mean, sd, tex=False):
    sep = r" $\pm$ " if tex else " ± "
    return pct(mean) + (sep + pct(sd) if pd.notna(sd) else "")


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(ASSETS / f"{name}.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)


def frame(title, body):
    return "\n\\begin{frame}[t]{" + title + "}\n\\small\n" + body + "\n\\end{frame}\n"


def band(text):
    return "\n\\vfill\n\\band{\\small " + text + "}\n"


def source():
    return r"\src{Measured rerun: \href{https://github.com/dathuynh1108/OCC/tree/main/results/reproduction-2026-09-14}{OCC / reproduction-2026-09-14}. See CSVs and source audit.}"


def image(name, height="4.6cm"):
    return (
        "\n\\centering\\includegraphics[width=.98\\textwidth,height="
        + height
        + ",keepaspectratio]{"
        + name
        + ".pdf}\n"
    )


def bullets(items):
    return (
        "\n\\begin{itemize}\n"
        + "".join("\\item " + s + "\n" for s in items)
        + "\\end{itemize}\n"
    )


def table(headers, rows, width=None):
    body = (
        "\\begin{center}\n\\renewcommand{\\arraystretch}{1.16}\n\\begin{tabular}{l"
        + "r" * (len(headers) - 1)
        + "}\n\\toprule\n"
        + " & ".join("\\textbf{" + h + "}" for h in headers)
        + r" \\"
        + "\n\\midrule\n"
    )
    body += (
        "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{center}\n"
    )
    return body


def main():
    audit = json.loads((OUT / "independent_metric_audit.json").read_text())
    assert audit["passed"] and audit["common_complete"]
    common = pd.read_csv(OUT / "common_summary.csv").set_index("method")
    native = pd.read_csv(OUT / "native_summary.csv")
    paper = json.loads((ROOT / "docs/reproduction/paper_reference.json").read_text())

    def reference(track, dataset, variant, normal=None):
        if track == "PatchCore":
            return paper["roth2022"]["mvtec_image_auroc_percent"]
        if track == "DROCC":
            values = paper["goyal2020"][dataset]
        else:
            key = "kernel_best_nu" if track == "Gaussian_OCSVM_equiv_SVDD" else variant
            values = paper["ruff2018"][dataset][key]
        return float(np.mean(values)) if normal is None else values[int(normal)]

    native["paper_reference_auroc_percent"] = [
        reference(r.track, r.dataset, r.variant) for _, r in native.iterrows()
    ]
    native["comparison_status"] = native.track.map(
        {
            "PatchCore": "released_code_replication; image reweighting differs from paper",
            "DeepSVDD": "later_author_port_replication; reduced epochs; historical parity unverified",
            "DROCC": "released_image_CNN_replication; reduced epochs; selection policy explicit",
            "Gaussian_OCSVM_equiv_SVDD": "source_protocol_modern_solver_adaptation; separate nu rows",
        }
    )
    assert native.comparison_status.notna().all()
    native.to_csv(OUT / "paper_vs_measured_summary.csv", index=False)
    per_class = pd.read_csv(OUT / "native_per_class_per_seed.csv")
    per_class = per_class[per_class.track != "PatchCore"].copy()
    per_class["paper_reference_auroc_percent"] = [
        reference(r.track, r.dataset, r.variant, r.class_or_category)
        for _, r in per_class.iterrows()
    ]
    per_class.to_csv(OUT / "paper_vs_measured_per_class.csv", index=False)
    coverage = pd.read_csv(OUT / "coverage_summary.csv")
    inference = pd.read_csv(OUT / "inference_bottle_seed0.csv")
    cats = pd.read_csv(OUT / "common_category_summary.csv")
    methods = list(common.index)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
        }
    )

    def bars(selected, name):
        fig, ax = plt.subplots(figsize=(11, 4.7 if len(selected) > 6 else 3.6))
        for i, m in enumerate(selected):
            r = common.loc[m]
            color = ORANGE if m == "NBD" else TEAL
            ax.errorbar(
                r.auroc_mean * 100,
                i,
                xerr=r.auroc_sd * 100,
                fmt="o",
                color=color,
                capsize=4,
                markersize=7,
            )
            ax.text(
                104,
                i,
                stat(r.auroc_mean, r.auroc_sd),
                va="center",
                fontsize=11,
                weight="bold" if m == "NBD" else "normal",
            )
        ax.set_yticks(range(len(selected)), [LABELS[m] for m in selected])
        ax.invert_yaxis()
        ax.set_xlim(0, 102)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("Image AUROC (%) — category macro mean ± SD across 3 seeds")
        ax.grid(axis="x", alpha=0.15)
        fig.subplots_adjust(left=0.33, right=0.82, bottom=0.17, top=0.96)
        save(fig, name)

    bars(methods, "rerun_all_methods")
    bars(
        [
            "PatchScore_same_centers",
            "PatchScore_byte_budget",
            "RBF_SVDD",
            "DeepSVDD_head",
            "DROCC_head",
            "PatchCore_same_fit_top1pct",
        ],
        "rerun_baselines",
    )
    bars(
        ["Bubble_B", "Bubble_BA", "Bubble_BAD", "Bubble_BAF", "NBD"], "rerun_ablations"
    )
    mat = cats.pivot(index="method", columns="category", values="auroc_mean").loc[
        methods
    ]
    fig, ax = plt.subplots(figsize=(12, 5.6))
    im = ax.imshow(
        mat.to_numpy() * 100, vmin=0, vmax=100, aspect="auto", cmap="viridis"
    )
    ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods], fontsize=10)
    ax.set_xticks(
        range(15),
        [x.replace("_", " ") for x in mat.columns],
        rotation=40,
        ha="right",
        fontsize=10,
    )
    for i in range(len(methods)):
        for j in range(15):
            value = mat.iloc[i, j] * 100
            ax.text(
                j,
                i,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value < 50 else NAVY,
            )
    fig.colorbar(im, ax=ax, label="Image AUROC (%)", fraction=0.025, pad=0.02)
    fig.tight_layout()
    save(fig, "rerun_categories")
    # A 2D schematic of the native CIFAR source setting, not a synthetic benchmark.
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=TEAL, lw=2)
    ax.scatter([0], [0], color=NAVY, s=70)
    ax.scatter([1], [0], color=ORANGE, s=80)
    ax.annotate("normal x", (0, 0), (0.1, 0.2))
    ax.annotate(
        "generated negative",
        (1, 0),
        (0.7, 0.5),
        arrowprops={"arrowstyle": "->", "color": ORANGE},
    )
    ax.annotate("", (1, 0), (0, 0), arrowprops={"arrowstyle": "<->", "color": NAVY})
    ax.text(0.43, -0.18, "r")
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 2.1)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")
    save(fig, "drocc_source_sampling")
    # Markdown results: one common benchmark, native rows separate by target.
    rows = [["Method", "Image AUROC (%)", "AP (%)", "FPR (%)", "TPR (%)"]]
    for m, r in common.iterrows():
        rows.append(
            [
                LABELS[m],
                stat(r.auroc_mean, r.auroc_sd),
                stat(r.ap_mean, r.ap_sd),
                pct(r.fpr_mean),
                pct(r.tpr_mean),
            ]
        )

    def md(rows):
        return (
            "| "
            + " | ".join(rows[0])
            + " |\n| "
            + " | ".join(["---"] * len(rows[0]))
            + " |\n"
            + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in rows[1:])
        )

    native_rows = [
        [
            "Target",
            "Dataset",
            "Variant",
            "Paper AUROC (%)",
            "Measured AUROC (%)",
            "Repeats",
        ]
    ]
    for _, r in native.iterrows():
        native_rows.append(
            [
                r.track,
                r.dataset,
                r.variant,
                f"{r.paper_reference_auroc_percent:.2f}",
                stat(r.auroc_mean, r.auroc_sd),
                str(r.repeats),
            ]
        )
    inference_rows = [
        ["Method", "Median ms", "p95 ms", "State MiB", "NBD calibration MiB"]
    ]
    for _, r in inference.iterrows():
        inference_rows.append(
            [
                LABELS.get(r.method, "Shared author WR50 descriptor"),
                f"{r.median_ms:.2f}",
                f"{r.p95_ms:.2f}",
                f"{r.state_array_bytes / 2**20:.2f}",
                f"{r.separate_NBD_component_calibration_bytes / 2**20:.2f}",
            ]
        )
    nbd = common.loc["NBD"]
    delta = (
        pd.read_csv(OUT / "common_paired_deltas.csv")
        .groupby("seed")
        .mean(numeric_only=True)
    )
    comparisons = []
    for m in [
        "PatchScore_same_centers",
        "PatchScore_byte_budget",
        "PatchCore_same_fit_top1pct",
    ]:
        values = delta["NBD_minus_" + m] * 100
        comparisons.append(
            f"- NBD minus {LABELS[m]}: {values.mean():+.2f} ± {values.std(ddof=1):.2f} AUROC percentage points."
        )
    incomplete = coverage[coverage["sum"] < coverage["count"]]
    coverage_text = (
        "All selected matrix rows completed."
        if incomplete.empty
        else "Incomplete targets remain explicit:\n"
        + md(
            [["Target", "Completed", "Planned"]]
            + [[r.track, r["sum"], r["count"]] for _, r in incomplete.iterrows()]
        )
    )
    report = f"""# OCC source-audited reproduction and common MVTec benchmark

The common MVTec experiment completed 15 categories × 3 seeds × 11 variants.
**Final declared NBD (B+A+D+F): {stat(nbd.auroc_mean, nbd.auroc_sd)}% image AUROC; {stat(nbd.ap_mean, nbd.ap_sd)}% AP.**
Native-paper datasets are separate method evaluations, not a cross-dataset ranking.

The user replaced the original long schedule with about two hours remaining.
All scores below are real measured exports. Reduced-epoch results are **not full-schedule historical-paper reproductions**.

## Common MVTec results

{md(rows)}
AUROC/AP: mean of category-macro seed means; sample SD across three seeds.
FPR/TPR: category-and-seed macro means at normal-only calibration thresholds.
Poor results and every ablation remain visible. The best ablation does not replace NBD.

{chr(10).join(comparisons)}

Shared protocol: official MVTec AD, 3,629 normal train images and all 1,725 test
images (467 normal, 1,258 anomaly). Exact historical v2 image split IDs: 60% FIT,
20% NBD component calibration, remainder image threshold calibration. Shared
WR50-2 ImageNet V1, author 3×3 Unfold/alignment/pooling into 784×1,024 descriptors;
author startup BN probe retained, then frozen/eval. No v2 feature cache reused.
All common rows use mean of top 1% patch scores; thresholds use alpha=.05,
k=ceil((n+1)*.95), infinity when k>n, strict score>threshold. No threshold tuning on test.

Deep head: AE 5+SVDD 15. DROCC feature head:15 epochs (10 warmup + 5 adversarial),50
ascent steps. Native-method names must not be applied to these feature heads.
NBD math, graph, calibration,128 bubbles/rank16 and byte-budget controls are unchanged
from91223d6; only author descriptor construction and the user-approved epoch budgets
change. This is a post-result protocol correction, not evidence of a causal improvement.

## Native method results

{md(native_rows)}
Paper columns are reference values, not matched-protocol deltas. For Ruff/Goyal,
they are macros calculated from the published Table 1 class means; no seed SD is
inferred from class SDs. Ruff's kernel and soft-boundary paper rows select the
better nu, whereas the measured rows retain the declared separate configurations.
PatchCore references the paper's 10% row (99.0%), whose equation differs from
the released image-score code. Source URLs and exact printed values are in
[paper_reference.json](../../docs/reproduction/paper_reference.json).
Per-class reference/measured values are in paper_vs_measured_per_class.csv.
{coverage_text}

- PatchCore: author-code WR50/V1, full normal training,10% approximate greedy coreset,
  FP32 GPU FAISS squared-L2 1NN, image maximum. Pixel AUROC is in native_summary.csv;
  AUPRO not evaluated. Released source omits paper Eq.7 reweighting, so that exact
  paper target is not claimed. Three declared repeat seeds0/1/2.
- Deep SVDD: unchanged author PyTorch LeNet modules, MNIST/CIFAR10, all normal classes,
  seeds1..10, AE 5+SVDD 12, source 10-epoch soft-boundary warmup. Final-epoch selection.
  Original LR milestones remain outside this short run. Theano historical runtime
  was not reproduced; modern author-port parity is verified only on this runtime.
- DROCC: unchanged author CIFAR CNN and adversarial function, five epochs, 50 ascent
  steps, projection every10, class-specific Table 11 parameters and gamma 1. Both
  final and best-test-selected outputs retained. **Best-test-selected is optimistic
  and uses test labels**, as the released runner does; it is not a clean final-test estimate.
- Gaussian OC-SVM/equivalent RBF SVDD: modern libsvm/source-protocol adaptation,
  full selected-class training except source MNIST batch-multiple rule, train-only
  PCA 95%, gamma2**[-10..-1], both nu=.01/.1. Gamma uses1000 labeled test samples;
  evaluated9000 exclude that holdout. Nu rows remain separate. This is not an
  executed MATLAB dd_tools or Tax--Duin2004 numeric reproduction.
- Tax--Duin Table2 remains unreproduced: exact Iris folds, sigma search and numerical
  settings were not recovered. No arbitrary substitute score is reported.

## Inference scope and stored state

{md(inference_rows)}
Measured after training, on one bottle image / seed 0, batch size 1 (784 patches),
30 repetitions after 3 warmups, CUDA synchronized. This is an illustrative actual
checkpoint measurement, not a dataset-wide latency claim. Shared descriptor time
excludes disk/PIL; method times exclude the CNN and include CPU descriptor input,
transfers and image pooling. NBD includes the current implementation's ECDF sorting.
State counts numerical arrays only; calibration is separate, not peak GPU memory.
Full scope: inference_scope.json. Geometry budgets across all category/seed runs:
common_resources.csv. The common backbone is counted once, separately.

## Verification, provenance and rerun

{audit["audited_prediction_rows"]:,} exported image scores were checked independently
with Python standard-library tied-rank AUROC and threshold-group AP. Common FPR/TPR
were recalculated with the declared strict threshold. IDs, labels, uniqueness,
counts, prediction hashes and checkpoint replay gates were checked. See
independent_metric_audit.json and coverage.csv for exact evidence.

Pinned source hashes: source_lock.json. Original schedule: run_plan.json.
Budgeted schedule: run_plan_budgeted.json. Source analysis:
docs/reproduction/SOURCE_AUDIT.md. Numeric fixtures cover upstream Deep/DROCC
losses, gradients, parameters, BN and resume; PatchCore descriptors, NN/maps and
cached/direct coreset/RNG parity; shallow extracted source splits/PCA and independent
QP equivalence. Every admitted trained result replays its persisted checkpoint.

A shallow replay gate caught a float32 PCA batch-shape difference (9,000 versus 10,000
rows, maximum score error about2.04e-6). Replay was corrected to transform the same
full10000 rows before selecting the9000 evaluated IDs, restoring exact agreement;
no metric tolerance, trained estimator, gamma selection or dataset was changed.
Failed attempts are retained in runtime logs and are not admitted as results.

Training ran on one rented RTX3090 with FP32, AMP/TF32 disabled. Native GPU jobs and
CPU libsvm jobs overlapped, so training wall times are not comparable standalone
latencies. Geometric/coreset bytes exclude the shared backbone, calibration and
allocator overhead. Any separate inference timing is labeled by measured scope.

The independent audit also checks exact CSV float64 parsing. Pandas default parsing
can merge nearly tied kernel scores at about1e-16. Exports now use round-trip
parsing, retain previous metrics for traceability, and are checked against saved
estimator scores. No gamma selection or trained model was changed.

Code, CSVs, histories, manifests, figures and updated Beamer source/PDF are in Git.
Large checkpoints, score maps, official datasets and V1 weights are retained in
the verified local backup; hashes and rerun commands ship in the repository.
Historical results/mvtec-full-v2 are preserved unchanged and must not be mixed
with this author-encoder/reduced-epoch experiment.
"""
    (OUT / "REPRODUCTION_REPORT.md").write_text(report)
    handoff = f"""# Slide update handoff — measured 14 September 2026

Use the updated `slides/paper-faithful-review/review.tex` and `review.pdf`.
The slide text is intentionally short; this report and CSVs hold the detail.

## Main message

NBD B+A+D+F: **{stat(nbd.auroc_mean, nbd.auroc_sd)}% AUROC**, **{stat(nbd.ap_mean, nbd.ap_sd)}% AP** on the common MVTec benchmark.
Keep NBD as the declared model even when an ablation/baseline scores higher.

{md(rows)}
## What changed in the deck

- Use the new1024-D author descriptor, preserve the author's startup BN behavior.
- Show one common15-category ×3-seed MVTec table with11 explicitly named variants.
- Put MNIST/CIFAR native results only on their own method reproduction slides.
- Label AE 5/SVDD 12, nativeDROCC5 and sharedhead15 epochs visibly; do not say full
  historical-paper reproduction. Keep original full commands in the rerun guide.
- Keep best-test-selected DROCC distinct from fixed-final results.
- Explain DROCC negatives briefly: Gaussian start, gradient ascent, radius projection,
  anomaly training label. Native CIFAR gamma 1 gives a sphere; feature adaptation differs.
- Show exact coverage; Tax2004 original numeric target stays unreproduced.
- Keep older v2 numbers as historical evidence in their original directory.

{coverage_text}

## Editing rules

Use concise English captions and tables, one main point per slide. Do not paste
this report as slide prose. Preserve source citations and uncertainty; do not
replace missing cells, low scores or final NBD with a stronger ablation.

Sources of numbers: common_summary.csv, common_category_summary.csv,
common_paired_deltas.csv, native_summary.csv, native_per_class_per_seed.csv,
paper_vs_measured_summary.csv and paper_vs_measured_per_class.csv.
Independent check: independent_metric_audit.json. Exact selected configuration:
run_plan_budgeted.json. Full detail: REPRODUCTION_REPORT.md and SOURCE_AUDIT.md.
"""
    (OUT / "SLIDE_UPDATE_HANDOFF.md").write_text(handoff)
    (ROOT / "CODEX_OCC_PAPER_REPRODUCTION.md").write_text(
        "# Updated OCC reproduction handoff\n\n"
        "The supplied task specification is preserved in docs/reproduction/REQUEST.md. "
        "This file records the measured delivery; it does not replace the original evidence.\n\n"
        + f"NBD B+A+D+F: **{stat(nbd.auroc_mean, nbd.auroc_sd)}% image AUROC**, "
        + f"**{stat(nbd.ap_mean, nbd.ap_sd)}% AP** on all 15 MVTec categories × 3 seeds.\n\n"
        + "- [Complete slide-update handoff](results/reproduction-2026-09-14/SLIDE_UPDATE_HANDOFF.md)\n"
        + "- [Measured report and protocol differences](results/reproduction-2026-09-14/REPRODUCTION_REPORT.md)\n"
        + "- [Paper versus measured native results](results/reproduction-2026-09-14/paper_vs_measured_summary.csv)\n"
        + "- [Updated slide PDF](slides/paper-faithful-review/review.pdf) / [Overleaf source](slides/paper-faithful-review/review.tex)\n"
        + "- [Exact rerun commands](docs/reproduction/RERUN.md)\n\n"
        + "The user selected fewer epochs and about two hours remaining: native Deep "
        + "AE5/SVDD12, native DROCC5, common heads15. These are source-audited shortened "
        + "replications. Historical Theano numerical parity and Tax2004's exact Iris table "
        + "remain unverified/unreproduced. Native datasets and the common MVTec comparison "
        + "are separate. All weak scores and the declared final NBD remain visible.\n\n"
        + coverage_text
        + "\n"
    )
    readme = f"""# OCC: NBD and source-audited anomaly detection benchmarks

Latest delivery:14 September2026. One shared MVTec comparison, plus separate
native-dataset evaluations for each method.

**NBD B+A+D+F: {stat(nbd.auroc_mean, nbd.auroc_sd)}% image AUROC; {stat(nbd.ap_mean, nbd.ap_sd)}% AP.**
All15 MVTec categories ×3 seeds ×11 variants, using author WR50-2/V1 1,024-D
features and the same held-out image splits. Mean±sample SD across seed macro means.

- [Updated slide PDF](slides/paper-faithful-review/review.pdf) · [Overleaf source](slides/paper-faithful-review/review.tex)
- **[Handoff for OpenAI / slide updates](results/reproduction-2026-09-14/SLIDE_UPDATE_HANDOFF.md)**
- [Full report](results/reproduction-2026-09-14/REPRODUCTION_REPORT.md) · [Common table](results/reproduction-2026-09-14/common_summary.csv)
- [Native tables](results/reproduction-2026-09-14/native_summary.csv) · [Exact coverage](results/reproduction-2026-09-14/coverage.csv)
- [Independent metric audit](results/reproduction-2026-09-14/independent_metric_audit.json) · [Rerun guide](docs/reproduction/RERUN.md)
- [Source audit](docs/reproduction/SOURCE_AUDIT.md) · [Pinned sources](source_lock.json)

![Common MVTec image AUROC](slides/paper-faithful-review/assets/rerun_all_methods.png)

The user selected about two hours remaining and reduced epochs: native Deep SVDD
AE5+SVDD12, native DROCC5, common deep heads15. PatchCore/NBD/kernel fitting keeps
its declared non-epoch protocol. These are measured reduced-epoch method
replications, **not full-schedule historical-paper reproductions**. Native image
CNNs and shared-feature heads are labeled separately. Best-test-selected DROCC
is retained as an explicitly optimistic source behavior. Tax2004's exact numeric
experiment remains unreproduced because historical settings were not recovered.

{coverage_text}

Git contains code, raw predictions, histories, configs, evidence and slide source/PDF.
Large fitted checkpoints, maps, original datasets and V1 weights are retained in
the local backup; see backup_manifest.json and backup_verified.json in the latest
result directory. Runtime/lifecycle status is recorded in execution_closure.json.

To reproduce: follow [RERUN.md](docs/reproduction/RERUN.md).
`run_plan_budgeted.json` is the measured short schedule; `run_plan.json` retains
original long schedules with separate output identities. Source licenses remain
with each pinned upstream checkout. Dataset images are not redistributed in Git.

Historical v2 results remain unchanged: [v2 report](results/mvtec-full-v2/REPORT.md),
[v2 README](README_V2.md). Their1536-D feature pipeline and longer head schedules
must not be mixed into this corrected1024-D experiment.
"""
    (ROOT / "README.md").write_text(readme)
    # Preserve the supplied visual style and scientific NBD theory frames.
    original = (DECK / "review.input.tex").read_text()
    pre = original[: original.index(r"\begin{document}")]
    pre = pre.replace(
        "% Same measured v2 snapshot. Added original-method sources, exact PatchCore feature path, Git links and rerun handoff. No new benchmark run.",
        "% Measured author-source rerun; reduced epochs explicitly selected by the user.",
    )
    pre = pre.replace(r"\def\phase{NBD}", r"\def\phase{OCC REPRODUCTION}")
    pre = pre.replace(
        "Full MVTec AD; shared frozen CNN benchmark; repo 91223d6",
        "Source-audited native methods and common MVTec; reduced epoch budgets",
    )
    pre = pre.replace(
        "Full MVTec AD results and model review",
        "Source-audited methods and shared MVTec results",
    )
    frames = re.findall(
        r"\\begin\{frame\}(?:\[[^\]]*\])?.*?\\end\{frame\}", original, re.S
    )

    def pick(title):
        return next(f for f in frames if "{" + title + "}" in f)

    title = frames[0].replace(
        "Full MVTec AD results and model review",
        "Source-audited methods and shared MVTec results",
    )
    title = title.replace(
        "Baselines $\\;\\to\\;$ Baseline results $\\;\\to\\;$ NBD $\\;\\to\\;$ Comparison",
        "Native methods $\\;\\to\\;$ Shared MVTec $\\;\\to\\;$ NBD",
    )
    problem = (
        pick("One-class detection")
        .replace(
            "Training and calibration: normal only.",
            "Common MVTec train/calibration: normal only.",
        )
        .replace(
            r"Image benchmark: $z_p=f_{\rm CNN}(x)_p$; the same frozen CNN descriptors are used by every method.",
            "Common MVTec uses shared frozen CNN descriptors. Native methods use their own input pipelines.",
        )
    )
    contents = [title, problem]
    contents.append(
        frame(
            "Two evaluation tracks",
            table(
                ["Track", "Data", "Purpose"],
                [
                    [
                        "Native methods",
                        "MNIST / CIFAR / MVTec",
                        "Method-specific replication",
                    ],
                    [
                        "Common benchmark",
                        "MVTec:15 categories",
                        "Same features and splits",
                    ],
                ],
            )
            + bullets(
                [
                    r"User-selected runtime budget: about two hours; training epochs shortened.",
                    r"Keep all selected classes, categories and seeds in the coverage manifest.",
                    r"Native image CNNs and shared-feature heads are different experiments.",
                ]
            )
            + band(r"Full data does not mean full paper training schedule."),
        )
    )
    pcframe = pick("PatchCore: author feature extractor")
    pcframe = re.sub(
        r"\\band\{\\footnotesize \\textbf\{Our recorded v2 run:.*?\n\\src",
        lambda m: (
            r"\band{\footnotesize Current rerun: author 1,024-D descriptors. Preserve the startup BN shape probe, then freeze/eval. Native image maximum; common table top-1\% mean.}"
            + "\n\\src"
        ),
        pcframe,
        flags=re.S,
    )
    contents.append(pcframe)

    def native_table(track):
        part = native[native.track == track]
        rows = []
        for _, r in part.iterrows():
            label = {
                "one-class": "one-class",
                "soft-boundary": "soft boundary",
                "fixed_final_epoch": "fixed final",
                "test_selected": "best test epoch",
                "author_code": "author code",
                "nu_0.01": r"$\nu=.01$",
                "nu_0.1": r"$\nu=.1$",
            }.get(r.variant, esc(r.variant))
            rows.append(
                [
                    esc(r.dataset),
                    label,
                    f"{r.paper_reference_auroc_percent:.2f}",
                    stat(r.auroc_mean, r.auroc_sd, True),
                ]
            )
        return (
            table(["Dataset", "Variant", "Paper (\\%)", "Measured (\\%)"], rows)
            if rows
            else r"No balanced complete repeat available; see coverage.csv."
        )

    contents.append(
        frame(
            "PatchCore: native MVTec result",
            native_table("PatchCore")
            + (
                r"\remarktext{Pixel AUROC: "
                + pct(native[native.track == "PatchCore"].iloc[0].pixel_auroc_mean)
                + r"\%. AUPRO not evaluated.}"
                if len(native[native.track == "PatchCore"])
                else ""
            )
            + bullets(
                [
                    r"Full normal train; 10\% coreset; 1NN squared L2; image maximum.",
                    r"WR50-2 ImageNet V1; all 15 categories, three seeds.",
                    r"Released author code omits paper Eq.(7) reweighting.",
                ]
            )
            + source(),
        )
    )
    contents.append(
        frame(
            "Deep SVDD: collapse and safeguards",
            r"\[s(x)=\|f_\theta(x)-c\|^2,\qquad f_\theta(x)\equiv c\ \Rightarrow\ s(x)=0.\]"
            + bullets(
                [
                    r"Train the image CNN end to end; fix a nonzero center $c$.",
                    r"No convolution/linear bias; non-affine BatchNorm; leaky ReLU.",
                    r"Reconstruct normal images with an AE, copy its encoder, discard the decoder.",
                    r"AE is an initialization; output variance is a collapse diagnostic.",
                ]
            )
            + band(
                r"MNIST: conv 8/4 $\to$32. CIFAR: conv 32/64/128 $\to$128. No ImageNet backbone."
            )
            + r"\src{\href{https://github.com/lukasruff/Deep-SVDD/tree/e20f18c8d0ad9dc01cad09fdf311bd861351a9ad}{Original Theano source} $\mid$ \href{https://github.com/lukasruff/Deep-SVDD-PyTorch/tree/1901612d595e23675fb75c4ebb563dd0ffebc21e}{Executed author PyTorch modules}.}",
        )
    )
    contents.append(
        frame(
            "Deep SVDD: native image CNNs",
            native_table("DeepSVDD")
            + bullets(
                [
                    r"Author PyTorch LeNet modules; AE 5 + SVDD 12 epochs.",
                    r"Soft-boundary warmup 10; final epoch; 10 classes $\times$10 seeds.",
                    r"Later author-code replication; original Theano numeric parity unverified.",
                ]
            )
            + r"\src{Paper: \href{https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf}{Ruff et al., Table 1}; class macro of printed means. Protocols differ.}"
            + source(),
        )
    )
    contents.append(
        frame(
            "DROCC: how negatives are generated",
            r"\begin{columns}[T]\begin{column}{.48\textwidth}"
            + image("drocc_source_sampling", "3.5cm")
            + r"\end{column}\begin{column}{.49\textwidth}"
            + bullets(
                [
                    r"Add Gaussian noise to a normal image.",
                    r"Ascend the loss for the anomaly target.",
                    r"Project displacement to $[r,\gamma r]$.",
                    r"Train on the resulting negative.",
                ]
            )
            + r"\end{column}\end{columns}"
            + band(
                r"CIFAR author code: 50 ascent steps; project every 10; $\gamma=1$ gives a sphere."
            )
            + r"\src{\href{https://github.com/microsoft/EdgeML/tree/81025fce8ba28707eabe72e11bf3987a8d745608/examples/pytorch/DROCC}{Author CIFAR runner} $\mid$ \href{https://github.com/microsoft/EdgeML/blob/81025fce8ba28707eabe72e11bf3987a8d745608/pytorch/edgeml_pytorch/trainer/drocc_trainer.py}{Adversarial training source}.}",
        )
    )
    contents.append(
        frame(
            "DROCC: native CIFAR result",
            native_table("DROCC")
            + bullets(
                [
                    r"Author image CNN and Table 11 parameters; five epochs.",
                    r"All 10 classes, three seeds; 5,000 train normals / 10,000 test.",
                    r"Best-test epoch uses test labels; interpret it as optimistic.",
                ]
            )
            + r"\src{Paper: \href{https://proceedings.mlr.press/v119/goyal20c/goyal20c.pdf}{Goyal et al., Table 1}; class macro of printed means. Protocols differ.}"
            + source(),
        )
    )
    contents.append(
        frame(
            "Kernel SVDD: source protocol and limits",
            native_table("Gaussian_OCSVM_equiv_SVDD")
            + bullets(
                [
                    r"Ruff image baseline: train-only PCA 95\%, full kernel fit.",
                    r"Gamma tuned on 1,000 labeled test images; evaluate the other 9,000.",
                    r"Modern libsvm / Gaussian SVDD equivalence checked by independent QP.",
                ]
            )
            + band(
                r"Tax--Duin2004 original numeric table remains unreproduced: missing exact folds and search settings."
            )
            + r"\src{\href{https://github.com/DMJTax/dd_tools/tree/efaaf04efae1f8be78906836a5d31547b48be7af}{Tax author reference toolbox} $\mid$ Paper column: \href{https://proceedings.mlr.press/v80/ruff18a/ruff18a.pdf}{Ruff Table 1, better nu}; measured: 10 seeds per nu.}",
        )
    )
    contents.append(
        frame(
            "Common MVTec: one shared benchmark",
            table(
                ["Normal train", "Test normal", "Test anomaly"],
                [["3,629", "467", "1,258"]],
            )
            + bullets(
                [
                    r"15 categories $\times$3 seeds; author WR50/V1 1,024-D descriptors.",
                    r"Image split:60\% FIT /20\% component calibration /remainder threshold.",
                    r"All methods: top1\% patch mean; normal-only image thresholds.",
                    r"Deep head: AE 5+SVDD 15; DROCC head:15 epochs including10 warmup.",
                ]
            )
            + band(r"Feature heads are controlled adaptations; the CNN is frozen."),
        )
    )
    contents.append(
        frame(
            "Common MVTec: baseline results",
            image("rerun_baselines", "4.3cm")
            + band(
                r"PatchCore keeps a 10\% coreset. PatchScore byte budget is the separate NBD memory control."
            )
            + source(),
        )
    )
    threshold_rows = [
        [
            LABELS[m].replace(" (feature adaptation)", "").replace(" (15 epochs)", ""),
            pct(common.loc[m, "fpr_mean"]),
            pct(common.loc[m, "tpr_mean"]),
        ]
        for m in [
            "PatchScore_same_centers",
            "PatchScore_byte_budget",
            "RBF_SVDD",
            "DeepSVDD_head",
            "DROCC_head",
            "PatchCore_same_fit_top1pct",
            "NBD",
        ]
    ]
    threshold_rows = [[esc(x) for x in row] for row in threshold_rows]
    contents.append(
        frame(
            "Common MVTec: operating points",
            table(["Method", "FPR (\\%)", "TPR (\\%)"], threshold_rows)
            + band(
                r"$\alpha=.05$, $k=\lceil(n+1)\cdot.95\rceil$; threshold $=\infty$ if $k>n$; strict $s>\tau$."
            )
            + source(),
        )
    )
    for t in [
        "NBD: local bubbles, one shared graph",
        "A local normality bubble",
        "Scoring a query",
        "Comparing two bubbles",
        "From bubbles to a graph",
        "Diffusion distance",
        "Does the query attach to one region?",
        "NBD: component scaling and final score",
    ]:
        contents.append(pick(t))
    contents.append(
        frame(
            "NBD versus common baselines",
            image("rerun_all_methods", "4.65cm") + source(),
        )
    )
    contents.append(
        frame(
            "Ablations: keep the declared NBD result",
            image("rerun_ablations", "3.9cm")
            + band(
                r"$B$: bubble fit; $A$: support; $D$: diffusion; $F$: frame. Final NBD stays $B+A+D+F$."
            )
            + source(),
        )
    )
    contents.append(
        frame(
            "All 15 categories, all 11 variants",
            image("rerun_categories", "5.0cm") + source(),
        )
    )
    table_rows = [
        [
            esc(LABELS[m]),
            stat(r.auroc_mean, r.auroc_sd, True),
            stat(r.ap_mean, r.ap_sd, True),
        ]
        for m, r in common.iterrows()
    ]
    contents.append(
        frame(
            "Complete common results",
            r"\footnotesize"
            + table(["Method", "Image AUROC (\\%)", "AP (\\%)"], table_rows)
            + source(),
        )
    )
    cvrows = [
        [esc(r.track), f"{int(r['sum'])}/{int(r['count'])}"]
        for _, r in coverage.iterrows()
    ]
    contents.append(
        frame(
            "Coverage and reproducibility",
            table(["Track", "Completed / planned result groups"], cvrows)
            + bullets(
                [
                    f"Independent image-score audit: {audit['audited_prediction_rows']:,} exported predictions.",
                    r"Pinned sources, configs, split IDs, checkpoints, replay and raw CSVs retained.",
                    r"Original full schedules and historical v2 results remain available.",
                ]
            )
            + source(),
        )
    )
    timing_rows = [
        [
            esc(
                row[0].replace(" (feature adaptation)", "").replace(" (15 epochs)", "")
            ),
            row[1],
            row[3],
        ]
        for row in inference_rows[1:]
    ]
    contents.append(
        frame(
            "Inference: shared CNN and scorer costs",
            r"\footnotesize"
            + table(["Method", "Median ms", "State MiB"], timing_rows)
            + band(
                r"One bottle image, seed 0; batch 1; 30 repeats. NBD calibration arrays are additional; see report."
            )
            + r"\src{CUDA-synchronized after training. Scorer time excludes shared CNN; state is not peak GPU memory.}",
        )
    )
    contents.append(
        frame(
            "References and evidence",
            bullets(
                [
                    r"\href{https://proceedings.mlr.press/v80/ruff18a.html}{Ruff et al.: Deep One-Class Classification (2018).}",
                    r"\href{https://proceedings.mlr.press/v119/goyal20c.html}{Goyal et al.: DROCC (2020).}",
                    r"\href{https://github.com/amazon-science/patchcore-inspection}{Roth et al.: PatchCore (2022), released author code.}",
                    r"Tax and Duin: Support Vector Data Description (2004).",
                    r"Coifman and Lafon: Diffusion Maps (2006).",
                    r"\href{https://github.com/dathuynh1108/OCC}{Code, full tables, handoff and source audit: dathuynh1108/OCC.}",
                ]
            )
            + band(r"Scores describe these declared protocols and training budgets."),
        )
    )
    old = pd.read_csv(ROOT / "results/mvtec-full-v2/summary.csv")
    historical_rows = [
        [
            esc(LABELS[r.method].replace(" (15 epochs)", " (100 epochs)")),
            stat(r.auroc_mean, r.auroc_sd, True),
        ]
        for _, r in old.iterrows()
    ]
    contents.insert(
        -1,
        frame(
            "Appendix: historical v2 shared-feature results",
            r"\footnotesize"
            + table(["Historical method", "Image AUROC (\\%)"], historical_rows)
            + band(
                r"Previous 1,536-D extractor, AE 50 / heads 100 epochs. Preserved evidence; a different protocol."
            )
            + r"\src{\href{https://github.com/dathuynh1108/OCC/tree/91223d63adc3289786bbdeff16d5feeb59aeb3d7/results/mvtec-full-v2}{Historical v2 at 91223d6}. Do not infer a causal feature effect from this comparison.}",
        ),
    )
    (DECK / "review.tex").write_text(
        pre + "\\begin{document}\n" + "\n".join(contents) + "\n\\end{document}\n"
    )
    (DECK / "PROVENANCE.md").write_text(
        "# Measured source-audited revision\n\nGenerated by reproduction/render_delivery.py from independently audited\nresults/reproduction-2026-09-14 CSVs. The supplied source is preserved as\nreview.input.tex and in Git history. Historical v2 figures/data stay available;\ncurrent slides reference only the rerun_*.pdf plots for measured comparisons.\n\nBudget: user requested about two hours and fewer epochs. See run_plan_budgeted.json.\nCompile: tectonic --keep-logs review.tex (or pdflatex on Overleaf).\n"
    )
    print("Generated slides:", len(contents))


if __name__ == "__main__":
    main()
