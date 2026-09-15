"""Create an auditable report for the completed full MVTec image-model run."""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from .common import ROOT, sha256, write_json
from .mvtec_image_baselines import FULL_OUTPUT, OUTPUT


def csv_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def result(root, category, relative):
    return json.loads((root / category / relative / "result.json").read_text())


def complete_seconds(root, category, method):
    return float(json.loads((root / category / method / "COMPLETE.json").read_text())["seconds"])


def macro(values, key):
    return mean(float(value[key]) for value in values)


def percent(value):
    return f"{100 * value:.2f}"


def finite_history(path, epochs, fields):
    history = json.loads(path.read_text())
    assert len(history) == epochs, (path, len(history), epochs)
    for row in history:
        for field in fields:
            assert math.isfinite(float(row[field])), (path, field, row[field])
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FULL_OUTPUT)
    parser.add_argument("--light-output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output, light = args.output, args.light_output

    protocol = json.loads((output / "protocol.json").read_text())
    verification = json.loads((output / "comparison_verification.json").read_text())
    assert protocol["schedule"] == "full"
    assert verification["passed"] and verification["drocc_primary_selection"] == "fixed_final_epoch"
    assert protocol["deep"]["ae_settings"]["epochs"] == 350
    assert protocol["deep"]["svdd_epochs"] == 150
    assert protocol["drocc"]["epochs"] == 100

    comparison = csv_rows(output / "comparison_per_category.csv")
    categories = sorted({row["category"] for row in comparison})
    assert len(categories) == 15
    by_key = {(row["method"], row["category"]): row for row in comparison}
    assert len(by_key) == 120

    deep_one, deep_soft, drocc_final, drocc_selected = [], [], [], []
    light_one, light_soft, light_drocc_final = [], [], []
    timings, diagnostics, category_rows = [], [], []
    for category in categories:
        one = result(output, category, "deep/one-class")
        soft = result(output, category, "deep/soft-boundary")
        final = result(output, category, "drocc/fixed_final_epoch")
        selected = result(output, category, "drocc/author_test_selected")
        assert one["selection"] == soft["selection"] == final["selection"] == "fixed_final_epoch"
        assert selected["selection"] == "author_test_selected"
        assert by_key["DeepSVDD_one-class", category]["auroc"] == str(one["auroc"])
        assert by_key["DeepSVDD_soft-boundary", category]["auroc"] == str(soft["auroc"])
        assert by_key["DROCC", category]["auroc"] == str(final["auroc"])
        deep_one.append(one)
        deep_soft.append(soft)
        drocc_final.append(final)
        drocc_selected.append(selected)
        light_one.append(result(light, category, "deep/one-class"))
        light_soft.append(result(light, category, "deep/soft-boundary"))
        light_drocc_final.append(result(light, category, "drocc/fixed_final_epoch"))

        deep_seconds = complete_seconds(output, category, "deep")
        drocc_seconds = complete_seconds(output, category, "drocc")
        timings.append((category, deep_seconds, drocc_seconds))
        ae_history = finite_history(output / category / "deep/ae/history.json", 350, ("loss", "gradient_norm", "lr"))
        one_history = finite_history(output / category / "deep/one-class/history.json", 150, ("loss", "gradient_norm", "lr"))
        soft_history = finite_history(output / category / "deep/soft-boundary/history.json", 150, ("loss", "gradient_norm", "lr", "radius"))
        drocc_history = finite_history(output / category / "drocc/history.json", 100, ("ce_loss", "adv_loss", "gradient_norm", "lr"))
        diagnostics.append({
            "category": category,
            "ae_final_loss": ae_history[-1]["loss"],
            "one_class_final_loss": one_history[-1]["loss"],
            "soft_boundary_final_loss": soft_history[-1]["loss"],
            "soft_boundary_final_radius": soft_history[-1]["radius"],
            "drocc_final_ce_loss": drocc_history[-1]["ce_loss"],
            "drocc_final_adv_loss": drocc_history[-1]["adv_loss"],
            "drocc_best_test_epoch": selected["selected_epoch"],
        })
        category_rows.append({
            "category": category,
            "one": one,
            "soft": soft,
            "final": final,
            "selected": selected,
            "deep_seconds": deep_seconds,
            "drocc_seconds": drocc_seconds,
        })

    metrics = [
        ("DeepSVDD one-class", deep_one, light_one),
        ("DeepSVDD soft-boundary", deep_soft, light_soft),
        ("DROCC fixed-final", drocc_final, light_drocc_final),
    ]
    summary_rows = []
    for label, full_values, light_values in metrics:
        summary_rows.append({
            "label": label,
            "full_auroc": macro(full_values, "auroc"),
            "full_ap": macro(full_values, "average_precision"),
            "light_auroc": macro(light_values, "auroc"),
            "light_ap": macro(light_values, "average_precision"),
        })

    selected_auroc = macro(drocc_selected, "auroc")
    selected_ap = macro(drocc_selected, "average_precision")
    final_auroc = macro(drocc_final, "auroc")
    final_ap = macro(drocc_final, "average_precision")
    total_deep_seconds = sum(value[1] for value in timings)
    total_drocc_seconds = sum(value[2] for value in timings)

    lines = [
        "# MVTec AD: full image-model adaptation",
        "",
        "All 15 MVTec AD categories, 3,629 normal training images and 1,725 test images; seed 0.",
        "Deep SVDD and DROCC are author-image-model adaptations to MVTec, not published MVTec benchmarks from those papers.",
        "",
        "## Locked full schedules",
        "",
        "- Deep SVDD: unchanged MVTec preprocessing (per-image L1 GCN, normal-train scalar min/max), CIFAR CNN/AE, AE 350 epochs with milestone 250, then 150 SVDD epochs with milestone 50; both objectives; fixed-final metric.",
        "- DROCC: unchanged author-CIFAR normalization and MVTec adaptation parameters (Adam 0.001, batch 128, radius 0.2, gamma 2, mu 1, effective 50 ascent steps, projection every 10, no CE-only warmup); 100 epochs; source LR thresholds at 40% and 80%; fixed-final metric is primary.",
        "- RBF SVDD has no epoch budget and is reused from the verified identical light export. PatchCore and NBD predictions are reused unchanged.",
        "",
        "## Full versus light (category-macro image metrics, %)",
        "",
        "| Method | Light AUROC | Full AUROC | Delta pp | Light AP | Full AP | Delta pp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {percent(row['light_auroc'])} | {percent(row['full_auroc'])} | "
            f"{100 * (row['full_auroc'] - row['light_auroc']):+.2f} | {percent(row['light_ap'])} | "
            f"{percent(row['full_ap'])} | {100 * (row['full_ap'] - row['light_ap']):+.2f} |"
        )
    lines += [
        "",
        "## DROCC selection export (full schedule)",
        "",
        "| Export | AUROC | AP |",
        "| --- | ---: | ---: |",
        f"| Fixed final epoch (comparison) | {percent(final_auroc)} | {percent(final_ap)} |",
        f"| Source test-selected diagnostic | {percent(selected_auroc)} | {percent(selected_ap)} |",
        f"| Diagnostic minus fixed-final (pp) | {100 * (selected_auroc - final_auroc):+.2f} | {100 * (selected_ap - final_ap):+.2f} |",
        "",
        "## Per-category full results (AUROC / AP, %)",
        "",
        "| Category | Deep one-class | Deep soft-boundary | DROCC fixed-final | DROCC test-selected diagnostic | Deep s | DROCC s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in category_rows:
        value = lambda result: f"{percent(result['auroc'])} / {percent(result['average_precision'])}"
        lines.append(
            f"| {row['category']} | {value(row['one'])} | {value(row['soft'])} | {value(row['final'])} | "
            f"{value(row['selected'])} | {row['deep_seconds']:.1f} | {row['drocc_seconds']:.1f} |"
        )
    lines += [
        "",
        "## Runtime and convergence checks",
        "",
        f"- Deep SVDD wall time: {total_deep_seconds:.1f} s total, {total_deep_seconds / 15:.1f} s/category (AE shared by both objectives).",
        f"- DROCC wall time: {total_drocc_seconds:.1f} s total, {total_drocc_seconds / 15:.1f} s/category.",
        "- Every 350-epoch AE, 150-epoch one-class/soft-boundary phase, and 100-epoch DROCC history has the required length and finite tracked loss, gradient, LR and radius values.",
        "- Every saved final and source-selected DROCC checkpoint was reloaded for score replay before its result was admitted. `comparison_verification.json` independently recomputes AUROC/AP and checks all test IDs and labels against the PatchCore reference.",
        "- The first fresh Deep SVDD fixture attempt is retained under `evidence/deep_fixture`; it stopped on a relative-output-path logging error before benchmark training. The runner was corrected to resolve the explicit fixture path, then the separate `evidence/deep_fixture_v2` fixture passed. No metric was reused from the failed fixture.",
        "",
        "## Provenance",
        "",
        f"- Full protocol SHA256: `{sha256(output / 'protocol.json')}`",
        f"- Prediction/metric verification SHA256: `{sha256(output / 'comparison_verification.json')}`",
        f"- Input cache manifest SHA256: `{protocol['data_manifest_sha256']}`",
        f"- Source lock SHA256: `{protocol['source_lock_sha256']}`",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    write_json(output / "report_manifest.json", {
        "passed": True,
        "full_protocol_sha256": sha256(output / "protocol.json"),
        "report_sha256": sha256(output / "REPORT.md"),
        "categories": categories,
        "deep_total_seconds": total_deep_seconds,
        "drocc_total_seconds": total_drocc_seconds,
        "full_summary": summary_rows,
        "drocc_fixed_final": {"auroc": final_auroc, "average_precision": final_ap},
        "drocc_test_selected_diagnostic": {"auroc": selected_auroc, "average_precision": selected_ap},
        "diagnostics": diagnostics,
    })
    print(output / "REPORT.md")


if __name__ == "__main__":
    main()
