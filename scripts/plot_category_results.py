"""Render a publication heatmap with the full AUROC scale, without changing data."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(directory):
    source = directory / "category_summary.csv"
    methods = pd.read_csv(directory / "summary.csv").method.tolist()
    heat = (
        pd.read_csv(source)
        .pivot(index="category", columns="method", values="auroc_mean")
        .reindex(columns=methods)
    )
    fig, ax = plt.subplots(figsize=(13, 7.5))
    plot = ax.imshow(heat.values * 100, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Full MVTec AD · category mean across 3 seeds · image AUROC")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j] * 100
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < 60 else "black",
            )
    fig.colorbar(plot, ax=ax, label="Image AUROC (%)", ticks=range(0, 101, 10))
    fig.tight_layout()
    output = directory / "plots/category_auroc_full_scale.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    metadata = {
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "display_range_percent": [0, 100],
        "score_values_changed": False,
        "reason": "Show below-chance category scores without color saturation at 50%.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="results/mvtec-full-v2")
    main(Path(parser.parse_args().directory))
