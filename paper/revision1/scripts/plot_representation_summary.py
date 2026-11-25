#!/usr/bin/env python3
"""Create the reader-facing representation summary from revision1 CSV inputs.

The figure intentionally reports descriptive silhouette summaries only.  It does
not label the projections as mechanisms, latent separability, or acquisition
rules.  All inputs and outputs are local to paper/revision1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "figures" / "fig_representation_summary.pdf"
SILHOUETTE = ROOT / "data_processed" / "latent_silhouette_all_spaces.csv"
SENSITIVITY = ROOT / "data_processed" / "latent_embedding_sensitivity.csv"


def _projection_summary(sensitivity: pd.DataFrame) -> tuple[float, float, float, float]:
    """Return mean/std for t-SNE settings and PHATE nominal seeds."""
    tsne_by_perplexity = (
        sensitivity.loc[sensitivity["method"].eq("tsne")]
        .groupby("perplexity", as_index=False)["overall"]
        .mean()
    )
    tsne_mean = float(tsne_by_perplexity["overall"].mean())
    tsne_sd = float(tsne_by_perplexity["overall"].std(ddof=1))
    phate = sensitivity.loc[sensitivity["method"].eq("phate"), "overall"]
    phate_mean = float(phate.mean())
    phate_sd = float(phate.std(ddof=1))
    return tsne_mean, tsne_sd, phate_mean, phate_sd


def make_figure(output: Path) -> None:
    silhouette = pd.read_csv(SILHOUETTE)
    sensitivity = pd.read_csv(SENSITIVITY)

    space_rows = [
        ("Raw Euclidean", "original", "euclidean"),
        ("Cosine", "original", "cosine"),
        ("z-scored Euclidean", "original_zscore", "euclidean"),
        ("PCA 95%\n(z-scored)", "pca95_zscore", "euclidean"),
    ]
    space_values = [
        float(
            silhouette.loc[
                silhouette["space"].eq(space) & silhouette["metric"].eq(metric),
                "overall",
            ].iloc[0]
        )
        for _, space, metric in space_rows
    ]
    tsne_mean, tsne_sd, phate_mean, phate_sd = _projection_summary(sensitivity)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.78, wspace=0.38)

    ax = axes[0]
    x = np.arange(len(space_rows))
    colors = ["#355C7D", "#6C9FB8", "#9BB7C8", "#C7D6DE"]
    ax.bar(x, space_values, color=colors, width=0.68, edgecolor="#23313A", linewidth=0.4)
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.set_xticks(x, [label for label, _, _ in space_rows])
    ax.set_ylabel("Overall silhouette")
    ax.set_title("Original descriptor spaces", loc="left", fontweight="bold")
    ax.set_ylim(-0.10, 0.40)
    ax.grid(axis="y", color="#D9E0E4", linewidth=0.5)
    ax.set_axisbelow(True)
    for xpos, value in zip(x, space_values):
        ax.text(xpos, value + (0.018 if value >= 0 else -0.035), f"{value:.2f}", ha="center", va="center", fontsize=8)

    ax = axes[1]
    labels = ["t-SNE\n(p=5,30,50)", "PHATE"]
    means = [tsne_mean, phate_mean]
    errors = [tsne_sd, phate_sd]
    x = np.arange(2)
    ax.errorbar(
        x,
        means,
        yerr=errors,
        fmt="o",
        color="#9A4E3D",
        ecolor="#9A4E3D",
        elinewidth=1.2,
        capsize=3,
        markersize=5,
    )
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Overall silhouette")
    ax.set_title("Projection summaries", loc="left", fontweight="bold")
    ax.set_ylim(-0.05, 0.56)
    ax.grid(axis="y", color="#D9E0E4", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(0, means[0] + errors[0] + 0.035, f"{means[0]:.2f} $\u00b1$ {errors[0]:.2f}", ha="center", fontsize=8)
    ax.text(1, means[1] + errors[1] + 0.035, f"{means[1]:.2f} $\u00b1$ {errors[1]:.2f}", ha="center", fontsize=8)

    fig.suptitle(
        "Representation diagnostics: descriptive separation summaries",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.04,
        "Projection scores are qualitative baselines; they do not establish a causal mechanism or an acquisition rule.",
        ha="center",
        fontsize=8,
        color="#3F4B52",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    make_figure(args.output)


if __name__ == "__main__":
    main()
