"""Render the reader-facing NEB error overview for the revised main text.

The values are the grouped three-initialization summaries reported in
``tables/scratch5_path_resolved_descriptive.tex`` and the four zero-shot
Foundation checkpoints reported in the response-aligned benchmark tables.
Q1--Q3 are not converged QE NEB paths, so the y-axis uses fixed-path
energy-assignment error rather than a physical barrier label.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
# Canonical manuscript role: fixed-path error overview, typeset as main Fig. 3.
# The legacy fig4_* filename is retained as a provenance copy, but is not the
# manuscript input so that the asset name does not imply the PDF figure number.
OUT = ROOT / "paper/revision1/figures/fig_fixed_path_errors.pdf"


def main() -> None:
    # Values transcribed from revision1/tables/scratch5_path_resolved_descriptive.tex.
    g1 = {
        "FT-600K": (0.47, 0.12),
        "Scratch-5%": (0.68, 0.20),
        "Scratch": (1.20, 0.27),
    }
    foundation_g1 = np.asarray([1.50 - 0.336, 1.79 - 0.336,
                                1.60 - 0.336, 1.84 - 0.336])
    q_paths = ["Q1", "Q2", "Q3"]
    q_ft = np.asarray([9.60, 10.40, 9.70])
    q_ft_sd = np.asarray([0.57, 0.44, 0.20])
    q_scratch = np.asarray([0.51, 0.72, 1.77])
    q_scratch_sd = np.asarray([0.11, 0.55, 1.00])

    colors = {
        "FT-600K": "#2F5597",
        "Scratch-5%": "#7A7A7A",
        "Scratch": "#E69F00",
        "Foundation": "#B2182B",
    }

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), sharey=False)
    ax = axes[0]
    labels = list(g1)
    x = np.arange(len(labels), dtype=float)
    for i, label in enumerate(labels):
        mean, sd = g1[label]
        ax.errorbar(x[i], mean, yerr=sd, fmt="o", ms=7, capsize=4,
                    lw=1.6, color=colors[label], ecolor=colors[label],
                    label=label, zorder=3)
    fx = np.full(foundation_g1.shape, len(labels), dtype=float)
    fx += np.asarray([-0.12, -0.04, 0.04, 0.12])
    ax.scatter(fx, foundation_g1, s=38, marker="D",
               color=colors["Foundation"], edgecolor="white", linewidth=0.6,
               label="Foundation (4 checkpoints)", zorder=4)
    ax.set_xticks(np.arange(4), labels + ["Foundation"])
    ax.set_ylabel("G1 absolute fixed-path error (eV)")
    ax.set_ylim(0, 1.75)
    ax.set_title("Within-gap reference path", fontsize=12, fontweight="bold")
    ax.text(-0.12, 1.03, "a", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.00))

    ax = axes[1]
    x = np.arange(len(q_paths), dtype=float)
    width = 0.18
    ax.errorbar(x - width, q_ft, yerr=q_ft_sd, fmt="o", ms=6, capsize=3,
                lw=1.5, color=colors["FT-600K"], ecolor=colors["FT-600K"],
                label="FT-600K", zorder=3)
    ax.errorbar(x + width, q_scratch, yerr=q_scratch_sd, fmt="o", ms=6,
                capsize=3, lw=1.5, color=colors["Scratch"],
                ecolor=colors["Scratch"], label="Scratch", zorder=3)
    ax.set_xticks(x, q_paths)
    ax.set_ylabel("Absolute fixed-path error (eV)")
    ax.set_ylim(0, 11.5)
    ax.set_title("Higher-distortion image sets", fontsize=12,
                 fontweight="bold")
    ax.text(-0.12, 1.03, "b", transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.legend(frameon=False, loc="upper left")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.1)
        ax.spines["bottom"].set_linewidth(1.1)
        ax.tick_params(width=1.0, length=4)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.82,
                        wspace=0.34)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
