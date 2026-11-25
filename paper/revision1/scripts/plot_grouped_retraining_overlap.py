"""Regenerate the grouped-retraining figure with a neutral G1 label."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "figures/grouped_retraining_overlap.pdf"
COLORS = ["#2c5a92", "#ff9919", "#6fa8dc"]
SEED_VALUES = [
    [0.613127, 0.413228, 0.396247],
    [1.516380, 1.036198, 1.060642],
    [0.651368, 0.497052, 0.884486],
]
SOAP_RATIOS = [0.000542785133204049 / 0.0001, 0.0006374028848443736 / 0.0001]
RMSD_RATIOS = [2.055065268285385 / 0.05, 2.056638391614761 / 0.05]


def main() -> None:
    means = np.array([np.mean(values) for values in SEED_VALUES])
    sds = np.array([np.std(values, ddof=1) for values in SEED_VALUES])
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.25, 4.75), gridspec_kw={"width_ratios": [1.2, 1.0]})
    x = np.arange(3)
    ax_a.bar(x, means, yerr=sds, capsize=5, width=0.62, color=COLORS, edgecolor="#3a3a3a", linewidth=0.7)
    for i, values in enumerate(SEED_VALUES):
        ax_a.scatter(i + np.linspace(-0.08, 0.08, 3), values, s=30, facecolors="white",
                     edgecolors="#666666", linewidths=0.9, zorder=3)
    ax_a.axhline(0.160742, color="#555555", linestyle=(0, (4, 2)), linewidth=1.1)
    ax_a.scatter([3], [0.237235], marker="D", s=30, facecolor="#d75a2a", edgecolor="#222222",
                 linewidth=1.0, zorder=4)
    ax_a.set_xticks([0, 1, 2, 3], ["OMAT-FT", "Scratch", "Scratch-5%", "MPA0\n(single seed)"])
    ax_a.set_xlim(-0.55, 3.55)
    ax_a.set_ylim(0, 1.8)
    ax_a.set_xlabel("Model", fontweight="bold")
    ax_a.set_ylabel("Within-gap fixed-path reference error (eV)", fontweight="bold")
    ax_a.legend(handles=[
        plt.Line2D([0], [0], color="#555555", linestyle=(0, (4, 2)), label="Historical frame-level 0.16 eV"),
        plt.Line2D([0], [0], marker="D", color="#d75a2a", markerfacecolor="#d75a2a",
                   markeredgecolor="#222222", markersize=6, linestyle="None", label="MPA0 single seed"),
    ], frameon=False, loc="upper left", fontsize=8)
    ax_a.text(-0.12, 1.04, "a", transform=ax_a.transAxes, fontsize=16, fontweight="bold", va="bottom")

    split_x = np.arange(2)
    width = 0.36
    ax_b.bar(split_x - width / 2, SOAP_RATIOS, width, color="#6fa8dc", label="SOAP cosine")
    ax_b.bar(split_x + width / 2, RMSD_RATIOS, width, color="#d75a2a", label="Cr-local RMSD")
    ax_b.axhline(1.0, color="#555555", linestyle=(0, (4, 2)), linewidth=1.3)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.8, 100)
    ax_b.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax_b.set_xticks(split_x, ["Full grouped", "Scratch-5%"])
    ax_b.set_xlabel("Training split", fontweight="bold")
    ax_b.set_ylabel("Distance / declared near-duplicate gate", fontweight="bold")
    ax_b.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=8)
    ax_b.text(-0.12, 1.04, "b", transform=ax_b.transAxes, fontsize=16, fontweight="bold", va="bottom")

    for ax in (ax_a, ax_b):
        ax.tick_params(axis="both", which="major", width=1.1, length=5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for label in ax_a.get_xticklabels():
        label.set_rotation(25)
        label.set_horizontalalignment("right")
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.31, top=0.86, wspace=0.42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
