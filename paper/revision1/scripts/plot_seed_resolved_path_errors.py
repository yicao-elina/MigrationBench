"""Regenerate SI Fig. S6 with status-neutral fixed-path terminology.

The values below are the three-initialization fixed-image error summaries and
individual observations used by the revision1 SI table.  G1 and Q1--Q3 are
UNCONVERGED QE image sets, so the figure deliberately reports reference errors
instead of physical barrier errors.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "figures/seed_resolved_path_errors.pdf"

MODEL_COLORS = {"FT-600K": "#2c5a92", "Scratch": "#ff9919", "Scratch-5%": "#f4b75b"}
WITHIN_GAP = {
    "FT-600K": ([0.613127273, 0.413227993, 0.396246800], 0.4742006887, 0.1206131715),
    "Scratch": ([1.516379705, 1.036197790, 1.060642031], 1.2044065086, 0.2704530213),
    "Scratch-5%": ([0.651368400, 0.497051880, 0.884486281], 0.6776355201, 0.1950482631),
}
HIGH_DISTORTION = {
    "Scratch": ([0.640300648, 0.473727277, 0.423279728],
                 [0.444415310, 0.371658192, 1.351847853],
                 [2.216669827, 2.476813382, 0.626466305]),
    "FT-600K": ([10.254810636, 9.348785829, 9.192917245],
                 [10.885998623, 10.044561628, 10.255430414],
                 [9.543793667, 9.640671698, 9.929742803]),
}


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.0, 1.35]})

    models = ["FT-600K", "Scratch", "Scratch-5%"]
    means = [WITHIN_GAP[m][1] for m in models]
    sds = [WITHIN_GAP[m][2] for m in models]
    x = np.arange(len(models))
    ax_a.bar(x, means, yerr=sds, capsize=5, width=0.62,
             color=[MODEL_COLORS[m] for m in models], edgecolor="#666666", linewidth=0.7)
    for i, model in enumerate(models):
        jitter = np.linspace(-0.08, 0.08, 3)
        ax_a.scatter(i + jitter, WITHIN_GAP[model][0], s=48, facecolors="white",
                     edgecolors="#666666", linewidths=1.0, zorder=3)
    ax_a.axhline(0.16, color="#777777", linestyle="--", linewidth=1.2)
    ax_a.set_xticks(x, ["FT-600K", "Scratch", "Scratch-5%"], rotation=28, ha="right")
    ax_a.set_ylim(0, 1.65)
    ax_a.set_ylabel("Within-gap fixed-path reference error (eV)")
    ax_a.set_title("(a) Grouped within-gap image set")

    y = np.array([2, 1, 0])
    path_labels = ["Q1", "Q2", "Q3"]
    for model, offset in [("Scratch", 0.08), ("FT-600K", -0.08)]:
        values = np.array([np.mean(v) for v in HIGH_DISTORTION[model]])
        errors = np.array([np.std(v, ddof=1) for v in HIGH_DISTORTION[model]])
        ax_b.errorbar(values, y + offset, xerr=errors, fmt="o", capsize=4,
                      color=MODEL_COLORS[model], label=model, linewidth=1.5, markersize=7)
    ax_b.set_yticks(y, path_labels)
    ax_b.set_xlim(0, 11.5)
    ax_b.set_xlabel("Absolute fixed-path reference error (eV)")
    ax_b.set_title("(b) Gap-to-QL / into-QL image sets")
    ax_b.grid(axis="x", color="#dddddd", linewidth=0.8)
    ax_b.set_axisbelow(True)
    ax_b.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
