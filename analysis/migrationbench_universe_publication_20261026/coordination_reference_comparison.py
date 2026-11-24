#!/usr/bin/env python3
"""Compose a reference-versus-distortion panel for representative Cr environments.

The images are the real Blender renders from the audited representative frames.
The quantitative comparison uses the same 3.75 Å Cr coordination shell and the
same paired reference-shell convention as ``coordination_distortion.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"
RENDERS = HERE / "blender" / "cluster_renders"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def make_figure() -> None:
    configure_style()
    reps = pd.read_csv(DERIVED / "coordination_distortion.csv")

    # This is the Cr-containing endpoint reference used to define the common
    # local shell. It is not an undoped pristine Sb2Te3 structure: an undoped
    # cell has no Cr centre and therefore cannot define a Cr coordination shell.
    reference = {
        "cluster": "PERFECT",
        "cluster_color": "#202020",
        "source_label": "Cr-host endpoint reference",
        "coordination_number": 8,
        "mean_bond_A": 2.8389565390541422,
        "bond_length_std_A": 0.11653147880542937,
        "mean_abs_delta_A": 0.0,
        "rms_delta_A": 0.0,
        "coordination_count_delta": 0,
    }

    selected = [
        reference,
        {**reps[reps["cluster"].eq("C10")].iloc[0].to_dict(), "source_label": "FT-600K self-NEB"},
        {**reps[reps["cluster"].eq("C03")].iloc[0].to_dict(), "source_label": "grouped valid reference"},
        {**reps[reps["cluster"].eq("C09")].iloc[0].to_dict(), "source_label": "grouped train reference"},
    ]

    fig = plt.figure(figsize=(15.5, 8.7), dpi=180, facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 0.64],
        hspace=0.38,
        wspace=0.24,
        left=0.055,
        right=0.965,
        top=0.875,
        bottom=0.15,
    )

    for col, item in enumerate(selected):
        ax = fig.add_subplot(grid[0, col])
        cluster = str(item["cluster"])
        image_path = (
            RENDERS
            / ("PERFECT_REFERENCE" if cluster == "PERFECT" else cluster)
            / "opaque_rgb"
            / "cr_local_environment.png"
        )
        if image_path.exists():
            ax.imshow(mpimg.imread(image_path))
        else:
            ax.text(0.5, 0.5, "render missing", ha="center", va="center", color="#777777")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.0)
            spine.set_edgecolor(str(item["cluster_color"]))
        if cluster == "PERFECT":
            title = "Perfect reference\nCr-host endpoint"
        else:
            title = f"{cluster} · {item['source_label']}"
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=5)

    bar_ax = fig.add_subplot(grid[1, 0:2])
    labels = ["Perfect\nreference", "C10\nFT-600K self-NEB", "C03\nvalid reference", "C09\ntrain reference"]
    values = [float(item["rms_delta_A"]) for item in selected]
    colors = [str(item["cluster_color"]) for item in selected]
    y = np.arange(len(labels))
    bars = bar_ax.barh(y, values, color=colors, alpha=0.86, edgecolor="#333333", linewidth=0.5)
    bar_ax.set_yticks(y, labels, fontsize=8)
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("RMS paired bond-length deviation from reference (Å)")
    bar_ax.set_title("Bonding distortion", loc="left", fontweight="bold", pad=5)
    bar_ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    bar_ax.set_axisbelow(True)
    bar_ax.set_xlim(0, max(values) * 1.27 + 0.02)
    for bar, item in zip(bars, selected):
        text = (
            f"CN {int(item['coordination_number'])}  "
            f"⟨r⟩ {fmt(float(item['mean_bond_A']))} Å  "
            f"σr {fmt(float(item['bond_length_std_A']))} Å"
        )
        bar_ax.text(
            bar.get_width() + 0.008,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            fontsize=7.2,
            color="#222222",
        )
    for spine in bar_ax.spines.values():
        spine.set_color("#333333")

    table_ax = fig.add_subplot(grid[1, 2:4])
    table_ax.axis("off")
    table_ax.set_title("Interpretation of the same coordination shell", loc="left", fontweight="bold", pad=5)
    table_lines = [
        "Perfect reference: 8 neighbours (2 Sb + 6 Te), narrow shell.",
        "C10: one fewer neighbour and a compressed, low-spread shell along the self-NEB frame.",
        "C03/C09: reference-set environments, but with larger shell spread or coordination loss.",
        "Cylinders are distance-qualified coordination guides; no bond order is inferred.",
    ]
    y0 = 0.88
    for line in table_lines:
        table_ax.text(0.02, y0, "• " + line, transform=table_ax.transAxes, va="top", fontsize=8.2, color="#252525", wrap=True)
        y0 -= 0.22
    table_ax.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=table_ax.transAxes, fill=False, edgecolor="#BDBDBD", linewidth=0.8))

    fig.suptitle(
        "Coordination-sphere fingerprints of representative MigrationBench configurations",
        fontsize=15,
        fontweight="bold",
        y=0.945,
    )
    fig.text(
        0.5,
        0.055,
        "All panels use the same Cr-centred 3.75 Å shell. The reference is a Cr-containing host endpoint; an undoped pristine host has no Cr centre. "
        "RMS Δr is a structural descriptor, not an energy barrier or a model-ranking score.",
        ha="center",
        va="bottom",
        fontsize=7.4,
        color="#444444",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"coordination_reference_comparison.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
