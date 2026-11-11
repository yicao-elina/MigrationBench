#!/usr/bin/env python3
"""Render the canonical fixed-path and training-overlap figure."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "source_data/grouped_retraining_overlap_data.json"
# GROUPED_OVERLAP_OUT (a file stem) redirects the outputs; default is the committed figures/ copy.
OUT = Path(os.environ.get("GROUPED_OVERLAP_OUT", str(HERE / "figures/grouped_retraining_overlap_palette_updated")))

# Palette and typography follow analysis/canva_palette_vector_bundle_20261105.
COLORS = {
    "Foundation": "#8FC7AB",
    "Fine-tuning": "#29A7B6",
    "Scratch": "#327FB1",
    "DFT": "#303840",
    "MACE-MP": "#2F6B4F",
    "MACE-OMAT-0": "#5F9E73",
    "MPA0": "#8FC7AB",
    "in_gap_heading": "#FABAB9",
    "grid": "#D9E0E4",
}
SEED_MARKERS = {"123": "o", "234": "^", "345": "s"}


def style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(COLORS["DFT"])
        spine.set_linewidth(1.05)
    ax.tick_params(direction="out", width=1.0, length=4.0,
                   color=COLORS["DFT"], labelcolor=COLORS["DFT"], labelsize=11)
    ax.xaxis.label.set_color(COLORS["DFT"])
    ax.yaxis.label.set_color(COLORS["DFT"])


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    data = json.loads(DATA_PATH.read_text())
    panel_a = data["panel_a"]
    panel_b = data["panel_b"]
    trained = panel_a["trained_models"]
    foundations = panel_a["foundation_models"]

    # 7.25 in wide matches the manuscript's single-column figure width closely.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.25, 4.70), gridspec_kw={"width_ratios": [1.48, 0.92]}
    )
    fig.subplots_adjust(left=0.245, right=0.985, bottom=0.15, top=0.76, wspace=0.48)

    # Panel a: aligned horizontal intervals retain individual seed variation
    # while using the approved seed, temperature, family, and checkpoint encodings.
    labels = list(trained) + [f"Foundation ({len(foundations)} models)"]
    y = np.arange(len(labels))
    family = {
        "FT-600K": "Fine-tuning", "FT-MultiT": "Fine-tuning",
        "Scratch-600K": "Scratch", "Scratch-MultiT": "Scratch",
    }
    for i, name in enumerate(trained):
        vals = np.asarray(trained[name], dtype=float)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1))
        color = COLORS[family[name]]
        ax_a.errorbar(mean, y[i], xerr=sd, fmt="D", markersize=5.2,
                      markerfacecolor=color, markeredgecolor=COLORS["DFT"],
                      markeredgewidth=0.6, ecolor=COLORS["DFT"], elinewidth=1.05,
                      capsize=3.0, capthick=1.0, zorder=3)
        offsets = np.array([-0.16, 0.0, 0.16])
        for val, seed, offset in zip(vals, SEED_MARKERS, offsets):
            is_multit = "MultiT" in name
            ax_a.scatter(
                val, y[i] + offset, marker=SEED_MARKERS[seed], s=34,
                facecolor=color if is_multit else "white", edgecolor=color,
                linewidth=1.1, zorder=4,
            )

    foundation_colors = {
        "MP-0B3": COLORS["MACE-MP"],
        "OMAT-0": COLORS["MACE-OMAT-0"],
        "MPA0": COLORS["MPA0"],
    }
    foundation_y = len(trained)
    foundation_offsets = np.array([-0.16, 0.0, 0.16])
    foundation_values = np.asarray(list(foundations.values()), dtype=float)
    foundation_mean = float(foundation_values.mean())
    foundation_sd = float(foundation_values.std(ddof=1))
    ax_a.errorbar(foundation_mean, y[foundation_y], xerr=foundation_sd,
                  fmt="D", markersize=5.5, markerfacecolor=COLORS["Foundation"],
                  markeredgecolor=COLORS["DFT"], markeredgewidth=0.65,
                  ecolor=COLORS["DFT"], elinewidth=1.05, capsize=3.0,
                  capthick=1.0, zorder=3)
    for (name, value), offset in zip(foundations.items(), foundation_offsets):
        ax_a.scatter(value, y[foundation_y] + offset, marker="D", s=31,
                     facecolor=foundation_colors[name], edgecolor=COLORS["DFT"],
                     linewidth=0.55, zorder=4)

    ax_a.set_yticks(y, labels)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 1.32)
    ax_a.set_xlabel("Absolute fixed-path barrier error (eV)", fontsize=12,
                     fontweight="bold", labelpad=7)
    ax_a.set_title("a  Canonical in-gap fixed path", loc="left",
                   color=COLORS["DFT"], fontsize=14,
                   fontweight="bold", pad=9)
    ax_a.xaxis.grid(True, color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.tick_params(axis="y", length=0, pad=6, labelsize=11)

    # Compound keys keep seed/checkpoint semantics concise in the figure legend.
    seed_handle = tuple(
        Line2D([], [], marker=SEED_MARKERS[seed], linestyle="none",
               markerfacecolor="white", markeredgecolor=COLORS["DFT"], markersize=5.0)
        for seed in SEED_MARKERS
    )
    foundation_handle = tuple(
        Line2D([], [], marker="D", linestyle="none", markerfacecolor=foundation_colors[name],
               markeredgecolor=COLORS["DFT"], markersize=5.0)
        for name in foundations
    )
    summary_handle = Line2D(
        [], [], marker="D", linestyle="-", color=COLORS["DFT"],
        markerfacecolor=COLORS["Fine-tuning"], markeredgecolor=COLORS["DFT"],
        markersize=5.2,
    )
    # Panel b: legacy training-set distances normalized by declared duplicate gates.
    thresholds = panel_b["thresholds"]
    minima = panel_b["minimums"]
    splits = list(minima)
    positions = np.arange(len(splits))
    width = 0.34
    soap = [minima[k]["soap_cosine"] / thresholds["soap_cosine"] for k in splits]
    rmsd = [minima[k]["cr_local_rmsd_angstrom"] / thresholds["cr_local_rmsd_angstrom"] for k in splits]
    ax_b.bar(positions - width / 2, soap, width, color=COLORS["Fine-tuning"],
             edgecolor=COLORS["Fine-tuning"], linewidth=0.8, zorder=3)
    ax_b.bar(positions + width / 2, rmsd, width, color=COLORS["Scratch"],
             edgecolor=COLORS["Scratch"], linewidth=0.8, zorder=3)
    ax_b.axhline(1.0, color=COLORS["DFT"], linestyle=(0, (4, 2)),
                 linewidth=1.05, zorder=2)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.8, 180)
    ax_b.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax_b.set_xticks(positions, splits)
    ax_b.set_ylabel("Minimum distance / threshold", fontsize=12,
                    fontweight="bold", labelpad=7)
    ax_b.set_title("b  Legacy train–path overlap", loc="left",
                   color=COLORS["DFT"], fontsize=14, fontweight="bold", pad=9)
    ax_b.yaxis.grid(True, color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.tick_params(axis="x", length=0, pad=6, labelsize=11)
    overlap_handles = [
        Rectangle((0, 0), 1, 1, facecolor=COLORS["Fine-tuning"],
                  edgecolor=COLORS["Fine-tuning"], label="SOAP cosine"),
        Rectangle((0, 0), 1, 1, facecolor=COLORS["Scratch"],
                  edgecolor=COLORS["Scratch"], label="Cr-local RMSD"),
        Line2D([], [], color=COLORS["DFT"], linestyle=(0, (4, 2)),
               label="Threshold"),
    ]
    ax_b.legend(handles=overlap_handles, frameon=False, loc="upper left",
                fontsize=9.2, handletextpad=0.4, labelspacing=0.4,
                borderaxespad=0.4)

    style_axes(ax_a)
    style_axes(ax_b)
    # Restore grid after the common axis styling.
    ax_a.xaxis.grid(True, color=COLORS["grid"], linewidth=0.65, zorder=0)
    ax_b.yaxis.grid(True, color=COLORS["grid"], linewidth=0.65, zorder=0)

    # One aligned figure-level legend for panel a; panel b has a compact local key.
    fig.legend(handles=[seed_handle, summary_handle, foundation_handle],
               labels=["Seeds " + " / ".join(SEED_MARKERS), "Group mean ± sample SD",
                       f"Foundation ({len(foundations)} checkpoints)"],
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)},
               frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=3, fontsize=10, handletextpad=0.5, columnspacing=1.5,
               borderaxespad=0.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=320 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
