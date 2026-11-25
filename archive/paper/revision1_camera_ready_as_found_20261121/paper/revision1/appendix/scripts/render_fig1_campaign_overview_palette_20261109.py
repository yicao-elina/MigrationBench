#!/usr/bin/env python3
"""Re-render the endpoint campaign overview using the 2026-11-09 figure palette.

Source of truth: ../data/endpoint_level_summary.csv (846 audited task rows).
Run from any directory with matplotlib, numpy, and pandas installed:
    python render_fig1_campaign_overview_palette_20261109.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
APPENDIX = HERE.parent
DATA = APPENDIX / "data" / "endpoint_level_summary.csv"
OUT = APPENDIX / "figures" / "fig1_campaign_overview"
ROOT = HERE.parents[5]
PALETTE_PATH = ROOT / "analysis/canva_palette_vector_bundle_20261105/MIGRATIONBENCH_FIGURE_PALETTE.json"
PALETTE = json.loads(PALETTE_PATH.read_text())

SYSTEMS = ["Cr@Sb2Te3", "V@Sb2Te3", "Mn@Sb2Te3", "Cr@Bi2Te3", "V@Bi2Te3", "Mn@Bi2Te3"]
SYSTEM_LABELS = ["Cr@Sb2Te3", "V@Sb2Te3", "Mn@Sb2Te3", "Cr@Bi2Te3", "V@Bi2Te3", "Mn@Bi2Te3"]
ARMS = ["base", "perturb_a_dx020", "perturb_b_tilt020"]
ARM_LABELS = {"base": "Base", "perturb_a_dx020": "Perturb A", "perturb_b_tilt020": "Perturb B"}
INK = PALETTE["model_families"]["DFT_reference"]
ARM_COLORS = {
    "base": PALETTE["campaign_task_types"]["Base"]["color"],
    "perturb_a_dx020": PALETTE["campaign_task_types"]["Perturb A"]["color"],
    "perturb_b_tilt020": PALETTE["campaign_task_types"]["Perturb B"]["color"],
}
ARM_TEXT_COLORS = {
    "base": "#B65471",
    "perturb_a_dx020": "#C94381",
    "perturb_b_tilt020": ARM_COLORS["perturb_b_tilt020"],
}
MARKERS = {"base": "o", "perturb_a_dx020": "^", "perturb_b_tilt020": "s"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": PALETTE["typography"]["font_family"],
    "font.size": 14.0,
    "axes.titlesize": 18.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 17.0,
    "xtick.labelsize": 13.2,
    "ytick.labelsize": 13.2,
    "legend.fontsize": 13.0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.9,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.facecolor": "white",
})


def panel_label(ax, letter: str) -> None:
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=18.5, fontweight="bold", color=INK, clip_on=False)


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#E7ECEF", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", length=3.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(0.8)


def draw_box_groups(ax, df: pd.DataFrame, column: str, ylabel: str, title: str, letter: str,
                    ylim=None, log=False) -> None:
    positions, values, colors = [], [], []
    for si, system in enumerate(SYSTEMS):
        for ai, arm in enumerate(ARMS):
            vals = df.loc[(df.system == system) & (df.arm == arm), column].to_numpy(float)
            positions.append(si * 4 + ai)
            values.append(vals)
            colors.append(ARM_COLORS[arm])
    bp = ax.boxplot(values, positions=positions, widths=0.62, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.25},
                    whiskerprops={"color": INK, "linewidth": 0.85},
                    capprops={"color": INK, "linewidth": 0.85},
                    boxprops={"edgecolor": INK, "linewidth": 0.75})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.26)
    for pos, vals, color in zip(positions, values, colors):
        rng = np.random.default_rng(pos + 5)
        ax.scatter(rng.normal(pos, 0.055, len(vals)), vals, s=8, color=color, alpha=0.32,
                   edgecolors="none", rasterized=False, zorder=2)
    ax.set_xticks([si * 4 + 1 for si in range(6)], SYSTEM_LABELS, rotation=32, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    if log:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    style_axis(ax)
    panel_label(ax, letter)


def main() -> None:
    df = pd.read_csv(DATA)
    if len(df) != 846 or set(df.arm) != set(ARMS) or set(df.system) != set(SYSTEMS):
        raise ValueError("The checked-in endpoint table does not match the expected 846-task campaign.")
    if not df.scientific_success.astype(bool).all():
        raise ValueError("The overview renderer expects the audited 846/846 success result.")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    draw_box_groups(axes[0, 0], df, "force", r"Final $F_{\max}$ (eV/Å)",
                    "Force after relaxation", "A", ylim=(0.0095, 0.031))
    draw_box_groups(axes[0, 1], df, "disp", "Maximum displacement (Å)",
                    "Geometric response", "B", ylim=(-0.15, 4.5))
    draw_box_groups(axes[1, 0], df, "steps", "FIRE steps", "Relaxation effort", "C")

    ax = axes[1, 1]
    for arm in ARMS:
        sub = df[df.arm == arm]
        ax.scatter(sub.initial_force, sub.disp, s=21, marker=MARKERS[arm],
                   color=ARM_COLORS[arm], alpha=0.62, label=ARM_LABELS[arm],
                   edgecolors="white", linewidths=0.25, rasterized=False)
    ax.set_xlabel(r"Initial $F_{\max}$ (eV/Å)")
    ax.set_ylabel("Maximum displacement (Å)")
    ax.set_title("Force–geometry relation", pad=8)
    ax.legend(frameon=False, fontsize=12.5, loc="upper right", handletextpad=0.5)
    style_axis(ax)
    panel_label(ax, "D")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT.with_suffix(f".{suffix}"), dpi=320 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    print(f"Rendered {OUT.with_suffix('.pdf')} and {OUT.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
