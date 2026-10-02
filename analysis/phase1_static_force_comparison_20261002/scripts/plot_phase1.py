#!/usr/bin/env python3
"""Publication-style Phase 1 figures using the local Scientific Plot Atlas grammar."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np

PLOT_LIBRARY = Path("/Users/alina/Documents/Codex/plot_library")
if (PLOT_LIBRARY / "src").exists():
    sys.path.insert(0, str(PLOT_LIBRARY / "src"))
try:
    from plot_atlas import apply_publication_style, get_palette, save_figure
    from plot_atlas.style import bold_tick_labels
except Exception:
    def apply_publication_style(font_size=11):
        plt.rcParams.update({"font.family": "Arial", "font.size": font_size, "axes.linewidth": 1.5})
    def save_figure(fig, path):
        fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    def get_palette(_name):
        return {"categorical": ["#0072B2", "#009E73", "#CC79A7", "#D55E00"], "sequential": ["#F7FBFF", "#6BAED6", "#08306B"], "neutral": "#777777"}
    def bold_tick_labels(ax):
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontweight("bold")


ATLAS_PALETTE = get_palette("nature_gdml_accuracy")
ATLAS_SEQUENTIAL = get_palette("descriptor_reduction")["sequential"]
COLORS = dict(zip(("from_scratch", "naive_ft", "multi_t", "foundation"), ATLAS_PALETTE["categorical"][:4]))
COLORS["ensemble"] = ATLAS_PALETTE["neutral"]
LABELS = {"foundation": "Foundation", "naive_ft": "FT-600K", "multi_t": "FT-MultiT", "from_scratch": "Scratch", "ensemble": "4-model mean"}


def style_axis(ax):
    """Apply the Plot Atlas house grammar to a standalone quantitative axis."""
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#222222")
    ax.tick_params(width=1.0, length=4.5, pad=3)
    bold_tick_labels(ax)


def style_legend(legend):
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_edgecolor("#CFCFCF")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.96)


def step_formatter(value, _position=None):
    return f"{int(value):,}"


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, out, stem):
    for ext in ("png", "pdf", "svg"):
        save_figure(fig, out / f"{stem}.{ext}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style(8.8)
    plt.rcParams.update({"font.family": "Arial", "axes.linewidth": 1.1, "xtick.major.width": 1.0, "ytick.major.width": 1.0, "axes.spines.top": True, "axes.spines.right": True})
    summary = rows(args.analysis_dir / "model_step_summary.csv")
    pairwise = rows(args.analysis_dir / "pairwise_force_comparison.csv")
    atoms = rows(args.analysis_dir / "atom_force_disagreement.csv")
    geom = rows(args.analysis_dir / "geometry_diagnostics.csv")
    model_order = ["from_scratch", "naive_ft", "multi_t", "foundation"]
    steps = sorted({int(r["step"]) for r in summary})

    # Force disagreement: a compact pairwise heatmap plus force vector spread by step.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"width_ratios": [1.0, 1.35]})
    # Build by symmetric lookup to avoid depending on pair ordering.
    def pval(a, b):
        if a == b: return 0.0
        for r in pairwise:
            if int(r["step"]) == steps[-1] and {r["model_a"], r["model_b"]} == {a, b}:
                return float(r["force_vector_mean_norm_eV_A"])
        return np.nan
    mat = np.array([[pval(a, b) for b in model_order] for a in model_order])
    heatmap_cmap = LinearSegmentedColormap.from_list("atlas_sequential", ATLAS_SEQUENTIAL)
    im = ax0.imshow(mat, cmap=heatmap_cmap, vmin=0, aspect="equal")
    ax0.set_xticks(range(4), [LABELS[x] for x in model_order], rotation=32, ha="right")
    ax0.set_yticks(range(4), [LABELS[x] for x in model_order])
    for i in range(4):
        for j in range(4):
            ax0.text(j, i, "—" if i == j else f"{mat[i,j]:.2f}", ha="center", va="center", color="white" if mat[i,j] > np.nanmedian(mat) else "black", fontsize=8)
    ax0.set_title(f"Force disagreement at step {steps[-1]:,}")
    ax0.set_xlabel("Pairwise mean |ΔF| (eV Å$^{-1}$)")
    colorbar = fig.colorbar(im, ax=ax0, fraction=.046, pad=.04)
    colorbar.ax.tick_params(width=.8, length=3)
    colorbar.set_label("eV Å$^{-1}$", labelpad=5)
    style_axis(ax0)
    for m in model_order:
        y = [float(next(r for r in summary if int(r["step"]) == s and r["model_id"] == m)["force_vs_other_models_rmse_eV_A"]) for s in steps]
        ax1.plot(steps, y, marker="o", lw=2.0, ms=4.8, color=COLORS[m], label=LABELS[m], solid_capstyle="round")
    ax1.set_xlabel("MD step")
    ax1.set_ylabel("LOO force RMSE (eV Å$^{-1}$)")
    ax1.set_title("Model-to-other-model disagreement")
    ax1.set_xticks(steps, [step_formatter(s) for s in steps], rotation=42, ha="right")
    ax1.margins(x=.04)
    style_axis(ax1)
    legend = ax1.legend(loc="lower center", bbox_to_anchor=(.5, 1.24), ncol=2, frameon=True, fontsize=8.2, handlelength=2.0, columnspacing=1.0, borderpad=.35)
    style_legend(legend)
    fig.subplots_adjust(left=.085, right=.985, bottom=.29, top=.68, wspace=.34)
    save(fig, args.output_dir, "fig1_force_disagreement")

    # Key local force panel: Cr atoms and their local shell around the most disputed atom.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for m in model_order:
        y = [float(next(r for r in summary if int(r["step"]) == s and r["model_id"] == m)["key_Cr_force_norm_mean_eV_A"]) for s in steps]
        ax.plot(steps, y, marker="o", lw=2.0, ms=5.0, color=COLORS[m], label=LABELS[m], solid_capstyle="round")
    ax2 = ax.twinx()
    spread = [float(next(r for r in geom if int(r["step"]) == s)["top_force_pairwise_spread_eV_A"]) for s in steps]
    ax2.plot(steps, spread, color=COLORS["ensemble"], lw=1.8, ls=(0, (5, 3)), marker="s", ms=4.2, label="max atom spread", solid_capstyle="round")
    ax.set_xlabel("MD step")
    ax.set_ylabel("Mean |F| on Cr atoms (eV Å$^{-1}$)")
    ax2.set_ylabel("Largest atom force spread (eV Å$^{-1}$)")
    ax.set_title("Key Cr-local force response")
    ax.set_xticks(steps)
    ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))
    ax.margins(x=.04)
    style_axis(ax)
    style_axis(ax2)
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(lines + lines2, labels + labels2, loc="lower center", bbox_to_anchor=(.5, 1.22), ncol=3, fontsize=8.2, frameon=True, handlelength=2.0, columnspacing=1.0, borderpad=.35)
    style_legend(legend)
    fig.subplots_adjust(left=.11, right=.90, bottom=.18, top=.69)
    save(fig, args.output_dir, "fig2_key_local_force")

    # Geometry / step variation panel, making explicit that the input geometry is fixed across models.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(steps))
    min_d = [float(next(r for r in geom if int(r["step"]) == s)["min_pair_distance_A"]) for s in steps]
    shell_n = [float(next(r for r in geom if int(r["step"]) == s)["local_shell_atom_count_radius_A"]) for s in steps]
    ax.plot(x, min_d, color=ATLAS_PALETTE["categorical"][0], marker="o", lw=2.0, ms=5.0, label="minimum MIC pair distance", solid_capstyle="round")
    ax.set_xticks(x, [f"{s:,}" for s in steps], rotation=0)
    ax.set_ylabel("Minimum pair distance (Å)")
    ax.set_xlabel("Common input snapshot")
    ax.set_title("Geometry diagnostics and local environment")
    ax2 = ax.twinx(); ax2.plot(x, shell_n, color=ATLAS_PALETTE["categorical"][3], marker="s", ms=5.0, lw=2.0, label="atoms within 3 Å of max-spread atom", solid_capstyle="round"); ax2.set_ylabel("Local-shell atom count")
    style_axis(ax)
    style_axis(ax2)
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax.legend(lines + lines2, labels + labels2, loc="lower center", bbox_to_anchor=(.5, 1.22), ncol=2, fontsize=8.2, frameon=True, handlelength=2.0, columnspacing=1.0, borderpad=.35)
    style_legend(legend)
    fig.subplots_adjust(left=.11, right=.90, bottom=.18, top=.70)
    save(fig, args.output_dir, "fig3_geometry_step_diagnostics")
    print(json.dumps({"figures": [str(args.output_dir / f) for f in ("fig1_force_disagreement.png", "fig2_key_local_force.png", "fig3_geometry_step_diagnostics.png")]}, indent=2))


if __name__ == "__main__":
    main()
