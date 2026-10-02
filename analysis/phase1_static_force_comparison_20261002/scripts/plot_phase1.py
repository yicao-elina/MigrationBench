#!/usr/bin/env python3
"""Publication-style Phase 1 figures using the local Scientific Plot Atlas grammar."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOT_LIBRARY = Path("/Users/alina/Documents/Codex/plot_library")
if (PLOT_LIBRARY / "src").exists():
    sys.path.insert(0, str(PLOT_LIBRARY / "src"))
try:
    from plot_atlas import apply_publication_style, save_figure
except Exception:
    def apply_publication_style(font_size=11):
        plt.rcParams.update({"font.family": "Arial", "font.size": font_size, "axes.linewidth": 1.5})
    def save_figure(fig, path):
        fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")


COLORS = {"foundation": "#0072B2", "naive_ft": "#D55E00", "multi_t": "#009E73", "from_scratch": "#CC79A7", "ensemble": "#333333"}
LABELS = {"foundation": "Foundation-OMAT", "naive_ft": "naive-FT", "multi_t": "multi-T FT", "from_scratch": "from-scratch", "ensemble": "4-model mean"}


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
    apply_publication_style(11)
    plt.rcParams.update({"font.family": "Arial", "axes.linewidth": 1.5, "xtick.major.width": 1.3, "ytick.major.width": 1.3, "axes.spines.top": True, "axes.spines.right": True})
    summary = rows(args.analysis_dir / "model_step_summary.csv")
    pairwise = rows(args.analysis_dir / "pairwise_force_comparison.csv")
    atoms = rows(args.analysis_dir / "atom_force_disagreement.csv")
    geom = rows(args.analysis_dir / "geometry_diagnostics.csv")
    model_order = ["from_scratch", "naive_ft", "multi_t", "foundation"]
    steps = sorted({int(r["step"]) for r in summary})

    # Force disagreement: a compact pairwise heatmap plus force vector spread by step.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={"width_ratios": [1.0, 1.3]})
    # Build by symmetric lookup to avoid depending on pair ordering.
    def pval(a, b):
        if a == b: return 0.0
        for r in pairwise:
            if int(r["step"]) == steps[-1] and {r["model_a"], r["model_b"]} == {a, b}:
                return float(r["force_vector_mean_norm_eV_A"])
        return np.nan
    mat = np.array([[pval(a, b) for b in model_order] for a in model_order])
    im = ax0.imshow(mat, cmap="magma", vmin=0, aspect="equal")
    ax0.set_xticks(range(4), [LABELS[x] for x in model_order], rotation=35, ha="right", fontsize=8)
    ax0.set_yticks(range(4), [LABELS[x] for x in model_order], fontsize=8)
    for i in range(4):
        for j in range(4):
            ax0.text(j, i, "—" if i == j else f"{mat[i,j]:.2f}", ha="center", va="center", color="white" if mat[i,j] > np.nanmedian(mat) else "black", fontsize=8)
    ax0.set_title(f"Force disagreement at step {steps[-1]:,}", fontsize=10)
    ax0.set_xlabel("Model pair mean |ΔF| (eV Å$^{-1}$)")
    fig.colorbar(im, ax=ax0, fraction=.046, pad=.04)
    for m in model_order:
        y = [float(next(r for r in summary if int(r["step"]) == s and r["model_id"] == m)["force_vs_other_models_rmse_eV_A"]) for s in steps]
        ax1.plot(steps, y, marker="o", lw=1.8, ms=4, color=COLORS[m], label=LABELS[m])
    ax1.set_xlabel("MD step")
    ax1.set_ylabel("LOO force RMSE (eV Å$^{-1}$)")
    ax1.set_title("Model-to-other-model disagreement", fontsize=10)
    ax1.ticklabel_format(axis="x", style="plain")
    ax1.legend(frameon=True, fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, args.output_dir, "fig1_force_disagreement")

    # Key local force panel: Cr atoms and their local shell around the most disputed atom.
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for m in model_order:
        y = [float(next(r for r in summary if int(r["step"]) == s and r["model_id"] == m)["key_Cr_force_norm_mean_eV_A"]) for s in steps]
        ax.plot(steps, y, marker="o", lw=1.8, ms=4, color=COLORS[m], label=LABELS[m])
    ax2 = ax.twinx()
    spread = [float(next(r for r in geom if int(r["step"]) == s)["top_force_pairwise_spread_eV_A"]) for s in steps]
    ax2.plot(steps, spread, color="#444444", lw=1.5, ls="--", marker="s", ms=3, label="max atom spread")
    ax.set_xlabel("MD step")
    ax.set_ylabel("Mean |F| on Cr atoms (eV Å$^{-1}$)")
    ax2.set_ylabel("Largest atom force spread (eV Å$^{-1}$)")
    ax.set_title("Key Cr-local force response", fontsize=10)
    ax.ticklabel_format(axis="x", style="plain")
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax.legend(lines + lines2, labels + labels2, fontsize=8, frameon=True)
    fig.tight_layout()
    save(fig, args.output_dir, "fig2_key_local_force")

    # Geometry / step variation panel, making explicit that the input geometry is fixed across models.
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(steps))
    min_d = [float(next(r for r in geom if int(r["step"]) == s)["min_pair_distance_A"]) for s in steps]
    shell_n = [float(next(r for r in geom if int(r["step"]) == s)["local_shell_atom_count_radius_A"]) for s in steps]
    ax.plot(x, min_d, color="#0072B2", marker="o", lw=1.8, label="minimum MIC pair distance")
    ax.set_xticks(x, [f"{s:,}" for s in steps], rotation=25)
    ax.set_ylabel("Minimum pair distance (Å)")
    ax.set_xlabel("Common input snapshot")
    ax.set_title("Geometry diagnostics and local environment", fontsize=10)
    ax2 = ax.twinx(); ax2.plot(x, shell_n, color="#D55E00", marker="s", lw=1.6, label="atoms within 3 Å of max-spread atom"); ax2.set_ylabel("Local-shell atom count")
    lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax.legend(lines + lines2, labels + labels2, fontsize=8, frameon=True, loc="best")
    fig.tight_layout()
    save(fig, args.output_dir, "fig3_geometry_step_diagnostics")
    print(json.dumps({"figures": [str(args.output_dir / f) for f in ("fig1_force_disagreement.png", "fig2_key_local_force.png", "fig3_geometry_step_diagnostics.png")]}, indent=2))


if __name__ == "__main__":
    main()
