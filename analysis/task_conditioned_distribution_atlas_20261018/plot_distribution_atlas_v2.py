#!/usr/bin/env python3
"""Render the revised grouped-split distribution atlas.

The figure contains only plot-native elements: axes, titles, legends, density
contours, and data marks.  Scope and evidence qualifications belong in the
caption and the accompanying audit tables, not as small in-figure notes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
FIG = HERE / "figures"
TABLES = HERE / "tables"
PLOT_ATLAS = Path("/Users/alina/Documents/Codex/plot_library/src")
if str(PLOT_ATLAS) not in sys.path:
    sys.path.insert(0, str(PLOT_ATLAS))
from plot_atlas import apply_publication_style, get_palette  # noqa: E402


PAL = get_palette("jhu_mace_models")
MODEL_ORDER = ["Foundation", "Scratch", "FT-600K", "FT-MultiT"]
MODEL_COLORS = {
    "Foundation": PAL["roles"]["foundation_omat"],
    "Scratch": PAL["roles"]["from_scratch"],
    "FT-600K": PAL["roles"]["ft_600k"],
    "FT-MultiT": PAL["roles"]["ft_multit"],
}
SPLIT_COLORS = {"train": "#A8ADB5", "valid": "#5F7897", "test": "#D34F2D"}
INK = "#222222"


def style(ax, *, xlabel=None, ylabel=None, title=None):
    ax.grid(False)
    ax.tick_params(direction="out", width=1.15, length=4.5, labelsize=12.5)
    for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        t.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.1)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13.5, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13.5, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=15.5, fontweight="bold", pad=8)


def panel_letter(ax, label):
    ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=17,
            fontweight="bold", va="bottom", ha="left", color=INK)


def kde_grid(x, y, bounds, n=160):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    xx = np.linspace(bounds[0], bounds[1], n)
    yy = np.linspace(bounds[2], bounds[3], n)
    X, Y = np.meshgrid(xx, yy)
    try:
        Z = gaussian_kde(np.vstack([x, y]))(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    except np.linalg.LinAlgError:
        return None
    return X, Y, Z


def kde_line(x, bounds, n=240):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.ptp(x) == 0:
        return None
    grid = np.linspace(bounds[0], bounds[1], n)
    try:
        return grid, gaussian_kde(x)(grid)
    except np.linalg.LinAlgError:
        return None


def draw_joint_reference(fig, spec, ref):
    sub = spec.subgridspec(4, 4, width_ratios=[1, 1, 1, 0.24], height_ratios=[0.60, 1, 1, 1],
                           wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(sub[1:, :3])
    ax_top = fig.add_subplot(sub[0, :3], sharex=ax)
    ax_right = fig.add_subplot(sub[1:, 3], sharey=ax)
    xlim = (-2.7, 3.25)
    ylim = (-2.65, 2.75)
    for split in ["train", "valid", "test"]:
        part = ref.loc[ref["split"].eq(split)]
        color = SPLIT_COLORS[split]
        ax.scatter(part.pca1, part.pca2, s=9 if split == "train" else 15,
                   color=color, alpha=0.26 if split == "train" else 0.45,
                   linewidths=0, rasterized=True, zorder=2)
        grid = kde_grid(part.pca1, part.pca2, (*xlim, *ylim))
        if grid is not None:
            X, Y, Z = grid
            levels = np.linspace(np.nanpercentile(Z, 55), np.nanmax(Z) * 0.98, 5)
            ax.contour(X, Y, Z, levels=levels, colors=[color], linewidths=1.0,
                       alpha=0.78, zorder=3)
            ax.contourf(X, Y, Z, levels=levels, colors=[color], alpha=0.055, zorder=1)
        top = kde_line(part.pca1, xlim)
        right = kde_line(part.pca2, ylim)
        if top is not None:
            ax_top.plot(top[0], top[1], color=color, linewidth=1.8, label=f"{split} ({len(part):,})")
        if right is not None:
            ax_right.plot(right[1], right[0], color=color, linewidth=1.8)
    neb = ref.loc[ref["split"].eq("neb")]
    ax.scatter(neb.pca1, neb.pca2, s=46, marker="D", facecolor="white", edgecolor=INK,
               linewidth=1.05, label="fixed-path NEB", zorder=6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    style(ax, xlabel="SOAP PCA 1", ylabel="SOAP PCA 2")
    ax_top.set_title("Grouped DFT reference space", fontsize=14.0, fontweight="bold", pad=7)
    ax_top.legend(frameon=True, fontsize=12.0, loc="lower left", ncol=3,
                  borderpad=0.35, handlelength=1.5, columnspacing=0.8)
    ax_top.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
    ax_right.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    for a in [ax_top, ax_right]:
        for spine in a.spines.values():
            spine.set_visible(False)
    ax_top.set_ylabel(""); ax_right.set_xlabel("")
    panel_letter(ax, "a")
    return ax


def prepare_static_points():
    grouped = pd.read_csv(ROOT / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/grouped_seed_comparison.csv")
    rows = []
    for _, r in grouped.iterrows():
        fam = {"OMAT-FT": "FT-600K", "Scratch": "Scratch"}.get(r.family)
        if fam:
            rows.append({"family": fam, "seed": r.seed, "energy_rmse_mev_atom": r.energy_rmse_mev_atom,
                         "force_rmse_mev_A": r.force_rmse_mev_A, "evidence_status": "grouped_multi_seed"})
    foundation = pd.read_csv(ROOT / "data/processed/foundation_grouped/foundation_vs_grouped_comparison.csv")
    for _, r in foundation.loc[foundation["class"].eq("foundation")].iterrows():
        rows.append({"family": "Foundation", "seed": "checkpoint", "energy_rmse_mev_atom": r.energy_rmse_meV_atom,
                     "force_rmse_mev_A": r.force_rmse_meV_A, "evidence_status": "grouped_foundation_representative"})
    # FT-MultiT was not retrained under the grouped protocol.  Preserve the
    # declared historical test point as descriptive context rather than
    # silently dropping the fourth manuscript family.
    rows.append({"family": "FT-MultiT", "seed": "historical", "energy_rmse_mev_atom": 0.5,
                 "force_rmse_mev_A": 45.5, "evidence_status": "historical_frame_level"})
    out = pd.DataFrame(rows)
    out.to_csv(DERIVED / "static_error_points.csv", index=False)
    return out


def draw_static_errors(fig, spec, static):
    ax = fig.add_subplot(spec)
    force_column = "force_rmse_meV_A" if "force_rmse_meV_A" in static else "force_rmse_mev_A"
    for j, fam in enumerate(MODEL_ORDER):
        part = static.loc[static.family.eq(fam)].copy()
        x = pd.to_numeric(part["energy_rmse_mev_atom"], errors="coerce").to_numpy()
        y = pd.to_numeric(part[force_column], errors="coerce").to_numpy()
        good = np.isfinite(x) & np.isfinite(y)
        if not np.any(good):
            continue
        rng = np.random.default_rng(9400 + j)
        x = x[good] + (rng.uniform(-0.45, 0.45, int(good.sum())) if good.sum() > 1 else 0.0)
        y = y[good]
        ax.scatter(x, y, s=66, color=MODEL_COLORS[fam], edgecolor="white", linewidth=0.8,
                   alpha=0.92, label=fam, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("test energy RMSE (meV atom$^{-1}$)", fontsize=13.5, fontweight="bold")
    ax.set_ylabel("test force RMSE (meV Å$^{-1}$)", fontsize=13.5, fontweight="bold")
    ax.set_title("Static pointwise test error", fontsize=14.0, fontweight="bold", pad=7)
    ax.tick_params(direction="out", width=1.1, length=4, labelsize=12.5)
    for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]: t.set_fontweight("bold")
    for spine in ax.spines.values(): spine.set_linewidth(1.05); spine.set_color(INK)
    ax.grid(False)
    ax.legend(frameon=True, fontsize=11.5, loc="upper right", ncol=1,
              borderpad=0.4, handlelength=1.0, labelspacing=0.5)
    panel_letter(ax, "b")
    return ax


def draw_md_joint(fig, spec, ref, md):
    sub = spec.subgridspec(4, 4, width_ratios=[1, 1, 1, 0.24], height_ratios=[0.60, 1, 1, 1],
                           wspace=0.05, hspace=0.05)
    ax = fig.add_subplot(sub[1:, :3])
    ax_top = fig.add_subplot(sub[0, :3], sharex=ax)
    ax_right = fig.add_subplot(sub[1:, 3], sharey=ax)
    xlim = (-2.7, 3.25); ylim = (-2.65, 2.75)
    train = ref.loc[ref.split.eq("train")]
    grid = kde_grid(train.pca1, train.pca2, (*xlim, *ylim))
    if grid is not None:
        X, Y, Z = grid
        levels = np.linspace(np.nanpercentile(Z, 45), np.nanmax(Z) * 0.98, 6)
        ax.contour(X, Y, Z, levels=levels, colors=["#6B7280"], linewidths=1.0, alpha=0.7, zorder=2)
        ax.contourf(X, Y, Z, levels=levels, colors=["#A8ADB5"], alpha=0.12, zorder=1)
    ax.scatter(train.pca1, train.pca2, s=6, color="#A8ADB5", alpha=0.10, linewidths=0, rasterized=True)
    for fam in MODEL_ORDER:
        part = md.loc[md.family.eq(fam)]
        if part.empty:
            continue
        color = MODEL_COLORS[fam]
        ax.scatter(part.pca1, part.pca2, s=3.0, color=color, alpha=0.14, linewidths=0, rasterized=True, label=fam, zorder=3)
        top = kde_line(part.pca1, xlim)
        right = kde_line(part.pca2, ylim)
        if top is not None:
            ax_top.plot(top[0], top[1], color=color, linewidth=1.7, label=fam)
        if right is not None:
            ax_right.plot(right[1], right[0], color=color, linewidth=1.7)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    style(ax, xlabel="SOAP PCA 1", ylabel="SOAP PCA 2")
    handles = [Patch(facecolor="#A8ADB5", edgecolor="none", label="grouped train")]
    handles += [Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[f],
                       markeredgecolor="none", markersize=6, label=f) for f in MODEL_ORDER if not md.loc[md.family.eq(f)].empty]
    ax_top.set_title("600 K MLIP-generated MD", fontsize=14.0, fontweight="bold", pad=7)
    ax.legend(handles=handles, frameon=True, fontsize=11.5, loc="upper left", ncol=1,
              borderpad=0.35, handlelength=1.3, columnspacing=0.8)
    ax_top.tick_params(labelbottom=False, bottom=False, labelleft=False, left=False)
    ax_right.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    for a in [ax_top, ax_right]:
        for spine in a.spines.values(): spine.set_visible(False)
    panel_letter(ax, "c")
    return ax


def main():
    DERIVED.mkdir(exist_ok=True, parents=True); FIG.mkdir(exist_ok=True, parents=True); TABLES.mkdir(exist_ok=True, parents=True)
    apply_publication_style(font_size=12)
    ref = pd.read_csv(DERIVED / "grouped_reference_embedding.csv")
    static = prepare_static_points()
    md_files = sorted((DERIVED / "md_soap").glob("*_md_soap.csv"))
    md = pd.concat([pd.read_csv(p) for p in md_files], ignore_index=True) if md_files else pd.DataFrame(columns=["family", "pca1", "pca2"])
    if not md.empty:
        md.to_csv(DERIVED / "md_distribution_points.csv", index=False)
    if not md.empty:
        summary_rows = []
        for fam, part in md.groupby("family", sort=False):
            status = part.get("soap_status", pd.Series("ok", index=part.index)).fillna("")
            summary_rows.append({
                "family": fam,
                "n_rows": int(len(part)),
                "n_soap_ok": int(status.eq("ok").sum()),
                "n_short_contact_skipped": int(status.eq("short_contact_skipped").sum()),
                "n_pca_points": int(part[["pca1", "pca2"]].notna().all(axis=1).sum()),
            })
        summary = pd.DataFrame(summary_rows)
    else:
        summary = pd.DataFrame(columns=["family", "n_rows", "n_soap_ok", "n_short_contact_skipped", "n_pca_points"])
    summary.to_csv(DERIVED / "md_distribution_summary.csv", index=False)
    fig = plt.figure(figsize=(6.6, 5.1), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.10, 1.0], wspace=0.34, hspace=0.60,
                              left=0.13, right=0.98, bottom=0.12, top=0.94)
    draw_joint_reference(fig, grid[0, :], ref)
    draw_static_errors(fig, grid[1, 0], static)
    draw_md_joint(fig, grid[1, 1], ref, md)
    for ext, dpi in [("pdf", 300), ("png", 450), ("svg", 300)]:
        fig.savefig(FIG / f"fig_task_conditioned_distribution_atlas.{ext}", format=ext, dpi=dpi,
                    facecolor="white", metadata={"Creator": "Plot Atlas / MigrationBench"})
    plt.close(fig)
    (DERIVED / "plot_qc.json").write_text(json.dumps({
        "reference_counts": ref.groupby("split").size().to_dict(),
        "md_counts": summary.to_dict(orient="records"),
        "static_counts": static.groupby(["family", "evidence_status"]).size().reset_index(name="n").to_dict(orient="records"),
        "model_order": MODEL_ORDER,
        "figure_rule": "No explanatory notes or footnotes inside panels; caption carries evidence scope.",
    }, indent=2) + "\n")
    print(FIG / "fig_task_conditioned_distribution_atlas.pdf")


if __name__ == "__main__":
    from compose_task_conditioned_distribution_atlas import render

    render()
