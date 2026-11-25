#!/usr/bin/env python3
"""Render the standalone 639-candidate failure/CAR diagnostic figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OTHER = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
RESULTS = OTHER / "analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/results_reconciled/mlip_candidate_results_all.csv"
SPREAD = OTHER / "analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/results_reconciled/candidate_model_spread.csv"
PLOT_ATLAS = Path("/Users/alina/Documents/Codex/plot_library/src")
if str(PLOT_ATLAS) not in sys.path:
    sys.path.insert(0, str(PLOT_ATLAS))
from plot_atlas import apply_publication_style, get_palette  # noqa: E402


PAL = get_palette("jhu_mace_models")
MODEL_COLORS = {"baseline": PAL["roles"]["from_scratch"], "ablation_lmax_2": PAL["roles"]["ft_600k"]}
STAGE_ORDER = ["stable_early", "stable_pre", "precursor", "threshold"]
STAGE_LABELS = {"stable_early": "early stable", "stable_pre": "pre-onset", "precursor": "precursor", "threshold": "threshold"}
STAGE_COLORS = {"stable_early": "#A8ADB5", "stable_pre": "#5F7897", "precursor": PAL["roles"]["foundation_omat"], "threshold": "#7A3E8E"}
INK = "#222222"


def plot_contiguous(ax, x, y, *, color, linewidth=1.55, label=None, max_gap=5.0):
    """Draw lines only across adjacent sampled times, not across stage gaps."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return
    starts = [0]
    starts.extend((np.where(np.diff(x) > max_gap)[0] + 1).tolist())
    ends = starts[1:] + [len(x)]
    first = True
    for start, end in zip(starts, ends):
        if end - start == 1:
            ax.scatter(x[start:end], y[start:end], color=color, s=10, alpha=.75, zorder=3,
                       label=label if first else None)
        else:
            ax.plot(x[start:end], y[start:end], color=color, linewidth=linewidth,
                    label=label if first else None)
        first = False


def style(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(False)
    ax.tick_params(direction="out", width=1.1, length=4.3, labelsize=11.0)
    for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]: t.set_fontweight("bold")
    for spine in ax.spines.values(): spine.set_color(INK); spine.set_linewidth(1.05)
    if xlabel: ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    if ylabel: ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    if title: ax.set_title(title, fontsize=14, fontweight="bold", pad=8)


def letter(ax, label):
    ax.text(-0.15, 1.06, label, transform=ax.transAxes, fontsize=17, fontweight="bold", ha="left", va="bottom")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    r = pd.read_csv(RESULTS)
    s = pd.read_csv(SPREAD)
    r["max_force_eV_A"] = pd.to_numeric(r["max_force_eV_A"], errors="coerce")
    r["time_ps"] = pd.to_numeric(r["time_ps"], errors="coerce")
    r = r.loc[r.evaluation_status.eq("success") & r.max_force_eV_A.notna() & r.time_ps.notna()].copy()
    s = s.loc[s.n_models_success.ge(3)].copy()
    join_cols = ["candidate_id", "candidate_group"]
    lookup = r.groupby(join_cols, as_index=False)["time_ps"].median()
    s = s.merge(lookup, on=join_cols, how="left")
    s["spread_eV_A"] = pd.to_numeric(s["pairwise_force_rms_median_eV_A"], errors="coerce")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), constrained_layout=True, facecolor="white")
    for ax in axes.flat: ax.set_facecolor("white")

    # a: direct explosion curve, with exact checkpoint identity retained.
    ax = axes[0, 0]
    for model_id, part in r.groupby("model_id", sort=False):
        color = MODEL_COLORS[part.model_family.iloc[0]]
        label = {"baseline_seed345": "baseline / run C", "ablation_lmax_2_seed123": "Lmax2 / run A", "ablation_lmax_2_seed234": "Lmax2 / run B"}.get(model_id, model_id)
        first = True
        for _, window_part in part.groupby("window", sort=False):
            stat = window_part.groupby("time_ps")["max_force_eV_A"].median().sort_index()
            q10 = window_part.groupby("time_ps")["max_force_eV_A"].quantile(0.10).reindex(stat.index)
            q90 = window_part.groupby("time_ps")["max_force_eV_A"].quantile(0.90).reindex(stat.index)
            plot_contiguous(ax, stat.index, stat.values, color=color, linewidth=1.55,
                            label=label if first else None)
            ax.fill_between(stat.index, q10.values, q90.values, color=color, alpha=0.11, linewidth=0)
            first = False
    ax.set_yscale("log"); ax.set_xlim(0, 1005); ax.set_ylim(0.8, max(2e3, r.max_force_eV_A.max() * 1.15))
    style(ax, "time (ps)", "maximum |F| (eV Å$^{-1}$)", "Failure trajectory")
    ax.legend(frameon=True, fontsize=10.0, loc="upper left", borderpad=0.35)
    letter(ax, "a")

    # b: committee disagreement as the force envelope approaches the threshold.
    ax = axes[0, 1]
    for stage in STAGE_ORDER:
        part = s.loc[s.candidate_group.eq({"stable_early": "stable_early_control", "stable_pre": "stable_pre_onset", "precursor": "precursor_near_onset", "threshold": "threshold_or_terminal"}.get(stage, stage)) & s.time_ps.notna()].copy()
        if part.empty: continue
        stat = part.groupby("time_ps")["spread_eV_A"].median().sort_index()
        plot_contiguous(ax, stat.index, stat.values, color=STAGE_COLORS[stage], linewidth=1.6,
                        label=STAGE_LABELS[stage])
    ax.set_yscale("log"); ax.set_xlim(0, 1005); ax.set_ylim(0.03, max(10, s.spread_eV_A.max() * 1.15))
    style(ax, "time (ps)", "pairwise force spread (eV Å$^{-1}$)", "Model disagreement")
    ax.legend(frameon=True, fontsize=10.0, loc="upper left", borderpad=0.35)
    letter(ax, "b")

    # c: stage-wise force distributions; the complete candidate rows remain visible.
    ax = axes[1, 0]
    positions = []; labels = []
    for gi, stage in enumerate(STAGE_ORDER):
        vals = []
        for mi, model_family in enumerate(["baseline", "ablation_lmax_2"]):
            v = r.loc[(r.model_family.eq(model_family)) & (r.window.eq(stage)), "max_force_eV_A"].to_numpy()
            if len(v):
                pos = gi * 3 + mi + 1
                ax.boxplot(v, positions=[pos], widths=.55, showfliers=False, patch_artist=True,
                           boxprops={"facecolor": MODEL_COLORS[model_family], "edgecolor": INK, "alpha": .72},
                           medianprops={"color": INK, "linewidth": 1.0}, whiskerprops={"color": INK, "linewidth": .7}, capprops={"color": INK, "linewidth": .7})
                rng = np.random.default_rng(20261018 + gi * 5 + mi)
                ax.scatter(np.full(len(v), pos) + rng.uniform(-.12, .12, len(v)), v, color=MODEL_COLORS[model_family], s=8, alpha=.22, linewidths=0, rasterized=True)
        positions.append(gi * 3 + 1.5); labels.append(STAGE_LABELS[stage])
    ax.set_xticks(positions, labels); ax.set_yscale("log"); ax.set_ylim(.8, max(2e3, r.max_force_eV_A.max() * 1.15))
    style(ax, "source stage", "maximum |F| (eV Å$^{-1}$)", "Stage-wise force envelope")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["baseline"], markeredgecolor="none", markersize=7, label="baseline"), Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["ablation_lmax_2"], markeredgecolor="none", markersize=7, label="Lmax2")], frameon=True, fontsize=10.0, loc="upper left", borderpad=.35)
    letter(ax, "c")

    # d: the same candidates in representation distance versus force space.
    ax = axes[1, 1]
    stage_order = ["stable_early", "stable_pre", "precursor", "threshold"]
    for stage in stage_order:
        part = r.loc[r.window.eq(stage)]
        ax.scatter(part.distance_to_train, part.max_force_eV_A, s=12, alpha=.36, color=STAGE_COLORS[stage], linewidths=0, rasterized=True, label=STAGE_LABELS[stage])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(.3, max(1e3, r.distance_to_train.max() * 1.15)); ax.set_ylim(.8, max(2e3, r.max_force_eV_A.max() * 1.15))
    style(ax, "distance to grouped training space", "maximum |F| (eV Å$^{-1}$)", "Failure-space diagnostic")
    ax.legend(frameon=True, fontsize=10.0, loc="upper left", borderpad=.35)
    letter(ax, "d")

    for ext, dpi in [("pdf", 300), ("png", 450), ("svg", 300)]:
        fig.savefig(OUT / f"fig_failure_stage_grouped.{ext}", format=ext, dpi=dpi, facecolor="white", metadata={"Creator": "Plot Atlas / MigrationBench"})
    plt.close(fig)
    print(OUT / "fig_failure_stage_grouped.pdf")


if __name__ == "__main__":
    main()
