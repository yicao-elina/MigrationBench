#!/usr/bin/env python3
"""Publication figures for the from-scratch failure-attribution audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.linewidth": 1.5,
            "axes.grid": False,
            "xtick.major.width": 1.3,
            "ytick.major.width": 1.3,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "savefig.bbox": "tight",
        }
    )
    for key in plt.rcParams:
        if key.startswith("xtick.label") or key.startswith("ytick.label"):
            plt.rcParams[key] = plt.rcParams[key]


COLORS = {
    "original": "#263E69",
    "reproduce": "#F04B23",
    "stable_pre": "#69BDB0",
    "precursor": "#93459D",
    "threshold": "#E8EB26",
    "train": "#3A82B4",
    "valid": "#999999",
    "test": "#4AAF4F",
    "neb": "#F04B23",
}


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=350)
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def decorate(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.3, length=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def plot_reproduction(frame: pd.DataFrame, out: Path) -> None:
    seeds = [123, 234, 345]
    metrics = [
        ("temperature_K", "Temperature (K)"),
        ("max_force_eV_A", "Maximum force (eV Å$^{-1}$)"),
        ("min_distance_A", "Minimum pair distance (Å)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.0, 8.3), sharex="col", constrained_layout=True)
    for r, seed in enumerate(seeds):
        sub = frame[frame.training_seed.astype(int) == seed].copy()
        for c, (metric, ylabel) in enumerate(metrics):
            ax = axes[r, c]
            for source in ("original", "reproduce"):
                current = sub[sub.source == source].copy()
                if current.empty:
                    continue
                current["relative_time_ps"] = current.time_ps - current.onset_ps
                current = current.sort_values("relative_time_ps")
                ax.plot(current.relative_time_ps, current[metric], color=COLORS[source], lw=1.5, label=source)
            if metric == "temperature_K":
                ax.axhline(5000.0, color="#555555", lw=1.0, ls="--")
            if metric == "max_force_eV_A":
                ax.axhline(100.0, color="#555555", lw=1.0, ls="--")
            if metric == "min_distance_A":
                ax.axhline(0.5, color="#555555", lw=1.0, ls="--")
            ax.axvline(0, color="#222222", lw=1.0, ls=":")
            if r == 0:
                ax.set_title(ylabel, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"seed {seed}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            decorate(ax)
    axes[-1, 0].set_xlabel("Time relative to threshold onset (ps)")
    axes[-1, 1].set_xlabel("Time relative to threshold onset (ps)")
    axes[-1, 2].set_xlabel("Time relative to threshold onset (ps)")
    axes[0, 0].legend(frameon=True, fontsize=8, loc="best")
    fig.suptitle("Exact-protocol from-scratch failure reproduction", fontweight="bold")
    save(fig, out / "fig_reproduction_threshold_onset")


def plot_novelty(frame: pd.DataFrame, refs: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    for label, group in refs.groupby("reference"):
        ax.scatter(group.pca1, group.pca2, s=10, alpha=0.20, color=COLORS.get(label, "#777777"), label=label)
    for (seed, source), group in frame.groupby([frame.training_seed.astype(int), "source"]):
        for window in ("stable_pre", "precursor", "threshold"):
            current = group[group.window == window]
            if current.empty:
                continue
            ax.scatter(current.pca1, current.pca2, s=24, alpha=0.85, color=COLORS.get(window), edgecolor="white", linewidth=0.3, label=f"{source}, seed {seed}, {window}")
    ax.set_xlabel("PC1 (SOAP)")
    ax.set_ylabel("PC2 (SOAP)")
    ax.set_title("Configuration distribution", fontweight="bold")
    ax.legend(fontsize=6, frameon=True, ncol=2)
    decorate(ax)

    ax = axes[0, 1]
    for source, group in frame.groupby("source"):
        current = group[group.window.isin(["stable_pre", "precursor", "threshold"])].copy()
        current["relative_time_ps"] = current.time_ps - current.onset_ps
        current = current.sort_values("relative_time_ps")
        ax.plot(current.relative_time_ps, current.distance_to_train, marker="o", ms=2, lw=1.0, alpha=0.7, color=COLORS[source], label=source)
    ax.set_xlabel("Time relative to threshold onset (ps)")
    ax.set_ylabel("Nearest training-set distance (PCA SOAP)")
    ax.set_title("Training-distribution novelty", fontweight="bold")
    ax.axvline(0, color="#222222", lw=1, ls=":")
    decorate(ax)

    ax = axes[1, 0]
    distance_cols = ["distance_to_train", "distance_to_test", "distance_to_neb"]
    available = [c for c in distance_cols if c in frame]
    positions = np.arange(len(available))
    for index, column in enumerate(available):
        stable = frame.loc[frame.window == "stable_pre", column].dropna().to_numpy()
        transition = frame.loc[frame.window.isin(["precursor", "threshold"]), column].dropna().to_numpy()
        if len(stable):
            ax.boxplot(stable, positions=[index - 0.16], widths=0.25, patch_artist=True, boxprops={"facecolor": COLORS["stable_pre"], "edgecolor": "#333333"}, medianprops={"color": "#222222"})
        if len(transition):
            ax.boxplot(transition, positions=[index + 0.16], widths=0.25, patch_artist=True, boxprops={"facecolor": COLORS["precursor"], "edgecolor": "#333333"}, medianprops={"color": "#222222"})
    ax.set_xticks(positions, [name.replace("distance_to_", "") for name in available])
    ax.set_ylabel("Nearest-neighbour distance")
    ax.set_title("Stable versus failure-window novelty", fontweight="bold")
    ax.legend(handles=[plt.Line2D([0], [0], color=COLORS["stable_pre"], lw=8, label="stable pre-window"), plt.Line2D([0], [0], color=COLORS["precursor"], lw=8, label="precursor/threshold")], frameon=True, fontsize=7)
    decorate(ax)

    ax = axes[1, 1]
    cols = ["Cr_Te_coord_3p2_mean", "Cr_Sb_coord_3p2_mean", "CrTe_angle_mean_deg", "min_CrTe_A"]
    labels = ["Cr–Te coord.", "Cr–Sb coord.", "Cr–Te angle (deg)", "min Cr–Te (Å)"]
    available = [c for c in cols if c in frame]
    x = np.arange(len(available))
    width = 0.36
    for offset, window, color in [(-width / 2, "stable_pre", COLORS["stable_pre"]), (width / 2, "precursor", COLORS["precursor"])]:
        means = [frame.loc[frame.window == window, c].mean() for c in available]
        ax.bar(x + offset, means, width=width, color=color, edgecolor="#333333", label=window)
    ax.set_xticks(x, [labels[cols.index(c)] for c in available], rotation=25, ha="right")
    ax.set_ylabel("Window mean")
    ax.set_title("Local chemical-geometry features", fontweight="bold")
    ax.legend(frameon=True, fontsize=7)
    decorate(ax)
    fig.suptitle("From-scratch failure attribution in SOAP and local geometry space", fontweight="bold")
    save(fig, out / "fig_failure_distribution_attribution")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    style()
    frame = pd.read_csv(args.analysis / "frame_features.csv")
    refs = pd.read_csv(args.analysis / "reference_embedding.csv")
    args.output.mkdir(parents=True, exist_ok=True)
    plot_reproduction(frame, args.output)
    plot_novelty(frame, refs, args.output)
    print(f"wrote figures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
