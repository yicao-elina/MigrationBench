#!/usr/bin/env python3
"""Plot the complete from-scratch reproduction audit with Plot Atlas grammar."""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from plot_atlas import (
    apply_publication_style,
    bold_tick_labels,
    get_palette,
    grouped_bar,
    save_figure,
    timecourse,
)

SOURCE_COLORS = {
    "original": get_palette("nature_steered_dynamics")["categorical"][0],
    "reproduce-1": get_palette("nature_steered_dynamics")["categorical"][1],
    "reproduce-2": get_palette("nature_steered_dynamics")["categorical"][3],
}
SOURCE_STYLES = {"original": "-", "reproduce-1": "--", "reproduce-2": ":"}
WINDOW_COLORS = {
    "stable_pre": get_palette("descriptor_reduction")["total"],
    "precursor": get_palette("descriptor_reduction")["energy"],
    "threshold": get_palette("descriptor_reduction")["target"],
}
REFERENCE_COLORS = {
    "train": get_palette("nature_feature_analysis")["roles"]["global"],
    "valid": get_palette("nature_feature_analysis")["roles"]["not_retrained"],
    "test": get_palette("nature_feature_analysis")["roles"]["retrained"],
    "neb": get_palette("descriptor_reduction")["energy"],
}
ELEMENT_COLORS = {"Cr": "#4C4C4C", "Sb": "#98449F", "Te": "#337FB0"}


def decorate(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#222222")
    ax.tick_params(width=1.1, length=5)
    bold_tick_labels(ax)


def save_panel(fig: plt.Figure, output: Path, stem: str) -> None:
    save_figure(fig, output / f"{stem}.png", dpi=350)
    fig.savefig(output / f"{stem}.pdf", dpi=350, bbox_inches="tight", metadata={"Creator": "Plot Atlas", "Date": None})
    plt.close(fig)


def source_legend() -> List[Line2D]:
    return [Line2D([0], [0], color=color, lw=2.0, ls=SOURCE_STYLES[source], label=source) for source, color in SOURCE_COLORS.items()]


def plot_reproduction(metrics: pd.DataFrame, output: Path) -> None:
    seeds = [123, 234, 345]
    series = [
        ("temperature_K", "Temperature (K)", 5000.0),
        ("max_force_eV_A", "Recorded max |F| (eV Å$^{-1}$)", 100.0),
        ("min_distance_A", "Minimum pair distance (Å)", 0.5),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 8.5), sharex="col", constrained_layout=True)
    for row_index, seed in enumerate(seeds):
        subset = metrics[metrics["training_seed"].astype(int) == seed]
        for col_index, (metric, ylabel, threshold) in enumerate(series):
            ax = axes[row_index, col_index]
            for source in ("original", "reproduce-1", "reproduce-2"):
                current = subset[subset["source"] == source].sort_values("relative_time_ps")
                if current.empty:
                    continue
                timecourse(ax, current["relative_time_ps"], [current[metric]], [source], [SOURCE_COLORS[source]], markers=False, legend=False, line_width=1.5, linestyles=[SOURCE_STYLES[source]])
            ax.axhline(threshold, color="#444444", lw=1.0, ls=(0, (3, 2)), zorder=1)
            ax.axvline(0.0, color="#222222", lw=1.0, ls=(0, (2, 2)), zorder=1)
            ax.set_xlim(-35.0, 2.0)
            if row_index == 0:
                ax.set_title(ylabel, fontweight="bold")
            if col_index == 0:
                ax.set_ylabel(f"seed {seed}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            if row_index == len(seeds) - 1:
                ax.set_xlabel("Time relative to threshold onset (ps)")
            decorate(ax)
    axes[0, 0].legend(handles=source_legend(), frameon=True, fontsize=8, loc="best")
    fig.suptitle("Exact-protocol from-scratch failure reproduction", fontweight="bold")
    save_panel(fig, output, "fig_reproduction_threshold_onset")


def plot_distribution(frame: pd.DataFrame, refs: pd.DataFrame, output: Path) -> None:
    # The diagnostic CSV stores absolute time and onset separately; derive the
    # relative axis here so plotting remains compatible with that data contract.
    frame = frame.copy()
    if "relative_time_ps" not in frame.columns:
        frame["relative_time_ps"] = frame["time_ps"] - frame["onset_ps"]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    for label, group in refs.groupby("reference"):
        ax.scatter(group["pca1"], group["pca2"], s=9, alpha=0.25, color=REFERENCE_COLORS.get(label, "#777777"), edgecolors="none", rasterized=True, label=label)
    for window, group in frame[frame["window"].isin(["stable_pre", "precursor", "threshold"])].groupby("window"):
        ax.scatter(group["pca1"], group["pca2"], s=26, alpha=0.88, color=WINDOW_COLORS[window], edgecolor="white", linewidth=0.35, label=window)
    ax.set_xlabel("PC1 (training-standardized SOAP)")
    ax.set_ylabel("PC2 (training-standardized SOAP)")
    ax.set_title("Reference clouds and MD windows", fontweight="bold")
    ax.legend(fontsize=6.5, frameon=True, ncol=2)
    decorate(ax)

    ax = axes[0, 1]
    distance_columns = [("distance_to_train", "train"), ("distance_to_test", "test"), ("distance_to_neb", "NEB")]
    for (seed, source), group in frame.groupby([frame["training_seed"].astype(int), "source"]):
        current = group[group["window"].isin(["stable_pre", "precursor", "threshold"])].sort_values("relative_time_ps")
        for column, label in distance_columns:
            if column not in current:
                continue
            timecourse(ax, current["time_ps"] - current["onset_ps"], [current[column]], [label], [REFERENCE_COLORS.get(label.lower(), "#777777")], markers=False, legend=False, line_width=0.9, linestyles=["-"])
    ax.axvline(0.0, color="#222222", lw=1.0, ls=(0, (2, 2)))
    ax.set_xlabel("Time relative to threshold onset (ps)")
    ax.set_ylabel("Nearest-neighbour distance")
    ax.set_title("Distribution distance near onset", fontweight="bold")
    decorate(ax)

    ax = axes[1, 0]
    categories = [("distance_to_train", "train"), ("distance_to_test", "test"), ("distance_to_neb", "NEB")]
    positions = np.arange(len(categories), dtype=float)
    for index, (column, label) in enumerate(categories):
        if column not in frame:
            continue
        stable = frame.loc[frame["window"] == "stable_pre", column].dropna().to_numpy(float)
        transition = frame.loc[frame["window"].isin(["precursor", "threshold"]), column].dropna().to_numpy(float)
        if len(stable):
            ax.boxplot(stable, positions=[index - .17], widths=.27, patch_artist=True, boxprops={"facecolor": WINDOW_COLORS["stable_pre"], "edgecolor": "#222222"}, medianprops={"color": "#222222"})
        if len(transition):
            ax.boxplot(transition, positions=[index + .17], widths=.27, patch_artist=True, boxprops={"facecolor": WINDOW_COLORS["precursor"], "edgecolor": "#222222"}, medianprops={"color": "#222222"})
    ax.set_xticks(positions, [label for _, label in categories])
    ax.set_ylabel("Nearest-neighbour distance")
    ax.set_title("Stable versus precursor/threshold novelty", fontweight="bold")
    ax.legend(handles=[Line2D([0], [0], color=WINDOW_COLORS["stable_pre"], lw=7, label="stable pre"), Line2D([0], [0], color=WINDOW_COLORS["precursor"], lw=7, label="precursor/threshold")], frameon=True, fontsize=7)
    decorate(ax)

    ax = axes[1, 1]
    features = ["Cr_Te_coord_3p2_mean", "Cr_Sb_coord_3p2_mean", "CrTe_bond_median_A", "CrTe_angle_mean_deg", "min_CrTe_A"]
    labels = ["Cr–Te coord.", "Cr–Sb coord.", "Cr–Te bond", "Cr–Te angle", "min Cr–Te"]
    window_order = ["stable_pre", "precursor", "threshold"]
    rows = []
    for window in window_order:
        values = []
        for feature in features:
            values.append(float(frame.loc[frame["window"] == window, feature].mean()) if feature in frame and frame[feature].notna().any() else np.nan)
        rows.append(values)
    matrix = np.asarray(rows, dtype=float).T
    baseline = np.nanmean(matrix[:, [0]], axis=1, keepdims=True)
    scale = np.nanstd(matrix[:, [0]], axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    matrix = (matrix - baseline) / scale
    grouped_bar(ax, labels, matrix, window_order, [WINDOW_COLORS[name] for name in window_order], ylabel="Change from stable pre (scaled)", legend=True)
    ax.set_title("Local Cr-centred geometry", fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    decorate(ax)
    fig.suptitle("Failure-distribution attribution", fontweight="bold")
    save_panel(fig, output, "fig_failure_distribution_attribution")


def plot_disagreement(disagreement: Optional[pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True, constrained_layout=True)
    seeds = [123, 234, 345]
    if disagreement is None or disagreement.empty:
        for ax, seed in zip(axes, seeds):
            ax.text(.5, .5, "force disagreement unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"seed {seed}", fontweight="bold")
            decorate(ax)
    else:
        for ax, seed in zip(axes, seeds):
            subset = disagreement[disagreement["training_seed"].astype(int) == seed].copy()
            for source, group in subset.groupby("source"):
                group = group.sort_values("time_ps")
                timecourse(ax, group["time_ps"] - group["onset_ps"], [group["ensemble_force_rms_std_eV_A"]], [source], [SOURCE_COLORS.get(source, "#777777")], markers=True, legend=False, line_width=1.2, marker_size=3.0, linestyles=[SOURCE_STYLES.get(source, "-")])
            ax.axvline(0.0, color="#222222", lw=1.0, ls=(0, (2, 2)))
            ax.set_title(f"seed {seed}", fontweight="bold")
            ax.set_xlabel("Time relative to onset (ps)")
            if seed == 123:
                ax.set_ylabel("Ensemble force spread RMS (eV Å$^{-1}$)")
            decorate(ax)
    axes[0].legend(handles=source_legend(), frameon=True, fontsize=7, loc="best")
    fig.suptitle("Cross-model force disagreement", fontweight="bold")
    save_panel(fig, output, "fig_force_disagreement")


def _local_snapshot(ax: plt.Axes, snapshot: Path, row: pd.Series) -> None:
    try:
        from ase.io import read

        atoms = read(snapshot, index=-1, format="extxyz")
        center = int(row.get("worst_force_atom_index", -1))
        if center < 0 or center >= len(atoms):
            center = 0
        vectors = atoms.get_distances(center, np.arange(len(atoms)), mic=True, vector=True)
        distances = np.linalg.norm(vectors, axis=1)
        keep = distances <= 5.0
        keep[center] = True
        # PCA-like deterministic local projection; fall back to x/y if degenerate.
        local = vectors[keep]
        if len(local) > 2:
            _, _, vh = np.linalg.svd(local - local.mean(axis=0), full_matrices=False)
            projected = (local - local.mean(axis=0)) @ vh[:2].T
        else:
            projected = local[:, :2]
        symbols = np.asarray(atoms.get_chemical_symbols())[keep]
        for symbol in ("Cr", "Sb", "Te"):
            mask = symbols == symbol
            ax.scatter(projected[mask, 0], projected[mask, 1], s=19, color=ELEMENT_COLORS[symbol], edgecolor="white", linewidth=.25, label=symbol if row["window"] == "stable_pre" else None)
        kept_indices = np.flatnonzero(keep)
        center_index = int(np.flatnonzero(kept_indices == center)[0]) if center in kept_indices else 0
        ax.scatter(projected[center_index, 0], projected[center_index, 1], s=45, facecolor="#E8EB26", edgecolor="#222222", linewidth=.8, zorder=4)
        ax.set_title(f"seed {int(row['training_seed'])} · {row['window']}\nstep {int(row['step'])} ({float(row['time_ps']):.1f} ps)", fontsize=8, fontweight="bold")
        ax.set_xlim(np.nanmin(projected[:, 0]) - .8, np.nanmax(projected[:, 0]) + .8)
        ax.set_ylim(np.nanmin(projected[:, 1]) - .8, np.nanmax(projected[:, 1]) + .8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
    except Exception as exc:
        ax.text(.5, .5, f"snapshot unavailable\n{type(exc).__name__}", ha="center", va="center", transform=ax.transAxes, fontsize=7)
    decorate(ax)


def plot_snapshot_atlas(frame: pd.DataFrame, analysis: Path, output: Path) -> None:
    if "snapshot_path" not in frame:
        return
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 8.6), constrained_layout=True)
    for row_index, seed in enumerate([123, 234, 345]):
        subset = frame[(frame["training_seed"].astype(int) == seed) & frame["window"].isin(["stable_pre", "precursor", "threshold"])].copy()
        for col_index, window in enumerate(["stable_pre", "precursor", "threshold"]):
            ax = axes[row_index, col_index]
            candidates = subset[subset["window"] == window]
            if candidates.empty:
                ax.text(.5, .5, "not available", ha="center", va="center", transform=ax.transAxes)
                decorate(ax)
                continue
            preferred = {"reproduce-2": 0, "reproduce-1": 1, "original": 2}
            selected = candidates.assign(_preference=candidates["source"].map(preferred).fillna(9)).sort_values("_preference").iloc[0]
            snapshot = Path(selected["snapshot_path"])
            if not snapshot.exists():
                snapshot = analysis / "snapshots" / snapshot.parent.name / snapshot.name
            _local_snapshot(ax, snapshot, selected)
    axes[0, 0].legend(frameon=True, fontsize=7, loc="upper right")
    fig.suptitle("Onset-neighbouring local configurations (5 Å around top-force atom)", fontweight="bold")
    save_panel(fig, output, "fig_snapshot_atlas")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    apply_publication_style(10)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.analysis / "metrics_all.csv")
    frame = pd.read_csv(args.analysis / "frame_features.csv")
    refs = pd.read_csv(args.analysis / "reference_embedding.csv")
    plot_reproduction(metrics, args.output)
    plot_distribution(frame, refs, args.output)
    disagreement_path = args.analysis / "force_disagreement.csv"
    disagreement = pd.read_csv(disagreement_path) if disagreement_path.exists() else None
    plot_disagreement(disagreement, args.output)
    plot_snapshot_atlas(frame, args.analysis, args.output)
    print(f"wrote Plot Atlas figures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
