#!/usr/bin/env python3
"""Render the clean, raw-PCA Figure 1A panel as an editable SVG/PDF."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from build_atomic_migration_figures import (
    AXIS_LABEL_SIZE,
    DFT_SPLIT_MARKERS,
    LEGEND_SIZE,
    TASK_MARKERS,
    TASK_GROUPS,
    _add_task_hulls,
    _ood_marker_sizes,
    f,
    save_figure,
    set_house_style,
    style_ax,
)


WILEY = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
SRC = WILEY / "analysis/migrationbench_universe_publication_20261026/derived/comprehensive_task_map_points.csv"
OUT = Path(__file__).resolve().parents[1] / "R06_clean_raw_pca" / "figures"
GROUP_COLORS = {
    "L0 DFT Reference": "#8DA9BF",
    "L1 Thermal Equilibrium": "#6E9E75",
    "L2 NEB Migration": "#C4547A",
    "L3 Layer Sliding": "#E68F1D",
}
VISIBLE_SPLITS = ("train", "valid", "test", "md_sample", "dft_neb", "dft_interlayer")


def main() -> None:
    with SRC.open(newline="", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    sizes, reference = _ood_marker_sizes(rows)
    for row, size in zip(rows, sizes):
        row["_ood_marker_size"] = float(size)
    visible_rows = [row for row in rows if row.get("split") in VISIBLE_SPLITS]

    set_house_style()
    fig, ax = plt.subplots(figsize=(8.8, 7.9), dpi=400)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.29, top=0.88)
    _add_task_hulls(ax, visible_rows)
    for task in TASK_GROUPS:
        for split in VISIBLE_SPLITS:
            sub = [row for row in visible_rows if row.get("task_group") == task and row.get("split") == split]
            if not sub:
                continue
            x = np.asarray([f(row, "x_pca") for row in sub], dtype=float)
            y = np.asarray([f(row, "y_pca") for row in sub], dtype=float)
            size = np.asarray([row["_ood_marker_size"] for row in sub], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(size)
            if not finite.any():
                continue
            marker = TASK_MARKERS[task]
            dense_split = split in {"train", "valid", "test"}
            ax.scatter(
                x[finite],
                y[finite],
                s=size[finite],
                marker=marker,
                color=GROUP_COLORS[task],
                alpha=0.28 if dense_split else 0.78,
                linewidths=0.12 if dense_split else 0.22,
                edgecolors="white",
                zorder=2 if dense_split else 3,
            )

    plotted = np.asarray([[f(row, "x_pca"), f(row, "y_pca")] for row in visible_rows], dtype=float)
    plotted = plotted[np.isfinite(plotted).all(axis=1)]
    lo, hi = np.nanmin(plotted, axis=0), np.nanmax(plotted, axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1e-12)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlabel("SOAP PCA 1", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("SOAP PCA 2", fontsize=AXIS_LABEL_SIZE)
    ax.set_title("A  Configuration universe", loc="left")
    style_ax(ax)

    task_handles = [
        Line2D(
            [], [], marker=TASK_MARKERS[task], linestyle="", color=GROUP_COLORS[task],
            markeredgecolor="white", label=task, markersize=8.5,
        )
        for task in TASK_GROUPS
    ]
    dft_split_handles = [
        Line2D(
            [], [], marker=marker, linestyle="", color="#777777", markerfacecolor="#888888",
            markeredgecolor="white", label=label, markersize=8.0,
        )
        for label, marker in DFT_SPLIT_MARKERS.items()
    ]
    size_handles = [
        Line2D([], [], marker="o", linestyle="", color="#555555", markerfacecolor="#555555", markeredgecolor="white", markersize=float(np.sqrt(size)), label=label)
        for size, label in zip([18.0, 70.0, 190.0], ["near train", "intermediate", "far OOD"])
    ]
    ax.legend(
        handles=[*task_handles, *dft_split_handles, *size_handles],
        frameon=False,
        fontsize=LEGEND_SIZE,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.25),
        borderaxespad=0.0,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    assets = save_figure(fig, OUT, "figure01a_clean_raw_pca")
    print({"assets": assets, "source": str(SRC), "foreground_splits": sorted(VISIBLE_SPLITS), "ood_reference_p98": reference})


if __name__ == "__main__":
    main()
