#!/usr/bin/env python3
"""Render honest Figure 1 embedding candidates without post-hoc task translation.

The source table already contains the common PCA, UMAP, and t-SNE coordinates.
This script compares those coordinates with the same task colors, source-split
markers, measured OOD point area, and source-provided path lines.  No task
anchors, per-task translations, or synthetic coordinates are introduced.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull, distance

from build_atomic_migration_figures import (
    AXIS_LABEL_SIZE,
    LEGEND_SIZE,
    MIN_TEXT_SIZE,
    PANEL_TITLE_SIZE,
    TICK_SIZE,
    TITLE_SIZE,
    apply_publication_style,
    f,
    save_figure,
    style_ax,
)


WILEY = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
SRC = WILEY / "analysis/migrationbench_universe_publication_20261026/derived/comprehensive_task_map_points.csv"
OUT = Path(__file__).resolve().parents[1] / "R05_embedding_comparison"
TASKS = ["L0 DFT Reference", "L1 Thermal Equilibrium", "L2 NEB Migration", "L3 Layer Sliding"]
TASK_COLORS = {
    "L0 DFT Reference": "#8DA9BF",
    "L1 Thermal Equilibrium": "#6E9E75",
    "L2 NEB Migration": "#C4547A",
    "L3 Layer Sliding": "#E68F1D",
}
MARKERS = {
    "train": ".",
    "valid": "^",
    "test": "s",
    "md_sample": "o",
    "dft_neb": "D",
    "dft_interlayer": "P",
    "mlip_self_neb": "X",
    "mlip_interlayer_self": "P",
}
SPLIT_ORDER = list(MARKERS)
EMBEDDINGS = {
    "PCA": ("x_pca", "y_pca", "SOAP PCA 1", "SOAP PCA 2"),
    "UMAP": ("x_umap", "y_umap", "UMAP 1", "UMAP 2"),
    "t-SNE": ("x_tsne", "y_tsne", "t-SNE 1", "t-SNE 2"),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", errors="replace") as handle:
        return list(csv.DictReader(handle))


def ood_sizes(rows: list[dict[str, str]]) -> tuple[np.ndarray, float]:
    values = np.asarray([f(row, "distance_to_training") for row in rows], dtype=float)
    finite = values[np.isfinite(values)]
    ref = max(float(np.nanpercentile(finite, 98)), 1e-12)
    norm = np.clip(np.nan_to_num(values, nan=0.0) / ref, 0.0, 1.4)
    return 18.0 + 180.0 * np.power(norm, 0.68), ref


def task_hulls(ax: plt.Axes, rows: list[dict[str, str]], xkey: str, ykey: str) -> None:
    for task in TASKS:
        xy = np.asarray([[f(row, xkey), f(row, ykey)] for row in rows if row.get("task_group") == task], dtype=float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 3 or len(np.unique(xy, axis=0)) < 3:
            continue
        try:
            hull = ConvexHull(xy)
        except Exception:
            continue
        color = TASK_COLORS[task]
        ax.add_patch(
            patches.Polygon(
                xy[hull.vertices],
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=0.85,
                alpha=0.018 if task == "L0 DFT Reference" else 0.055,
                zorder=0,
            )
        )


def path_lines(ax: plt.Axes, rows: list[dict[str, str]], xkey: str, ykey: str) -> None:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = row.get("path_key", "")
        if key and key.lower() != "nan":
            groups[key].append(row)
    for path_rows in groups.values():
        ordered = sorted(path_rows, key=lambda row: (f(row, "frame", np.inf), f(row, "sample_frame", np.inf)))
        xy = np.asarray([[f(row, xkey), f(row, ykey)] for row in ordered], dtype=float)
        finite = np.isfinite(xy).all(axis=1)
        if finite.sum() < 2:
            continue
        task = ordered[0].get("task_group", "")
        protocol = ordered[0].get("protocol_style", "")
        ax.plot(
            xy[finite, 0],
            xy[finite, 1],
            color=TASK_COLORS.get(task, "#777777"),
            linewidth=0.7,
            alpha=0.25,
            linestyle="--" if protocol == "self-relaxed" else "-",
            zorder=1,
        )


def separation_metrics(rows: list[dict[str, str]], xkey: str, ykey: str) -> dict[str, float | int]:
    rng = np.random.default_rng(42)
    points = []
    labels = []
    for task in TASKS:
        task_rows = [row for row in rows if row.get("task_group") == task]
        idx = np.arange(len(task_rows))
        if len(idx) > 250:
            idx = rng.choice(idx, 250, replace=False)
        for i in idx:
            x, y = f(task_rows[i], xkey), f(task_rows[i], ykey)
            if np.isfinite([x, y]).all():
                points.append([x, y])
                labels.append(task)
    X = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    centroids = np.asarray([X[labels == task].mean(axis=0) for task in TASKS])
    within = np.concatenate([np.linalg.norm(X[labels == task] - centroids[i], axis=1) for i, task in enumerate(TASKS)])
    centroid_dist = distance.cdist(centroids, centroids)
    between = centroid_dist[np.triu_indices(len(TASKS), 1)]
    nearest_centroid = np.asarray(TASKS)[np.argmin(distance.cdist(X, centroids), axis=1)]
    silhouettes = []
    for i, point in enumerate(X):
        own = np.linalg.norm(X[labels == labels[i]] - point, axis=1).mean()
        other = min(np.linalg.norm(X[labels == task] - point, axis=1).mean() for task in TASKS if task != labels[i])
        silhouettes.append((other - own) / max(own, other, 1e-12))
    return {
        "n_balanced_sample": int(len(X)),
        "min_centroid_distance": float(between.min()),
        "mean_within_task_distance": float(within.mean()),
        "centroid_separation_ratio": float(between.min() / max(within.mean(), 1e-12)),
        "nearest_centroid_accuracy": float(np.mean(nearest_centroid == labels)),
        "approx_balanced_silhouette": float(np.mean(silhouettes)),
    }


def render(rows: list[dict[str, str]], sizes: np.ndarray, out: Path) -> dict[str, dict[str, float | int]]:
    apply_publication_style(font_size=MIN_TEXT_SIZE)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": MIN_TEXT_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "axes.titleweight": "bold",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelweight": "bold",
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(20.0, 10.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.36)
    metrics: dict[str, dict[str, float | int]] = {}
    for index, (name, (xkey, ykey, xlabel, ylabel)) in enumerate(EMBEDDINGS.items()):
        ax = fig.add_subplot(gs[0, index])
        task_hulls(ax, rows, xkey, ykey)
        path_lines(ax, rows, xkey, ykey)
        for task in TASKS:
            for split in SPLIT_ORDER:
                sub = [row for row in rows if row.get("task_group") == task and row.get("split") == split]
                if not sub:
                    continue
                xy = np.asarray([[f(row, xkey), f(row, ykey)] for row in sub], dtype=float)
                point_sizes = np.asarray([sizes[rows.index(row)] for row in sub], dtype=float)
                finite = np.isfinite(xy).all(axis=1) & np.isfinite(point_sizes)
                if not finite.any():
                    continue
                marker = MARKERS[split]
                ax.scatter(
                    xy[finite, 0],
                    xy[finite, 1],
                    s=point_sizes[finite],
                    marker=marker,
                    color=TASK_COLORS[task],
                    alpha=0.24 if split == "train" else 0.72,
                    edgecolors="white" if marker != "." else "none",
                    linewidths=0.22 if marker != "." else 0.0,
                    zorder=2,
                )
        all_xy = np.asarray([[f(row, xkey), f(row, ykey)] for row in rows], dtype=float)
        all_xy = all_xy[np.isfinite(all_xy).all(axis=1)]
        lo, hi = np.nanmin(all_xy, axis=0), np.nanmax(all_xy, axis=0)
        pad = 0.05 * np.maximum(hi - lo, 1e-12)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{chr(65 + index)}  {name}", loc="left")
        style_ax(ax)
        metrics[name] = separation_metrics(rows, xkey, ykey)

    task_handles = [Line2D([], [], marker="o", linestyle="", color=TASK_COLORS[task], label=task, markersize=7) for task in TASKS]
    size_handles = [
        Line2D([], [], marker="o", linestyle="", color="#555555", markerfacecolor="#555555", markeredgecolor="white",
               markersize=float(np.sqrt(size)), label=label)
        for size, label in zip([18.0, 70.0, 190.0], ["near train", "intermediate", "far OOD"])
    ]
    path_handles = [
        Line2D([], [], color="#555555", linewidth=1.1, linestyle="-", label="fixed / sliding path"),
        Line2D([], [], color="#555555", linewidth=1.1, linestyle="--", label="self-relaxed path"),
    ]
    fig.legend(
        handles=[*task_handles, *size_handles, *path_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=False,
        fontsize=LEGEND_SIZE,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.22, top=0.86)
    fig.suptitle("Figure 1 embedding comparison | no post-hoc task translation", x=0.02, ha="left", y=0.975, fontsize=TITLE_SIZE, fontweight="bold")
    assets = save_figure(fig, out, "figure01_embedding_comparison")
    return metrics


def main() -> None:
    rows = read_rows(SRC)
    sizes, reference = ood_sizes(rows)
    metrics = render(rows, sizes, OUT / "figures")
    manifest = {
        "source": str(SRC),
        "coordinates": {name: {"x": xy[0], "y": xy[1], "semantics": "source-provided coordinates; no post-hoc task translation"} for name, xy in EMBEDDINGS.items()},
        "ood_marker_area": "distance_to_training; 98th percentile reference=" + str(reference),
        "separation_metrics": metrics,
        "selection_rule": "compare balanced task separation metrics and visual QA; never optimize by moving task points after embedding",
    }
    out_manifest = OUT / "manifests" / "FIGURE01_EMBEDDING_COMPARISON.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
