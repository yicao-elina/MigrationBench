#!/usr/bin/env python3
"""Compose the publication-facing MigrationBench configuration-universe map.

The component views remain available as separate editable/vector outputs. This
figure adds only a restrained, reader-facing composition and uses the same
descriptor, model hues, task markers, and distance colours as those panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull


# Frozen plotting contract; kept local so composition does not import the
# descriptor/registry module and accidentally scan external plugin metadata.
MODEL_COLORS = {
    "Foundation": "#E69F00", "Scratch-600K": "#0072B2", "Scratch-MultiT": "#7B2CBF",
    "FT-600K": "#009E73", "FT-MultiT": "#CC79A7", "DFT": "#202020", "Reference": "#8DA9BF",
}
DISTANCE_COLORS = {
    "near_training": "#B7E3C0", "intermediate": "#FEC44F", "far_from_training": "#D95F0E",
}
MARKERS = {"L0_reference": "o", "L2_DFT_path": "D", "L3_self_relaxed_NEB": "^", "L3_interlayer_self": "s"}


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"
FIGURES = HERE / "figures"
RENDERS = HERE / "blender" / "cluster_renders"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
            "axes.labelsize": 8, "axes.linewidth": 0.8, "xtick.major.size": 3,
            "ytick.major.size": 3, "legend.frameon": False, "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_cluster_hulls(ax, df: pd.DataFrame, x: str, y: str) -> None:
    for _, part in df[df["cluster"].ne("noise")].groupby("cluster"):
        if len(part) < 12:
            continue
        xy = part[[x, y]].to_numpy(float)
        if not np.isfinite(xy).all():
            continue
        try:
            hull = ConvexHull(xy)
        except Exception:
            continue
        color = part["cluster_color"].iloc[0]
        ax.add_patch(Polygon(xy[hull.vertices], closed=True, facecolor=color, edgecolor=color, alpha=0.035, linewidth=0.45, zorder=0))


def scatter_embedding(ax, df: pd.DataFrame, x: str, y: str, title: str, *, hulls: bool = False) -> None:
    if hulls:
        add_cluster_hulls(ax, df, x, y)
    ref = df[df["source_kind"].eq("grouped_reference")]
    ax.scatter(ref[x], ref[y], s=3.0, c="#B8CBD8", alpha=0.20, linewidths=0, rasterized=True, zorder=1)
    for task_layer, marker in MARKERS.items():
        part = df[df["task_layer"].eq(task_layer)]
        for family, group in part.groupby("model_family"):
            color = MODEL_COLORS.get(family, "#666666")
            pending = group["geometry_status"].eq("exploratory_pending_geometry")
            stable = group[~pending]
            if not stable.empty:
                ax.scatter(
                    stable[x], stable[y], s=13 if task_layer != "L2_DFT_path" else 19,
                    marker=marker, c=color, alpha=0.78, edgecolors="white", linewidths=0.22,
                    rasterized=True, zorder=3,
                )
            if not pending.empty:
                pending_group = group[pending]
                ax.scatter(
                    pending_group[x], pending_group[y], s=17, marker=marker, facecolors="none",
                    edgecolors=color, linewidths=0.65, alpha=0.82, rasterized=True, zorder=4,
                )
        for _, group in part.groupby("path_key"):
            if len(group) < 2 or not group["path_key"].iloc[0]:
                continue
            group = group.sort_values("frame")
            color = MODEL_COLORS.get(group["model_family"].iloc[0], "#666666")
            ax.plot(group[x], group[y], color=color, alpha=0.23, linewidth=0.55, zorder=2)
    ax.set_title(title, loc="left", pad=5, fontweight="bold")
    ax.tick_params(direction="out", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)


def annotate_representatives(ax, df: pd.DataFrame, reps: pd.DataFrame, x: str, y: str) -> None:
    for _, rep in reps.iterrows():
        point = df[(df["source_file"].eq(rep["source_file"])) & (df["frame"].eq(rep["source_frame"]))]
        if point.empty:
            continue
        row = point.iloc[0]
        ax.scatter(
            [row[x]], [row[y]], s=62, marker="*", facecolors="white",
            edgecolors=rep["cluster_color"], linewidths=1.0, zorder=7,
        )
        ax.text(
            row[x], row[y], rep["cluster"], ha="center", va="center", fontsize=5.5,
            fontweight="bold", color="#222222", zorder=8,
            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
        )


def make_figure() -> None:
    configure_style()
    df = pd.read_csv(DERIVED / "structure_universe_points.csv")
    reps = pd.read_csv(DERIVED / "cluster_representatives.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16.0, 10.6), dpi=180, facecolor="white")
    grid = fig.add_gridspec(
        3, 4, height_ratios=[1.0, 1.0, 0.74],
        width_ratios=[1.0, 1.0, 1.0, 1.0], hspace=0.34, wspace=0.28,
    )
    main = fig.add_subplot(grid[0:2, 0:2])
    pca = fig.add_subplot(grid[0, 2])
    tsne = fig.add_subplot(grid[1, 2])
    distance = fig.add_subplot(grid[0:2, 3])

    scatter_embedding(main, df, "x_umap", "y_umap", "(a) Local Cr-environment UMAP", hulls=True)
    annotate_representatives(main, df, reps, "x_umap", "y_umap")
    main.set_xlabel("UMAP-1", fontsize=8)
    main.set_ylabel("UMAP-2", fontsize=8)

    scatter_embedding(pca, df, "x_pca", "y_pca", "(b) PCA")
    pca.set_xlabel("PCA-1", fontsize=7)
    pca.set_ylabel("PCA-2", fontsize=7)
    scatter_embedding(tsne, df, "x_tsne", "y_tsne", "(c) t-SNE")
    tsne.set_xlabel("t-SNE-1", fontsize=7)
    tsne.set_ylabel("t-SNE-2", fontsize=7)

    for distance_bin, group in df.groupby("distance_bin", sort=False):
        distance.scatter(
            group["x_umap"], group["y_umap"], s=4.0,
            c=DISTANCE_COLORS.get(distance_bin, "#888888"), alpha=0.42,
            linewidths=0, rasterized=True, label=distance_bin.replace("_", " "),
        )
    distance.set_title("(d) Distance to training manifold", loc="left", pad=5, fontweight="bold")
    distance.set_xlabel("UMAP-1", fontsize=8)
    distance.set_ylabel("UMAP-2", fontsize=8)
    distance.legend(loc="lower right", fontsize=6.5, frameon=False, title="local SOAP distance", title_fontsize=6.5)
    distance.tick_params(direction="out", labelsize=7)
    for spine in distance.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.8)

    # Four representatives intentionally span a reference-like environment,
    # a held-out reference, and a model-generated self-relaxed path.
    preferred = ["C10", "C03", "C09", "C01"]
    chosen = [c for c in preferred if c in set(reps["cluster"])]
    for col, cluster in enumerate(chosen):
        ax = fig.add_subplot(grid[2, col])
        image = RENDERS / cluster / "opaque_rgb" / "cr_local_environment.png"
        if image.exists():
            ax.imshow(plt.imread(image))
        else:
            ax.text(0.5, 0.5, "render pending", ha="center", va="center", fontsize=8, color="#777777")
        ax.axis("off")
        rep = reps[reps["cluster"].eq(cluster)].iloc[0]
        source = str(rep["task_layer"]).replace("L3_", "").replace("L0_", "").replace("_", " ")
        ax.set_title(f"{cluster} · {source}", fontsize=8, loc="left", pad=3)

    model_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[name],
               markeredgecolor="white", markersize=5.5, label=name)
        for name in ["Foundation", "Scratch-600K", "Scratch-MultiT", "FT-600K", "FT-MultiT", "DFT"]
    ]
    task_handles = [
        Line2D([0], [0], marker=MARKERS[name], color="#444444", linestyle="none", markersize=5,
               label=name.replace("L0_", "").replace("L2_", "").replace("L3_", "").replace("_", " "))
        for name in ["L2_DFT_path", "L3_self_relaxed_NEB", "L3_interlayer_self"]
    ]
    fig.legend(
        handles=model_handles + task_handles, loc="upper center", bbox_to_anchor=(0.5, 0.992),
        ncol=9, frameon=False, fontsize=7.2, handletextpad=0.35, columnspacing=0.85,
    )
    fig.suptitle("MigrationBench configuration universe", y=0.965, fontsize=17, fontweight="bold")
    fig.text(
        0.5, 0.018,
        "Common non-periodic Cr-centered SOAP (r₍cut₎=5 Å, n=4, l=4, σ=0.5); UMAP/t-SNE use the same whitened 30-D PCA input. "
        "Colours encode model family; marker shapes encode task layer; open symbols mark pending geometry evidence. "
        "Bottom thumbnails are bright matte coordination ball-and-stick renders from real representative frames.",
        ha="center", va="bottom", fontsize=7.6, color="#444444",
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"migrationbench_universe_publication.{ext}", dpi=420 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
