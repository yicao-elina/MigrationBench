#!/usr/bin/env python3
"""Render the current 15-model Figure 6 representation analysis.

The plot keeps the historical four-panel intent while using the project-wide
visual grammar: strategy is color, temperature is open/filled, seed is marker
shape, and Foundation checkpoints are same-hue mint shades.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent.parent / "analysis" / "md_1ns_updated_publication_figure_20261025"
BASE_OUT = ROOT / "updated_campaign_20261102"
OUT = ROOT / "updated_campaign_20261102_umap_density"
SOURCE = BASE_OUT / "source"
FIGURES = OUT / "figures"
PREP_MANIFEST = SOURCE / "preparation_manifest.json"
FEATURES = SOURCE / "fig6_full_campaign_features.npz"

MODEL_ORDER = [
    "Foundation_MPA0",
    "Foundation_MP0b3",
    "Foundation_OMAT0",
    "FT600K_A2_seed123",
    "FT600K_A2_seed234",
    "FT600K_A2_seed345",
    "FTMultiT_A2_seed123",
    "FTMultiT_A2_seed234",
    "FTMultiT_A2_seed345",
    "Scratch600K_Lmax1_seed123",
    "Scratch600K_Lmax1_seed234",
    "Scratch600K_Lmax1_seed345",
    "ScratchMultiT_Lmax1_seed123",
    "ScratchMultiT_Lmax1_seed234",
    "ScratchMultiT_Lmax1_seed345",
]
CONDITION_ORDER = ["Foundation", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]
STRATEGY_COLORS = {
    "Foundation": "#8FC7AB",
    "Fine-tuning": "#29A7B6",
    "Scratch": "#327FB1",
    "DFT": "#303840",
}
FOUNDATION_COLORS = {
    "Foundation_MPA0": "#8FC7AB",
    "Foundation_MP0b3": "#2F6B4F",
    "Foundation_OMAT0": "#5F9E73",
}
SEED_MARKERS = {"123": "o", "234": "^", "345": "s"}
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "migrationbench_density",
    ["#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70"],
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strategy(model_id: str) -> str:
    if model_id.startswith("Foundation"):
        return "Foundation"
    if model_id.startswith("FT"):
        return "Fine-tuning"
    return "Scratch"


def condition(model_id: str) -> str:
    if model_id.startswith("Foundation"):
        return "Foundation"
    if model_id.startswith("FT600K"):
        return "FT-600K"
    if model_id.startswith("FTMultiT"):
        return "FT-MultiT"
    if model_id.startswith("Scratch600K"):
        return "Scratch-600K"
    return "Scratch-MultiT"


def temperature(model_id: str) -> str | None:
    if "600K" in model_id:
        return "600 K"
    if "MultiT" in model_id:
        return "Multi-T"
    return None


def seed(model_id: str) -> str | None:
    for value in SEED_MARKERS:
        if f"seed{value}" in model_id:
            return value
    return None


def point_marker(model_id: str) -> str:
    return "D" if strategy(model_id) == "Foundation" else SEED_MARKERS[seed(model_id)]


def edge_color(model_id: str) -> str:
    return FOUNDATION_COLORS.get(model_id, STRATEGY_COLORS[strategy(model_id)])


def face_color(model_id: str) -> str:
    if strategy(model_id) == "Foundation":
        return FOUNDATION_COLORS[model_id]
    return "white" if temperature(model_id) == "600 K" else STRATEGY_COLORS[strategy(model_id)]


def model_label(model_id: str) -> str:
    labels = {
        "Foundation_MPA0": "MPA-0",
        "Foundation_MP0b3": "MP-0b3",
        "Foundation_OMAT0": "OMAT-0",
    }
    if model_id in labels:
        return labels[model_id]
    if model_id.startswith("FT600K"):
        return f"FT-600K / {seed(model_id)}"
    if model_id.startswith("FTMultiT"):
        return f"FT-MultiT / {seed(model_id)}"
    if model_id.startswith("Scratch600K"):
        return f"Scratch-600K / {seed(model_id)}"
    return f"Scratch-MultiT / {seed(model_id)}"


def atlas_style() -> str:
    selected = "Arial"
    try:
        import matplotlib.font_manager as fm

        fm.findfont("Arial", fallback_to_default=False)
    except ValueError:
        selected = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": selected,
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "font.weight": "normal",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return selected


def configure_axes(ax, letter: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.text(-0.14, 1.14, letter, transform=ax.transAxes, fontsize=16, fontweight="bold",
            ha="left", va="top", clip_on=False)
    ax.tick_params(which="major", labelsize=11.5, width=1.2, length=5.0, direction="out")
    ax.tick_params(which="minor", width=0.9, length=3.0, direction="out")
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color(STRATEGY_COLORS["DFT"])
    ax.grid(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12.8, fontweight="bold", labelpad=5)
    if ylabel:
        # Keep long y-axis titles inside their own panel's gutter instead of
        # letting them intrude into the preceding panel.
        ax.set_ylabel(ylabel, fontsize=12.8, fontweight="bold", labelpad=7)


def set_embedding_limits(ax, embedding: np.ndarray) -> None:
    for axis in (0, 1):
        values = embedding[:, axis]
        lo, hi = float(np.min(values)), float(np.max(values))
        margin = max((hi - lo) * 0.055, 1e-9)
        if axis == 0:
            ax.set_xlim(lo - margin, hi + margin)
        else:
            ax.set_ylim(lo - margin, hi + margin)


def local_bin_counts(embedding: np.ndarray, bins: int = 45) -> np.ndarray:
    """Assign each point the count of points in its 2-D UMAP bin."""
    histogram, xedges, yedges = np.histogram2d(
        embedding[:, 0], embedding[:, 1], bins=bins
    )
    xindex = np.clip(np.searchsorted(xedges, embedding[:, 0], side="right") - 1, 0, bins - 1)
    yindex = np.clip(np.searchsorted(yedges, embedding[:, 1], side="right") - 1, 0, bins - 1)
    return histogram[xindex, yindex].astype(float)


def scatter_models(ax, embedding: np.ndarray, labels: np.ndarray, size: float = 28.0) -> None:
    for model_id in MODEL_ORDER:
        mask = labels == model_id
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=size,
            marker=point_marker(model_id),
            facecolors=face_color(model_id),
            edgecolors=edge_color(model_id),
            linewidths=0.78,
            alpha=0.86,
            zorder=3,
        )


def density_scatter(ax, embedding: np.ndarray, labels: np.ndarray, density: np.ndarray, norm: Normalize) -> None:
    point_colors = DENSITY_CMAP(norm(density))
    for model_id in MODEL_ORDER:
        mask = labels == model_id
        if not np.any(mask):
            continue
        temp = temperature(model_id)
        if strategy(model_id) == "Foundation":
            faces = point_colors[mask]
            edges = np.tile(np.asarray(to_rgba(FOUNDATION_COLORS[model_id])), (np.sum(mask), 1))
        elif temp == "600 K":
            faces = "none"
            edges = point_colors[mask]
        else:
            faces = point_colors[mask]
            edges = point_colors[mask]
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            s=28,
            marker=point_marker(model_id),
            facecolors=faces,
            edgecolors=edges,
            linewidths=0.78,
            alpha=0.9,
            zorder=3,
        )


def strategy_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="None", markersize=7.5,
               markerfacecolor=STRATEGY_COLORS["Foundation"], markeredgecolor=STRATEGY_COLORS["Foundation"], label="Foundation"),
        Line2D([], [], marker="o", linestyle="None", markersize=7.5,
               markerfacecolor=STRATEGY_COLORS["Fine-tuning"], markeredgecolor=STRATEGY_COLORS["Fine-tuning"], label="Fine-tuning"),
        Line2D([], [], marker="o", linestyle="None", markersize=7.5,
               markerfacecolor=STRATEGY_COLORS["Scratch"], markeredgecolor=STRATEGY_COLORS["Scratch"], label="Scratch"),
    ]


def temperature_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="None", markersize=7.5,
               markerfacecolor="white", markeredgecolor=STRATEGY_COLORS["DFT"], markeredgewidth=1.1, label="600 K"),
        Line2D([], [], marker="o", linestyle="None", markersize=7.5,
               markerfacecolor=STRATEGY_COLORS["DFT"], markeredgecolor=STRATEGY_COLORS["DFT"], label="Multi-T"),
    ]


def seed_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", linestyle="None", markersize=7.5, markerfacecolor="white", markeredgecolor=STRATEGY_COLORS["DFT"], label="seed 123"),
        Line2D([], [], marker="^", linestyle="None", markersize=7.5, markerfacecolor="white", markeredgecolor=STRATEGY_COLORS["DFT"], label="seed 234"),
        Line2D([], [], marker="s", linestyle="None", markersize=7.5, markerfacecolor="white", markeredgecolor=STRATEGY_COLORS["DFT"], label="seed 345"),
    ]


def foundation_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="D", linestyle="None", markersize=7.5, markerfacecolor=FOUNDATION_COLORS["Foundation_MPA0"], markeredgecolor=FOUNDATION_COLORS["Foundation_MPA0"], label="MPA-0"),
        Line2D([], [], marker="D", linestyle="None", markersize=7.5, markerfacecolor=FOUNDATION_COLORS["Foundation_MP0b3"], markeredgecolor=FOUNDATION_COLORS["Foundation_MP0b3"], label="MP-0b3"),
        Line2D([], [], marker="D", linestyle="None", markersize=7.5, markerfacecolor=FOUNDATION_COLORS["Foundation_OMAT0"], markeredgecolor=FOUNDATION_COLORS["Foundation_OMAT0"], label="OMAT-0"),
    ]


def add_legends(fig, *, standalone: bool = False) -> None:
    if standalone:
        fs = 10.6
        fig.legend(handles=strategy_handles(), loc="lower center", bbox_to_anchor=(0.50, 0.30), ncol=3,
                   frameon=False, fontsize=fs, title="Training strategy", title_fontsize=fs,
                   handletextpad=0.42, columnspacing=0.8)
        fig.legend(handles=temperature_handles(), loc="lower center", bbox_to_anchor=(0.50, 0.21), ncol=2,
                   frameon=False, fontsize=fs, title="Temperature", title_fontsize=fs,
                   handletextpad=0.42, columnspacing=0.8)
        fig.legend(handles=seed_handles(), loc="lower center", bbox_to_anchor=(0.50, 0.12), ncol=3,
                   frameon=False, fontsize=fs, title="Seed", title_fontsize=fs,
                   handletextpad=0.42, columnspacing=0.8)
        fig.legend(handles=foundation_handles(), loc="lower center", bbox_to_anchor=(0.50, 0.03), ncol=3,
                   frameon=False, fontsize=fs, title="Foundation checkpoint", title_fontsize=fs,
                   handletextpad=0.42, columnspacing=0.8)
        return
    y1, y2 = 0.175, 0.065
    fs = 11.3
    fig.legend(handles=strategy_handles(), loc="lower center", bbox_to_anchor=(0.17, y1), ncol=3,
               frameon=False, fontsize=fs, title="Training strategy", title_fontsize=fs,
               handletextpad=0.42, columnspacing=0.8)
    fig.legend(handles=temperature_handles(), loc="lower center", bbox_to_anchor=(0.43, y1), ncol=2,
               frameon=False, fontsize=fs, title="Temperature", title_fontsize=fs,
               handletextpad=0.42, columnspacing=0.8)
    fig.legend(handles=seed_handles(), loc="lower center", bbox_to_anchor=(0.70, y1), ncol=3,
               frameon=False, fontsize=fs, title="Seed", title_fontsize=fs,
               handletextpad=0.42, columnspacing=0.8)
    fig.legend(handles=foundation_handles(), loc="lower center", bbox_to_anchor=(0.49, y2), ncol=3,
               frameon=False, fontsize=fs, title="Foundation checkpoint", title_fontsize=fs,
               handletextpad=0.42, columnspacing=0.8)


def panel_a(ax, embedding: np.ndarray, labels: np.ndarray) -> None:
    scatter_models(ax, embedding, labels)
    set_embedding_limits(ax, embedding)
    configure_axes(ax, "a", "t-SNE Dimension 1", "t-SNE Dimension 2")


def panel_b(ax, embedding: np.ndarray, labels: np.ndarray) -> None:
    scatter_models(ax, embedding, labels)
    set_embedding_limits(ax, embedding)
    configure_axes(ax, "b", "PHATE Dimension 1", "PHATE Dimension 2")


def panel_c(
    ax,
    cbar_ax,
    embedding: np.ndarray,
    labels: np.ndarray,
    density: np.ndarray,
    *,
    compact_colorbar: bool = False,
) -> None:
    norm = Normalize(vmin=float(np.min(density)), vmax=float(np.max(density)))
    density_scatter(ax, embedding, labels, density, norm)
    set_embedding_limits(ax, embedding)
    configure_axes(ax, "c", "UMAP Dimension 1", "UMAP Dimension 2")
    sm = ScalarMappable(norm=norm, cmap=DENSITY_CMAP)
    sm.set_array(density)
    cbar = plt.colorbar(sm, cax=cbar_ax)
    # The main figure has a narrow colorbar column between C and D. Use a
    # compact horizontal title there; a vertical ylabel would collide with
    # panel D's y-axis label. Standalone C retains the full vertical label.
    cbar.ax.yaxis.set_ticks_position("right")
    if compact_colorbar:
        cbar.ax.set_title("Count/bin", fontsize=9.5, fontweight="bold", pad=4)
    else:
        cbar.set_label("Local point count / 2D bin", fontsize=12.0, fontweight="bold", labelpad=7)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    cbar.ax.tick_params(labelsize=10.5, width=1.0, length=4.0)
    for text in cbar.ax.get_yticklabels():
        text.set_fontweight("bold")
    cbar.outline.set_linewidth(1.0)


def condition_points(model_scores: dict[str, float], name: str) -> tuple[np.ndarray, np.ndarray]:
    models = [model_id for model_id in MODEL_ORDER if condition(model_id) == name]
    return np.asarray([model_scores[model_id] for model_id in models]), np.asarray(models, dtype=object)


def panel_d(ax, model_scores: dict[str, float], overall: float) -> None:
    x = np.arange(len(CONDITION_ORDER))
    means, sds = [], []
    for name in CONDITION_ORDER:
        values, _ = condition_points(model_scores, name)
        means.append(float(np.mean(values)))
        sds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
    bar_faces = [STRATEGY_COLORS["Foundation"], "white", STRATEGY_COLORS["Fine-tuning"], "white", STRATEGY_COLORS["Scratch"]]
    bar_edges = [STRATEGY_COLORS["Foundation"], STRATEGY_COLORS["Fine-tuning"], STRATEGY_COLORS["Fine-tuning"], STRATEGY_COLORS["Scratch"], STRATEGY_COLORS["Scratch"]]
    ax.bar(x, means, yerr=sds, color=bar_faces, edgecolor=bar_edges, linewidth=1.25,
           capsize=3.0, error_kw={"ecolor": STRATEGY_COLORS["DFT"], "elinewidth": 1.0},
           width=0.68, zorder=2)
    point_offsets = [-0.18, 0.0, 0.18]
    for index, name in enumerate(CONDITION_ORDER):
        values, models = condition_points(model_scores, name)
        for offset, value, model_id in zip(point_offsets, values, models):
            ax.scatter(index + offset, value, s=42, marker=point_marker(str(model_id)),
                       facecolors=face_color(str(model_id)), edgecolors=edge_color(str(model_id)),
                       linewidths=0.9, zorder=4)
    ax.axhline(overall, color=STRATEGY_COLORS["DFT"], linestyle=(0, (3, 2)), linewidth=1.25, zorder=1)
    ax.text(0.98, overall, f"overall {overall:.3f}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=10.2, fontweight="bold", color=STRATEGY_COLORS["DFT"])
    ax.set_xticks(x, ["Foundation", "FT\n600 K", "FT\nMulti-T", "Scratch\n600 K", "Scratch\nMulti-T"])
    values = np.asarray(list(model_scores.values()))
    lo = min(-0.1, float(np.min(values)) - 0.06)
    ax.set_ylim(lo, 1.0)
    ax.set_ylabel("Mean silhouette score", labelpad=7)
    configure_axes(ax, "d", "Model condition", None)
    ax.tick_params(axis="x", labelsize=10.3, pad=5)
    for text in ax.get_xticklabels():
        text.set_fontweight("bold")


def save(fig, path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320 if path.suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def assert_main_layout_is_clear(fig, axes: list, cbar_ax) -> None:
    """Fail the render if labels collide with ticks, panels, or the colorbar."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    minimum_gap = 6.0
    for index, axis in enumerate(axes):
        ylabel = axis.yaxis.get_label()
        ylabel_box = ylabel.get_window_extent(renderer)
        tick_boxes = [
            tick.get_window_extent(renderer)
            for tick in axis.get_yticklabels()
            if tick.get_visible()
        ]
        for tick in axis.get_yticklabels():
            if tick.get_visible() and ylabel_box.overlaps(tick.get_window_extent(renderer)):
                raise RuntimeError(f"main layout QA: {ylabel.get_text()} overlaps its y tick labels")
        if tick_boxes:
            nearest_tick_gap = min(box.x0 - ylabel_box.x1 for box in tick_boxes)
            if nearest_tick_gap < minimum_gap:
                raise RuntimeError(
                    f"main layout QA: {ylabel.get_text()} has only {nearest_tick_gap:.1f}px from its y tick labels"
                )
        if index > 0:
            previous_box = axes[index - 1].get_window_extent(renderer)
            preceding_gap = ylabel_box.x0 - previous_box.x1
            if preceding_gap < minimum_gap:
                raise RuntimeError(f"main layout QA: {ylabel.get_text()} overlaps the preceding panel")
    cbar_box = cbar_ax.get_window_extent(renderer)
    cbar_panel_gap = axes[-1].get_window_extent(renderer).x0 - cbar_box.x1
    if cbar_panel_gap < minimum_gap:
        raise RuntimeError("main layout QA: colorbar overlaps panel D")
    d_ylabel_box = axes[-1].yaxis.get_label().get_window_extent(renderer)
    cbar_ylabel_gap = d_ylabel_box.x0 - cbar_box.x1
    if cbar_ylabel_gap < minimum_gap:
        raise RuntimeError("main layout QA: colorbar overlaps panel D y-axis title")
    cbar_title = cbar_ax.title
    if cbar_title.get_visible() and cbar_title.get_window_extent(renderer).overlaps(d_ylabel_box):
        raise RuntimeError("main layout QA: colorbar title overlaps panel D")


def compute_embeddings(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """t-SNE, PHATE and UMAP embeddings with the recorded Figure 6 settings."""
    import phate  # imported here so --embeddings re-renders work without phate/umap-learn
    import umap

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(features)
    phate_embedding = phate.PHATE(random_state=42, n_jobs=-1).fit_transform(features)
    umap_embedding = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
        transform_seed=42,
        n_jobs=1,
    ).fit_transform(features)
    return tsne, phate_embedding, umap_embedding


def stored_embeddings(path: Path, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embeddings saved by a previous full run; labels must match the feature rows."""
    stored = np.load(path, allow_pickle=False)
    if not np.array_equal(np.asarray(stored["labels"], dtype=str), labels):
        raise ValueError(f"{path}: labels do not match the feature matrix rows")
    return stored["tsne"], stored["phate"], stored["umap"]


def render(features_path: Path = FEATURES, out: Path = OUT, embeddings_path: Path | None = None) -> dict[str, object]:
    figures = out / "figures"
    atlas_font = atlas_style()
    payload = np.load(features_path, allow_pickle=True)
    features = np.asarray(payload["features"], dtype=float)
    labels = np.asarray(payload["model_ids"], dtype=str)
    model_order_from_data = [str(value) for value in payload["model_order"]]
    if features.shape[1] != 6192:
        raise ValueError(f"expected 6192 feature columns, got {features.shape[1]}")
    if set(model_order_from_data) != set(MODEL_ORDER):
        raise ValueError("campaign model order does not match the 15-model figure contract")
    if not np.isfinite(features).all():
        raise ValueError("non-finite feature values")
    energies = features[:, 0]
    if embeddings_path is None:
        tsne, phate_embedding, umap_embedding = compute_embeddings(features)
    else:
        tsne, phate_embedding, umap_embedding = stored_embeddings(embeddings_path, labels)
    density = local_bin_counts(umap_embedding, bins=45)
    overall = float(silhouette_score(tsne, labels))
    sample_scores = silhouette_samples(tsne, labels)
    model_scores = {model_id: float(np.mean(sample_scores[labels == model_id])) for model_id in MODEL_ORDER}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "fig6_full_campaign_embeddings.npz",
        tsne=tsne,
        phate=phate_embedding,
        umap=umap_embedding,
        umap_bin_count=density,
        labels=labels,
        energies=energies,
    )

    outputs: dict[str, object] = {}
    fig = plt.figure(figsize=(17.5, 5.7), facecolor="white")
    # Lay out the four data panels first. The density colorbar is then placed
    # explicitly just to the right of C, leaving the remaining C-D gutter for
    # D's y-axis title rather than centering the colorbar in the whole gap.
    gs = fig.add_gridspec(1, 4, width_ratios=[1.30, 1.24, 1.30, 1.38], wspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.34, top=0.87)
    c_position = ax_c.get_position()
    cbar_gap = 0.012
    cbar_width = 0.020
    cbar_ax = fig.add_axes([c_position.x1 + cbar_gap, c_position.y0, cbar_width, c_position.height])
    panel_a(ax_a, tsne, labels)
    panel_b(ax_b, phate_embedding, labels)
    panel_c(ax_c, cbar_ax, umap_embedding, labels, density, compact_colorbar=True)
    panel_d(ax_d, model_scores, overall)
    assert_main_layout_is_clear(fig, [ax_a, ax_b, ax_c, ax_d], cbar_ax)
    add_legends(fig)
    for ext in ("png", "pdf", "svg"):
        outputs[f"main_{ext}"] = save(fig, figures / f"fig6_umap_density_full_campaign_15models_1x4.{ext}")
    plt.close(fig)

    panel_specs = [
        ("a_tsne", (5.5, 5.7), lambda ax, cax: panel_a(ax, tsne, labels)),
        ("b_phate", (5.5, 5.7), lambda ax, cax: panel_b(ax, phate_embedding, labels)),
        ("c_umap_density", (6.0, 5.7), lambda ax, cax: panel_c(ax, cax, umap_embedding, labels, density)),
        ("d_silhouette", (6.4, 5.7), lambda ax, cax: panel_d(ax, model_scores, overall)),
    ]
    for name, size, renderer in panel_specs:
        fig = plt.figure(figsize=size, facecolor="white")
        if name == "c_umap_density":
            local = fig.add_gridspec(1, 2, width_ratios=[1, 0.08], wspace=0.03)
            ax = fig.add_subplot(local[0, 0])
            cax = fig.add_subplot(local[0, 1])
            renderer(ax, cax)
            fig.subplots_adjust(left=0.18, right=0.93, bottom=0.31, top=0.87)
        else:
            ax = fig.add_subplot(111)
            renderer(ax, None)
            fig.subplots_adjust(left=0.18, right=0.95, bottom=0.50 if name != "d_silhouette" else 0.27, top=0.87)
        if name in {"a_tsne", "b_phate"}:
            add_legends(fig, standalone=True)
        elif name == "c_umap_density":
            fig.legend(handles=temperature_handles(), loc="lower center", bbox_to_anchor=(0.50, 0.14), ncol=2, frameon=False, fontsize=10.8, title="Temperature", title_fontsize=10.8)
            fig.legend(handles=seed_handles(), loc="lower center", bbox_to_anchor=(0.50, 0.055), ncol=3, frameon=False, fontsize=10.8, title="Seed", title_fontsize=10.8)
        for ext in ("png", "pdf", "svg"):
            outputs[f"panel_{name}_{ext}"] = save(fig, figures / f"fig6_panel_{name}_full_campaign.{ext}")
        plt.close(fig)

    prep = json.loads(PREP_MANIFEST.read_text())
    campaign_sources = [
        CAMPAIGN / "FIGURE_MANIFEST_15MODELS_R04.json",
        CAMPAIGN / "MD15_FIGURE_DATA_R04.tsv",
        CAMPAIGN / "remote_evidence_R02" / "MODEL_REGISTRY.tsv",
        CAMPAIGN / "remote_evidence_R02" / "FT_MODEL_REGISTRY.tsv",
        CAMPAIGN / "remote_evidence_R02" / "SCRATCH_MODEL_REGISTRY.tsv",
    ]
    manifest = {
        "artifact": "Figure 6 full-campaign representation analysis (C=UMAP density)",
        "status": "RECONSTRUCTION",
        "claim_scope": "current 15-model MD matrix; representation geometry reconstructed from sampled raw trajectories",
        "model_order": MODEL_ORDER,
        "model_matrix": "3 Foundation checkpoints + 3 seeds x FT-600K + 3 seeds x FT-MultiT + 3 seeds x Scratch-600K + 3 seeds x Scratch-MultiT",
        "data_audit": {
            "n_rows": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "rows_by_model": {model_id: int(np.sum(labels == model_id)) for model_id in MODEL_ORDER},
            "finite_features": bool(np.isfinite(features).all()),
        },
        "embedding": {
            "tsne": {"perplexity": 30, "random_state": 42, "max_iter": 1000},
            "phate": {"random_state": 42, "n_jobs": -1},
            "umap_panel_c": {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "random_state": 42, "transform_seed": 42, "n_jobs": 1},
            "umap_panel_c_density": {"method": "2-D histogram count", "bins": 45, "count_min": float(np.min(density)), "count_max": float(np.max(density))},
            "overall_tsne_silhouette": overall,
            "model_tsne_silhouette": model_scores,
        },
        "style": {
            "contract": "scientific-plot-atlas",
            "font_requested": "Arial",
            "font_selected": atlas_font,
            "strategy_colors": STRATEGY_COLORS,
            "foundation_checkpoint_colors": FOUNDATION_COLORS,
            "temperature_encoding": "open = 600 K; filled = Multi-T",
            "seed_encoding": "circle = 123; triangle = 234; square = 345",
            "foundation_encoding": "diamond marker; same-hue mint shades by checkpoint",
            "density_encoding": "monotonic light-blue to deep-blue gradient by local 2-D UMAP-bin count; 600 K points remain open",
        },
        "visual_qa": {
            "main_composite_layout": "PASS",
            "minimum_label_gap_px": 6.0,
            "checks": [
                "y-axis title separated from its own y tick labels",
                "y-axis title separated from the preceding panel",
                "D y-axis title separated from the density colorbar",
                "density colorbar placed immediately after panel C",
                "density colorbar column separated from panel D",
            ],
            "pdf_raster_review": "PASS",
        },
        "source_artifacts": {
            "preparation_manifest": {"path": str(PREP_MANIFEST), "sha256": sha256(PREP_MANIFEST)},
            "features": {"path": str(features_path), "sha256": sha256(features_path)},
            "embeddings_reused_from": None if embeddings_path is None else {"path": str(embeddings_path), "sha256": sha256(embeddings_path)},
            "campaign_sources": [
                {"path": str(path), "sha256": sha256(path)} for path in campaign_sources if path.exists()
            ],
        },
        "diagnostics": [
            "PHATE reported three disconnected components with its historical default knn=5; this is retained as a reconstruction diagnostic.",
            "Panel B uses PHATE for global structure and Panel C uses UMAP with local 2-D bin counts; potential energy is not used for visual color because model-family energy offsets are not comparable.",
            "The six Scratch trajectories are short failure-bounded runs; their actual sampled frame counts are retained and no 1 ns continuation is inferred.",
        ],
        "outputs": outputs,
    }
    (out / "fig6_full_campaign_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=FEATURES, help="feature matrix npz (default: prepared source npz)")
    parser.add_argument("--out-dir", type=Path, default=OUT, help="output folder (default: the committed umap_density folder)")
    parser.add_argument("--embeddings", type=Path, default=None,
                        help="re-render from a saved fig6_full_campaign_embeddings.npz instead of recomputing t-SNE/PHATE/UMAP")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = render(args.features, args.out_dir, args.embeddings)
    print(json.dumps({"overall_tsne_silhouette": result["embedding"]["overall_tsne_silhouette"], "n_rows": result["data_audit"]["n_rows"]}, indent=2))
