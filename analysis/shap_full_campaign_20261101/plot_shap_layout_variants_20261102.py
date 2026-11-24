#!/usr/bin/env python3
"""Render less-dense SHAP layout variants for visual review.

Variant 1 is the proposed publication layout: temperature facets for the
Top-12 dot plot and a Top-20 family heatmap. Variant 2 keeps the same Top-10
temperature-facet dot plot but expands the Top-20 heatmap to all 12
model-seed rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "plot_shap_full_campaign.py"
SPEC = importlib.util.spec_from_file_location("shap_base_plot", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(BASE_SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)


FAMILY_ORDER = BASE.FAMILY_ORDER
STRATEGY_BY_FAMILY = BASE.STRATEGY_BY_FAMILY
STRATEGY_COLORS = BASE.STRATEGY_COLORS
LABELS = BASE.LABELS
SHAP_CMAP = BASE.SHAP_CMAP
DFT_COLOR = BASE.DFT_COLOR
INK = BASE.INK
GREY = BASE.GREY
SUBTLE = BASE.SUBTLE
TYPE_PT = BASE.TYPE_PT
SEEDS = [123, 234, 345]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_features(aggregate: pd.DataFrame, n: int) -> list[str]:
    return BASE.select_features(aggregate, n)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TYPE_PT["tick"],
            "axes.labelsize": TYPE_PT["axis"],
            "axes.linewidth": 1.35,
            "axes.labelcolor": INK,
            "axes.edgecolor": GREY,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.35)
        ax.spines[side].set_color(GREY)
    ax.tick_params(which="major", width=1.25, length=5.0, direction="out", labelsize=TYPE_PT["tick"], colors=INK)
    ax.tick_params(which="minor", width=0.9, length=3.0, direction="out")
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")


def add_header(fig: plt.Figure, x: float, letter: str, title: str, subtitle: str) -> None:
    fig.text(x - 0.035, 0.965, letter, fontsize=TYPE_PT["panel_letter"], fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(x, 0.965, title, fontsize=TYPE_PT["title"], fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(x, 0.932, subtitle, fontsize=TYPE_PT["subtitle"], color=SUBTLE, ha="left", va="bottom")


def strategy_legend(fig: plt.Figure, *, x: float = 0.28, y: float = 0.045) -> None:
    handles = [
        Line2D([], [], color=STRATEGY_COLORS[name], linewidth=3.0, label=name)
        for name in ("Fine-tuning", "Scratch")
    ]
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(x, y),
        ncol=2,
        frameon=False,
        fontsize=TYPE_PT["legend"],
        prop={"weight": "bold"},
        handlelength=2.0,
        handletextpad=0.55,
        columnspacing=1.2,
    )


def draw_temperature_facets(
    fig: plt.Figure,
    axes: list[plt.Axes],
    aggregate: pd.DataFrame,
    features: list[str],
    *,
    header_x: float,
    letter: str,
    title: str,
) -> None:
    temperature_groups = {
        "600 K": ["FT-600K", "Scratch-600K"],
        "Multi-T": ["FT-MultiT", "Scratch-MultiT"],
    }
    upper_bounds: list[float] = []
    for temperature, ax in zip(("600 K", "Multi-T"), axes):
        families = temperature_groups[temperature]
        y = np.arange(len(features), dtype=float)
        offsets = [-0.14, 0.14]
        for offset, family in zip(offsets, families):
            data = (
                aggregate[(aggregate.model_family == family) & aggregate.feature.isin(features)]
                .set_index("feature")
                .reindex(features)
            )
            mean = data["mean_abs_shap_meV_per_atom"].to_numpy(dtype=float)
            sd = data["sd_abs_shap_meV_per_atom"].to_numpy(dtype=float)
            lower = np.minimum(sd, mean)
            upper_bounds.extend((mean + sd).tolist())
            color = STRATEGY_COLORS[STRATEGY_BY_FAMILY[family]]
            ax.errorbar(
                mean,
                y + offset,
                xerr=np.vstack([lower, sd]),
                fmt="o",
                markersize=5.5,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.8,
                ecolor=DFT_COLOR,
                elinewidth=1.15,
                capsize=2.8,
                capthick=1.0,
                linestyle="none",
                zorder=4,
            )
        ax.set_yticks(y, features)
        ax.tick_params(axis="y", labelsize=TYPE_PT["feature"], pad=4)
        if ax is not axes[0]:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.set_ylabel("SOAP / structural feature", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=8)
        ax.invert_yaxis()
        ax.set_title(temperature, fontsize=13.0, fontweight="bold", color=INK, pad=10)
        ax.set_xlabel(r"Mean $|$SHAP value$|$ (meV atom$^{-1}$)", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=7)
        style_axes(ax)
    x_max = max(upper_bounds) * 1.08 if upper_bounds else 1.0
    for ax in axes:
        ax.set_xlim(0.0, x_max)
    add_header(fig, header_x, letter, title, "Temperature is shown by facet; dots are mean ± SD across seeds")


def family_heatmap_matrix(aggregate: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.asarray(
        [
            aggregate[aggregate.model_family == family]
            .set_index("feature")
            .reindex(features)["mean_abs_shap_meV_per_atom"].to_numpy(dtype=float)
            for family in FAMILY_ORDER
        ],
        dtype=float,
    ).T


def draw_family_heatmap(fig: plt.Figure, ax: plt.Axes, aggregate: pd.DataFrame, features: list[str], *, header_x: float, letter: str) -> None:
    matrix = family_heatmap_matrix(aggregate, features)
    vmax = float(np.nanmax(matrix))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=SHAP_CMAP, norm=Normalize(vmin=0.0, vmax=vmax))
    ax.set_yticks(np.arange(len(features)), features)
    ax.tick_params(axis="y", length=0, pad=7, labelsize=TYPE_PT["feature"])
    ax.set_xticks(np.arange(len(FAMILY_ORDER)), ["Fine-tuning\n600 K", "Fine-tuning\nMulti-T", "Scratch\n600 K", "Scratch\nMulti-T"])
    ax.tick_params(axis="x", length=0, pad=7, labelsize=10.2)
    for text in ax.get_xticklabels():
        text.set_fontweight("bold")
    for family, text in zip(FAMILY_ORDER, ax.get_xticklabels()):
        text.set_color(STRATEGY_COLORS[STRATEGY_BY_FAMILY[family]])
    ax.set_xlabel("Model family and temperature", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=13)
    style_axes(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.15)
    cbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
    cbar.set_label(r"Mean $|$SHAP$|$ (meV atom$^{-1}$)", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=9)
    cbar.ax.tick_params(labelsize=TYPE_PT["tick"], width=1.0, length=4.0, colors=INK)
    for text in cbar.ax.get_yticklabels():
        text.set_fontweight("bold")
    cbar.outline.set_linewidth(1.1)
    cbar.outline.set_edgecolor(GREY)
    add_header(fig, header_x, letter, "Top 20 feature importance", "Temperature is written in the column titles; shared mean |SHAP| color scale")


def seed_heatmap_matrix(seed_values: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    families: list[str] = []
    for family in FAMILY_ORDER:
        for seed in SEEDS:
            data = seed_values[(seed_values.model_family == family) & (seed_values.seed == seed)].set_index("feature").reindex(features)
            rows.append(data["mean_abs_shap_meV_per_atom"].to_numpy(dtype=float))
            labels.append(f"{LABELS[family]} · s{seed}")
            families.append(family)
    return np.asarray(rows, dtype=float), labels, families


def draw_seed_heatmap(fig: plt.Figure, ax: plt.Axes, seed_values: pd.DataFrame, features: list[str], *, header_x: float, letter: str) -> None:
    matrix, labels, families = seed_heatmap_matrix(seed_values, features)
    vmax = float(np.nanmax(matrix))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=SHAP_CMAP, norm=Normalize(vmin=0.0, vmax=vmax))
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="y", length=0, pad=8, labelsize=10.0)
    for family, text in zip(families, ax.get_yticklabels()):
        text.set_color(STRATEGY_COLORS[STRATEGY_BY_FAMILY[family]])
        text.set_fontweight("bold")
    ax.set_xticks(np.arange(len(features)), features)
    ax.tick_params(axis="x", length=0, pad=5, labelsize=8.8)
    for text in ax.get_xticklabels():
        text.set_rotation(90)
        text.set_ha("center")
        text.set_va("top")
        text.set_fontweight("bold")
    ax.set_xlabel("Top 20 features in global campaign ranking", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=13)
    style_axes(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.15)
    cbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
    cbar.set_label(r"Mean $|$SHAP$|$ (meV atom$^{-1}$)", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=9)
    cbar.ax.tick_params(labelsize=TYPE_PT["tick"], width=1.0, length=4.0, colors=INK)
    for text in cbar.ax.get_yticklabels():
        text.set_fontweight("bold")
    cbar.outline.set_linewidth(1.1)
    cbar.outline.set_edgecolor(GREY)
    add_header(fig, header_x, letter, "Top 20 feature importance by seed", "Each row is one model-seed run; shared mean |SHAP| color scale")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=450 if path.suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_proposed_a(aggregate: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 8.4), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.83, bottom=0.22, wspace=0.08)
    draw_temperature_facets(fig, list(axes), aggregate, features, header_x=0.22, letter="a", title="Top 12 SHAP features")
    strategy_legend(fig, x=0.50)
    return fig


def make_proposed_b(aggregate: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.5, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.26, right=0.90, top=0.83, bottom=0.27)
    draw_family_heatmap(fig, ax, aggregate, features, header_x=0.26, letter="b")
    return fig


def make_proposed_combined(aggregate: pd.DataFrame, top12: list[str], top20: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(19.0, 8.8), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.94, 0.94, 1.35], wspace=0.09)
    ax_a1 = fig.add_subplot(grid[0, 0])
    ax_a2 = fig.add_subplot(grid[0, 1], sharey=ax_a1)
    ax_b = fig.add_subplot(grid[0, 2])
    fig.subplots_adjust(left=0.18, right=0.965, top=0.83, bottom=0.24, wspace=0.09)
    draw_temperature_facets(fig, [ax_a1, ax_a2], aggregate, top12, header_x=0.18, letter="a", title="Top 12 SHAP features")
    draw_family_heatmap(fig, ax_b, aggregate, top20, header_x=0.60, letter="b")
    strategy_legend(fig, x=0.35)
    return fig


def make_seed_a(aggregate: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.9), sharey=True, constrained_layout=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.83, bottom=0.22, wspace=0.08)
    draw_temperature_facets(fig, list(axes), aggregate, features, header_x=0.22, letter="a", title="Top 10 SHAP features")
    strategy_legend(fig, x=0.50)
    return fig


def make_seed_b(seed_values: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16.8, 8.1), constrained_layout=False)
    fig.subplots_adjust(left=0.24, right=0.90, top=0.83, bottom=0.29)
    draw_seed_heatmap(fig, ax, seed_values, features, header_x=0.24, letter="b")
    return fig


def make_seed_combined(aggregate: pd.DataFrame, seed_values: pd.DataFrame, top10: list[str], top20: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(19.0, 8.8), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.94, 0.94, 1.40], wspace=0.09)
    ax_a1 = fig.add_subplot(grid[0, 0])
    ax_a2 = fig.add_subplot(grid[0, 1], sharey=ax_a1)
    ax_b = fig.add_subplot(grid[0, 2])
    fig.subplots_adjust(left=0.18, right=0.965, top=0.83, bottom=0.25, wspace=0.09)
    draw_temperature_facets(fig, [ax_a1, ax_a2], aggregate, top10, header_x=0.18, letter="a", title="Top 10 SHAP features")
    draw_seed_heatmap(fig, ax_b, seed_values, top20, header_x=0.60, letter="b")
    strategy_legend(fig, x=0.35)
    return fig


def render_variant(name: str, output: Path, metadata: dict[str, object]) -> None:
    variant_dir = output / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        for figure_name, builder in metadata["builders"].items():
            fig = builder()
            save_figure(fig, variant_dir / f"{figure_name}.{suffix}")
    manifest = {k: v for k, v in metadata.items() if k != "builders"}
    manifest["renderer_script"] = str(Path(__file__).resolve())
    manifest["renderer_script_sha256"] = sha256_file(Path(__file__).resolve())
    manifest["files"] = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(variant_dir.iterdir())
        if path.suffix in {".pdf", ".png", ".svg"}
    }
    (variant_dir / "variant_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_matplotlib()
    aggregate = pd.read_csv(args.analysis / "shap_family_aggregate.csv")
    seed_values = pd.read_csv(args.analysis / "shap_seed_feature_importance.csv")
    top12 = select_features(aggregate, 12)
    top10 = select_features(aggregate, 10)
    top20 = select_features(aggregate, 20)

    proposed_builders = {
        "shap_layout_proposed": lambda: make_proposed_combined(aggregate, top12, top20),
        "panel_a_top12_temperature_facets": lambda: make_proposed_a(aggregate, top12),
        "panel_b_top20_family_heatmap": lambda: make_proposed_b(aggregate, top20),
    }
    seed_builders = {
        "shap_layout_seed_expanded": lambda: make_seed_combined(aggregate, seed_values, top10, top20),
        "panel_a_top10_temperature_facets": lambda: make_seed_a(aggregate, top10),
        "panel_b_top20_seed_heatmap": lambda: make_seed_b(seed_values, top20),
    }
    render_variant(
        "proposed_top12_top20",
        args.output,
        {
            "figure_id": "shap_full_campaign_20261101",
            "variant": "proposed_top12_top20",
            "description": "Top-12 temperature-facet dot plot plus Top-20 four-column family heatmap",
            "top12_features": top12,
            "top20_features": top20,
            "temperature_encoding": "facet title; no open/filled encoding in the heatmap",
            "colors": {"strategy": STRATEGY_COLORS, "continuous_shap_palette": ["#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70"]},
            "builders": proposed_builders,
        },
    )
    render_variant(
        "seed_expanded_top10_top20",
        args.output,
        {
            "figure_id": "shap_full_campaign_20261101",
            "variant": "seed_expanded_top10_top20",
            "description": "Top-10 temperature-facet dot plot plus Top-20 heatmap with 12 model-seed rows",
            "top10_features": top10,
            "top20_features": top20,
            "heatmap_rows": [f"{family} · s{seed}" for family in FAMILY_ORDER for seed in SEEDS],
            "temperature_encoding": "facet title for panel a; row label for panel b",
            "colors": {"strategy": STRATEGY_COLORS, "continuous_shap_palette": ["#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70"]},
            "builders": seed_builders,
        },
    )


if __name__ == "__main__":
    main()
