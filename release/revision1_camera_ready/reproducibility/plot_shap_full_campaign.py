#!/usr/bin/env python3
"""Render the full-campaign SHAP panels with the project visual standard."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


FAMILY_ORDER = ["FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]
STRATEGY_BY_FAMILY = {
    "FT-600K": "Fine-tuning",
    "FT-MultiT": "Fine-tuning",
    "Scratch-600K": "Scratch",
    "Scratch-MultiT": "Scratch",
}
TEMPERATURE_BY_FAMILY = {
    "FT-600K": "600 K",
    "FT-MultiT": "Multi-T",
    "Scratch-600K": "600 K",
    "Scratch-MultiT": "Multi-T",
}

# Project-wide semantic palette. Foundation is not present in this four-family
# SHAP campaign, but remains in the named palette for consistency with the
# other publication figures.
STRATEGY_COLORS = {
    "Foundation": "#8FC7AB",
    "Fine-tuning": "#29A7B6",
    "Scratch": "#327FB1",
}
DFT_COLOR = "#303840"
INK = "#242424"
GREY = "#73869A"
SUBTLE = "#5C6670"
LABELS = {
    "FT-600K": "Fine-tuning · 600 K",
    "FT-MultiT": "Fine-tuning · Multi-T",
    "Scratch-600K": "Scratch · 600 K",
    "Scratch-MultiT": "Scratch · Multi-T",
}

# Continuous SHAP magnitude gets its own blue channel, independent of model
# strategy. The range is shared by all four heatmap rows.
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_blue",
    ["#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70"],
)

TYPE_PT = {
    "panel_letter": 15,
    "title": 14,
    "axis": 12.8,
    "tick": 11.5,
    "legend": 12.8,
    "feature": 10.5,
    "subtitle": 10.5,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def style_axes(ax: plt.Axes, *, heatmap: bool = False) -> None:
    """Apply the shared axis, tick, and spine treatment."""
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.35)
        ax.spines[side].set_color(GREY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(heatmap)
        if heatmap:
            ax.spines[side].set_linewidth(1.15)
            ax.spines[side].set_color(GREY)
    ax.tick_params(
        which="major",
        width=1.25,
        length=5.0,
        direction="out",
        labelsize=TYPE_PT["tick"],
        colors=INK,
    )
    ax.tick_params(which="minor", width=0.9, length=3.0, direction="out")
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")


def panel_heading(ax: plt.Axes, letter: str, title: str, subtitle: str) -> None:
    """Place a panel heading outside the top tick-label band."""
    ax.text(
        0.0,
        1.12,
        title,
        transform=ax.transAxes,
        fontsize=TYPE_PT["title"],
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
    )
    ax.text(
        0.0,
        1.07,
        subtitle,
        transform=ax.transAxes,
        fontsize=TYPE_PT["subtitle"],
        color=SUBTLE,
        va="bottom",
        ha="left",
    )
    ax.text(
        -0.12,
        1.12,
        letter,
        transform=ax.transAxes,
        fontsize=TYPE_PT["panel_letter"],
        fontweight="bold",
        color=INK,
        va="bottom",
        ha="left",
        clip_on=False,
    )


def select_features(aggregate: pd.DataFrame, n: int) -> list[str]:
    order = (
        aggregate[["feature", "global_rank"]]
        .drop_duplicates()
        .sort_values("global_rank")
    )
    return order.head(n)["feature"].tolist()


def bar_style(family: str) -> dict[str, object]:
    strategy = STRATEGY_BY_FAMILY[family]
    color = STRATEGY_COLORS[strategy]
    temperature = TEMPERATURE_BY_FAMILY[family]
    return {
        "facecolor": "white" if temperature == "600 K" else color,
        "edgecolor": color,
        "linewidth": 0.85,
    }


def draw_bar_panel(
    ax: plt.Axes,
    aggregate: pd.DataFrame,
    features: list[str],
    *,
    letter: str = "a",
) -> None:
    y = np.arange(len(features), dtype=float)
    height = 0.18
    offsets = np.linspace(-1.5 * height, 1.5 * height, len(FAMILY_ORDER))
    for offset, family in zip(offsets, FAMILY_ORDER):
        data = (
            aggregate[(aggregate.model_family == family) & aggregate.feature.isin(features)]
            .set_index("feature")
            .reindex(features)
        )
        style = bar_style(family)
        ax.barh(
            y + offset,
            data["mean_abs_shap_meV_per_atom"].to_numpy(),
            height=height * 0.88,
            xerr=data["sd_abs_shap_meV_per_atom"].to_numpy(),
            capsize=2.0,
            error_kw={"elinewidth": 0.85, "ecolor": DFT_COLOR, "capthick": 0.85},
            zorder=3,
            **style,
        )
    ax.axvline(0, color=DFT_COLOR, linestyle=(0, (3, 2)), linewidth=1.15, zorder=1)
    ax.set_yticks(y, features)
    ax.tick_params(axis="y", labelsize=TYPE_PT["feature"], pad=4)
    ax.invert_yaxis()
    ax.set_xlabel(
        r"Mean $|$SHAP value$|$ (meV atom$^{-1}$)",
        fontsize=TYPE_PT["axis"],
        fontweight="bold",
        labelpad=7,
    )
    ax.set_ylabel(
        "SOAP / structural feature",
        fontsize=TYPE_PT["axis"],
        fontweight="bold",
        labelpad=8,
    )
    panel_heading(
        ax,
        letter,
        "Top 20 SHAP features",
        "Mean ± SD across training seeds 123, 234, 345",
    )
    style_axes(ax)


def build_heatmap_matrix(aggregate: pd.DataFrame, features: list[str]) -> np.ndarray:
    rows = []
    for family in FAMILY_ORDER:
        data = (
            aggregate[(aggregate.model_family == family) & aggregate.feature.isin(features)]
            .set_index("feature")
            .reindex(features)
        )
        rows.append(data["mean_abs_shap_meV_per_atom"].to_numpy())
    return np.asarray(rows, dtype=float)


def draw_heatmap_panel(
    ax: plt.Axes,
    aggregate: pd.DataFrame,
    features: list[str],
    *,
    letter: str = "b",
    colorbar_parent: plt.Figure | None = None,
) -> None:
    matrix = build_heatmap_matrix(aggregate, features)
    vmax = float(np.nanmax(matrix))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=SHAP_CMAP,
        norm=Normalize(vmin=0.0, vmax=vmax),
    )
    ax.set_yticks(np.arange(len(FAMILY_ORDER)), [LABELS[f] for f in FAMILY_ORDER])
    ax.tick_params(axis="y", length=0, pad=10, labelsize=TYPE_PT["tick"])
    for family, text in zip(FAMILY_ORDER, ax.get_yticklabels()):
        text.set_color(STRATEGY_COLORS[STRATEGY_BY_FAMILY[family]])
        text.set_fontweight("bold")
    ax.set_xticks(np.arange(len(features)), features)
    ax.tick_params(axis="x", length=0, pad=5, labelsize=8.2)
    for text in ax.get_xticklabels():
        text.set_fontweight("bold")
        text.set_rotation(90)
        text.set_ha("center")
        text.set_va("top")
    ax.set_xlabel(
        "Top 50 features in global campaign ranking",
        fontsize=TYPE_PT["axis"],
        fontweight="bold",
        labelpad=12,
    )
    panel_heading(
        ax,
        letter,
        "Top 50 feature importance",
        "Four model families; shared mean |SHAP| color scale",
    )
    style_axes(ax, heatmap=True)
    parent = colorbar_parent if colorbar_parent is not None else ax.figure
    cbar = parent.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
    cbar.set_label(
        r"Mean $|$SHAP$|$ (meV atom$^{-1}$)",
        fontsize=TYPE_PT["axis"],
        fontweight="bold",
        labelpad=9,
    )
    cbar.ax.tick_params(labelsize=TYPE_PT["tick"], width=1.0, length=4.0, colors=INK)
    for text in cbar.ax.get_yticklabels():
        text.set_fontweight("bold")
    cbar.outline.set_linewidth(1.1)
    cbar.outline.set_edgecolor(GREY)


def strategy_handles() -> list[Line2D]:
    return [
        Line2D([], [], color=STRATEGY_COLORS[name], linewidth=3.0, label=name)
        for name in ("Fine-tuning", "Scratch")
    ]


def temperature_handles() -> list[Patch]:
    return [
        Patch(facecolor="white", edgecolor=DFT_COLOR, linewidth=1.15, label="600 K"),
        Patch(facecolor=DFT_COLOR, edgecolor=DFT_COLOR, linewidth=1.15, label="Multi-T"),
    ]


def add_shared_legend(fig: plt.Figure, *, y: float = 0.045) -> None:
    handles = strategy_handles() + temperature_handles()
    fig.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=4,
        frameon=False,
        fontsize=TYPE_PT["legend"],
        prop={"weight": "bold"},
        handlelength=1.9,
        handletextpad=0.55,
        columnspacing=1.15,
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=450 if path.suffix == ".png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def make_panel_a(aggregate: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.6, 9.2), constrained_layout=False)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.86, bottom=0.22)
    draw_bar_panel(ax, aggregate, features, letter="a")
    add_shared_legend(fig, y=0.045)
    return fig


def make_panel_b(aggregate: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14.0, 7.6), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.90, top=0.86, bottom=0.34)
    draw_heatmap_panel(ax, aggregate, features, letter="b", colorbar_parent=fig)
    return fig


def make_combined(aggregate: pd.DataFrame, top20: list[str], top50: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(18.4, 8.8), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.24], wspace=0.11)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.18, right=0.965, top=0.86, bottom=0.25, wspace=0.11)
    draw_bar_panel(ax_a, aggregate, top20, letter="a")
    draw_heatmap_panel(ax_b, aggregate, top50, letter="b", colorbar_parent=fig)
    add_shared_legend(fig, y=0.045)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": TYPE_PT["tick"],
            "axes.labelsize": TYPE_PT["axis"],
            "axes.titlesize": TYPE_PT["title"],
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
    aggregate = pd.read_csv(args.analysis / "shap_family_aggregate.csv")
    top20 = select_features(aggregate, 20)
    top50 = select_features(aggregate, 50)

    for suffix in ("pdf", "png", "svg"):
        save_figure(make_panel_a(aggregate, top20), args.output / f"fig_shap_full_campaign_panel_a.{suffix}")
        save_figure(make_panel_b(aggregate, top50), args.output / f"fig_shap_full_campaign_panel_b.{suffix}")
        save_figure(make_combined(aggregate, top20, top50), args.output / f"fig_shap_full_campaign.{suffix}")

    manifest = {
        "figure_id": "shap_full_campaign_20261101",
        "status": "RENDERED_WITH_PROJECT_VISUAL_STANDARD",
        "analysis_input": str(args.analysis / "shap_family_aggregate.csv"),
        "top20_features": top20,
        "top50_features": top50,
        "families": FAMILY_ORDER,
        "strategy_by_family": STRATEGY_BY_FAMILY,
        "temperature_encoding": {"600 K": "open", "Multi-T": "filled"},
        "colors": {
            "strategy": STRATEGY_COLORS,
            "reference": DFT_COLOR,
            "continuous_shap": {
                "palette": ["#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70"],
                "scale": "shared across all four model families",
            },
        },
        "typography_pt": TYPE_PT,
        "files": {},
    }
    for path in sorted(args.output.glob("fig_shap_full_campaign*")):
        manifest["files"][path.name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
