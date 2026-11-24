#!/usr/bin/env python3
"""Render the Top-20 bar + transposed model/seed SHAP heatmap layout."""

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
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "plot_shap_top20_bar_seed_heatmap_20261102.py"
SPEC = importlib.util.spec_from_file_location("shap_top20_base_plot", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(BASE_SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

FAMILY_ORDER = BASE.FAMILY_ORDER
STRATEGY_BY_FAMILY = BASE.STRATEGY_BY_FAMILY
TEMPERATURE_BY_FAMILY = BASE.TEMPERATURE_BY_FAMILY
STRATEGY_COLORS = BASE.STRATEGY_COLORS
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


def configure_matplotlib() -> None:
    BASE.configure_matplotlib()


def feature_seed_matrix(seed_values: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    columns: list[np.ndarray] = []
    labels: list[str] = []
    strategies: list[str] = []
    for family in FAMILY_ORDER:
        strategy = STRATEGY_BY_FAMILY[family]
        temperature = TEMPERATURE_BY_FAMILY[family]
        for seed in SEEDS:
            data = (
                seed_values[(seed_values.model_family == family) & (seed_values.seed == seed)]
                .set_index("feature")
                .reindex(features)
            )
            columns.append(data["mean_abs_shap_meV_per_atom"].to_numpy(dtype=float))
            labels.append(f"{family}\ns{seed}")
            strategies.append(strategy)
    return np.column_stack(columns), labels, strategies


def style_axes(ax: plt.Axes, *, heatmap: bool = False) -> None:
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
    ax.tick_params(which="major", width=1.25, length=5.0, direction="out", labelsize=TYPE_PT["tick"], colors=INK)
    ax.tick_params(which="minor", width=0.9, length=3.0, direction="out")
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontweight("bold")


def add_header(fig: plt.Figure, x: float, letter: str, title: str, subtitle: str) -> None:
    fig.text(x - 0.035, 0.965, letter, fontsize=TYPE_PT["panel_letter"], fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(x, 0.965, title, fontsize=TYPE_PT["title"], fontweight="bold", color=INK, ha="left", va="bottom")
    fig.text(x, 0.932, subtitle, fontsize=TYPE_PT["subtitle"], color=SUBTLE, ha="left", va="bottom")


def draw_transposed_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    seed_values: pd.DataFrame,
    features: list[str],
    *,
    header_x: float,
    panel_letter: str = "b",
) -> None:
    matrix, column_labels, strategies = feature_seed_matrix(seed_values, features)
    vmax = float(np.nanmax(matrix))
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=SHAP_CMAP,
        norm=Normalize(vmin=0.0, vmax=vmax),
    )
    ax.set_yticks(np.arange(len(features)), features)
    ax.tick_params(axis="y", labelsize=TYPE_PT["feature"], pad=6)
    for text in ax.get_yticklabels():
        text.set_fontweight("bold")
    ax.set_xticks(np.arange(len(column_labels)), column_labels)
    ax.tick_params(axis="x", length=0, pad=8, labelsize=7.6)
    for text, strategy in zip(ax.get_xticklabels(), strategies):
        text.set_fontweight("bold")
        text.set_color(STRATEGY_COLORS[strategy])
        text.set_rotation(40)
        text.set_ha("right")
        text.set_va("top")
    ax.set_xlim(-0.5, len(column_labels) - 0.5)
    # Keep the heatmap itself on the exact 20-row data extent so its feature
    # centers align with panel a. Group headers are drawn outside this extent.
    ax.set_ylim(len(features) - 0.5, -0.5)
    ax.set_ylabel("SOAP / structural feature", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=9)
    ax.set_xlabel("Model condition and random seed", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=38)
    style_axes(ax, heatmap=True)

    # The upper framed labels encode the major strategy groups. They live in
    # axes coordinates above the matrix, so they do not consume heatmap rows
    # or create a white block inside the data area.
    for start, color, group in [
        (0, STRATEGY_COLORS["Fine-tuning"], "Fine-tuning"),
        (6, STRATEGY_COLORS["Scratch"], "Scratch"),
    ]:
        ax.add_patch(
            Rectangle(
                (start - 0.5, 1.02),
                6.0,
                0.085,
                facecolor=color,
                edgecolor=INK,
                linewidth=0.9,
                transform=ax.get_xaxis_transform(),
                clip_on=False,
                zorder=4,
            )
        )
        ax.text(
            start + 2.5,
            1.062,
            group,
            ha="center",
            va="center",
            fontsize=TYPE_PT["legend"],
            fontweight="bold",
            color="white",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=6,
        )

    # White separators distinguish the two major groups and the four model
    # conditions without introducing a decorative grid.
    ax.axvline(5.5, color="white", linewidth=3.0, zorder=5)
    for boundary in (2.5, 8.5):
        ax.axvline(boundary, color="white", linewidth=1.8, zorder=5)

    cbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
    cbar.set_label(r"Mean $|$SHAP$|$ (meV atom$^{-1}$)", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=9)
    cbar.ax.tick_params(labelsize=TYPE_PT["tick"], width=1.0, length=4.0, colors=INK)
    for text in cbar.ax.get_yticklabels():
        text.set_fontweight("bold")
    cbar.outline.set_linewidth(1.1)
    cbar.outline.set_edgecolor(GREY)
    add_header(
        fig,
        header_x,
        panel_letter,
        "Top 20 feature importance by model and seed",
        "Rows are features; columns retain model condition and random seed; top bars show training strategy",
    )


def make_combined(aggregate: pd.DataFrame, seed_values: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(20.0, 9.6), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.45], wspace=0.14)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.20, right=0.96, top=0.83, bottom=0.29, wspace=0.14)
    BASE.draw_top20_bars(ax_a, aggregate, features)
    add_header(fig, 0.20, "a", "Top 20 SHAP features", "Shared global feature ranking; bars are mean ± SD across seeds")
    draw_transposed_heatmap(fig, ax_b, seed_values, features, header_x=0.57)
    BASE.add_panel_a_legends(fig, x_strategy=0.30, x_temperature=0.48, y=0.045)
    return fig


def make_bar_panel(aggregate: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 9.1), constrained_layout=False)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.83, bottom=0.25)
    BASE.draw_top20_bars(ax, aggregate, features)
    add_header(fig, 0.27, "a", "Top 20 SHAP features", "Shared global feature ranking; bars are mean ± SD across seeds")
    BASE.add_panel_a_legends(fig, x_strategy=0.38, x_temperature=0.66, y=0.045)
    return fig


def make_heatmap_panel(seed_values: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16.8, 9.6), constrained_layout=False)
    fig.subplots_adjust(left=0.27, right=0.90, top=0.83, bottom=0.33)
    draw_transposed_heatmap(fig, ax, seed_values, features, header_x=0.27)
    return fig


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=450 if path.suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_matplotlib()
    aggregate = pd.read_csv(args.analysis / "shap_family_aggregate.csv")
    seed_values = pd.read_csv(args.analysis / "shap_seed_feature_importance.csv")
    features = BASE.select_features(aggregate, 20)
    variant_dir = args.output / "top20_bar_transposed_seed_heatmap_20261102"
    variant_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "shap_top20_bar_transposed_seed_heatmap": lambda: make_combined(aggregate, seed_values, features),
        "panel_a_top20_strategy_bars": lambda: make_bar_panel(aggregate, features),
        "panel_b_top20_transposed_seed_heatmap": lambda: make_heatmap_panel(seed_values, features),
    }
    for suffix in ("pdf", "png", "svg"):
        for name, builder in builders.items():
            save_figure(builder(), variant_dir / f"{name}.{suffix}")

    manifest = {
        "figure_id": "shap_full_campaign_20261101",
        "variant": "top20_bar_transposed_seed_heatmap_20261102",
        "description": "Shared Top-20 feature order with row-aligned panels; panel a keeps strategy-colored open/filled bars; panel b uses a normal 20-row feature heatmap with model-condition/seed columns and framed strategy headers drawn outside the heatmap extent.",
        "top20_features_shared_order": features,
        "panel_a_encoding": {
            "strategy_color": STRATEGY_COLORS,
            "temperature": {"600 K": "open bar", "Multi-T": "filled bar"},
            "uncertainty": "mean ± sample SD across three seeds",
        },
        "panel_b_encoding": {
            "matrix_shape": {"feature_rows": 20, "model_condition_seed_columns": 12},
            "column_order": [f"{family} / s{seed}" for family in FAMILY_ORDER for seed in SEEDS],
            "large_groups": {"Fine-tuning": "columns 1-6", "Scratch": "columns 7-12"},
            "headers": "framed color blocks above the matrix, outside the 20 feature rows",
            "feature_row_alignment_with_panel_a": True,
            "small_labels": "model condition and random seed shown below the matrix",
            "fill": "Mean |SHAP| shared continuous blue scale",
        },
        "colors": {
            "strategy": STRATEGY_COLORS,
            "reference": DFT_COLOR,
            "continuous_shap_palette": ["#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70"],
        },
        "renderer_script": str(Path(__file__).resolve()),
        "renderer_script_sha256": sha256_file(Path(__file__).resolve()),
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(variant_dir.iterdir())
            if path.suffix in {".pdf", ".png", ".svg"}
        },
    }
    (variant_dir / "variant_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
