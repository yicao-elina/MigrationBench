#!/usr/bin/env python3
"""Render the Top-20 bar + 12-seed heatmap SHAP layout."""

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
from matplotlib.patches import Patch, Rectangle


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "plot_shap_full_campaign.py"
SPEC = importlib.util.spec_from_file_location("shap_base_plot", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(BASE_SCRIPT)
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)

FAMILY_ORDER = BASE.FAMILY_ORDER
STRATEGY_BY_FAMILY = BASE.STRATEGY_BY_FAMILY
TEMPERATURE_BY_FAMILY = BASE.TEMPERATURE_BY_FAMILY
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


def select_features(aggregate: pd.DataFrame, n: int = 20) -> list[str]:
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


def bar_style(family: str) -> dict[str, object]:
    strategy = STRATEGY_BY_FAMILY[family]
    color = STRATEGY_COLORS[strategy]
    return {
        "facecolor": "white" if TEMPERATURE_BY_FAMILY[family] == "600 K" else color,
        "edgecolor": color,
        "linewidth": 0.9,
    }


def draw_top20_bars(ax: plt.Axes, aggregate: pd.DataFrame, features: list[str]) -> None:
    y = np.arange(len(features), dtype=float)
    height = 0.17
    offsets = np.linspace(-1.5 * height, 1.5 * height, len(FAMILY_ORDER))
    xmax = []
    for offset, family in zip(offsets, FAMILY_ORDER):
        data = (
            aggregate[(aggregate.model_family == family) & aggregate.feature.isin(features)]
            .set_index("feature")
            .reindex(features)
        )
        mean = data["mean_abs_shap_meV_per_atom"].to_numpy(dtype=float)
        sd = data["sd_abs_shap_meV_per_atom"].to_numpy(dtype=float)
        xmax.extend((mean + sd).tolist())
        style = bar_style(family)
        ax.barh(
            y + offset,
            mean,
            height=height * 0.88,
            xerr=np.vstack([np.minimum(sd, mean), sd]),
            capsize=2.0,
            error_kw={"elinewidth": 0.85, "ecolor": DFT_COLOR, "capthick": 0.85},
            zorder=3,
            **style,
        )
    ax.set_xlim(0.0, max(xmax) * 1.08)
    ax.set_yticks(y, features)
    ax.tick_params(axis="y", labelsize=TYPE_PT["feature"], pad=4)
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean $|$SHAP value$|$ (meV atom$^{-1}$)", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=7)
    ax.set_ylabel("SOAP / structural feature", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=8)
    ax.axvline(0.0, color=DFT_COLOR, linestyle=(0, (3, 2)), linewidth=1.1, zorder=1)
    style_axes(ax)


def add_panel_a_legends(fig: plt.Figure, *, x_strategy: float = 0.28, x_temperature: float = 0.54, y: float = 0.045) -> None:
    strategy_handles = [
        Patch(facecolor=STRATEGY_COLORS[name], edgecolor=STRATEGY_COLORS[name], label=name)
        for name in ("Fine-tuning", "Scratch")
    ]
    fig.legend(
        handles=strategy_handles,
        labels=[h.get_label() for h in strategy_handles],
        loc="lower center",
        bbox_to_anchor=(x_strategy, y),
        ncol=2,
        frameon=False,
        fontsize=TYPE_PT["legend"],
        prop={"weight": "bold"},
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    temperature_handles = [
        Patch(facecolor="white", edgecolor=STRATEGY_COLORS["Fine-tuning"], linewidth=1.15, label="600 K"),
        Patch(facecolor=STRATEGY_COLORS["Fine-tuning"], edgecolor=STRATEGY_COLORS["Fine-tuning"], linewidth=1.15, label="Multi-T"),
    ]
    legend = fig.legend(
        handles=temperature_handles,
        labels=[h.get_label() for h in temperature_handles],
        loc="lower center",
        bbox_to_anchor=(x_temperature, y),
        ncol=2,
        frameon=False,
        fontsize=TYPE_PT["legend"],
        prop={"weight": "bold"},
        handlelength=1.4,
        handletextpad=0.5,
        columnspacing=0.9,
        title="Panel a: temperature encoding",
    )
    legend.get_title().set_fontsize(11.5)
    legend.get_title().set_fontweight("bold")


def seed_heatmap_matrix(seed_values: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    families: list[str] = []
    for family in FAMILY_ORDER:
        for temperature in ("600 K", "Multi-T"):
            for seed in SEEDS:
                source_family = next(f for f in FAMILY_ORDER if TEMPERATURE_BY_FAMILY[f] == temperature and STRATEGY_BY_FAMILY[f] == STRATEGY_BY_FAMILY[family])
                data = seed_values[(seed_values.model_family == source_family) & (seed_values.seed == seed)].set_index("feature").reindex(features)
                rows.append(data["mean_abs_shap_meV_per_atom"].to_numpy(dtype=float))
                labels.append(f"{temperature} / s{seed}")
                families.append(source_family)
            # only emit each strategy block once; family is used below to map
            # the two temperature groups to the correct source rows.
            if family == "FT-600K":
                break
            if family == "Scratch-600K":
                break
    return np.asarray(rows, dtype=float), labels, families


def seed_heatmap_matrix_ordered(seed_values: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    group_names: list[str] = []
    for strategy in ("Fine-tuning", "Scratch"):
        for temperature in ("600 K", "Multi-T"):
            family = next(f for f in FAMILY_ORDER if STRATEGY_BY_FAMILY[f] == strategy and TEMPERATURE_BY_FAMILY[f] == temperature)
            for seed in SEEDS:
                data = seed_values[(seed_values.model_family == family) & (seed_values.seed == seed)].set_index("feature").reindex(features)
                rows.append(data["mean_abs_shap_meV_per_atom"].to_numpy(dtype=float))
                labels.append(f"{temperature} / s{seed}")
                group_names.append(strategy)
    return np.asarray(rows, dtype=float), labels, group_names


def draw_seed_heatmap(fig: plt.Figure, ax: plt.Axes, seed_values: pd.DataFrame, features: list[str], *, header_x: float) -> None:
    matrix, labels, groups = seed_heatmap_matrix_ordered(seed_values, features)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=SHAP_CMAP, norm=Normalize(vmin=0.0, vmax=float(np.nanmax(matrix))))
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="y", length=0, pad=8, labelsize=10.5)
    for group, text in zip(groups, ax.get_yticklabels()):
        text.set_color(STRATEGY_COLORS[group])
        text.set_fontweight("bold")
    ax.set_xticks(np.arange(len(features)), features)
    ax.tick_params(axis="x", length=0, pad=5, labelsize=8.7)
    for text in ax.get_xticklabels():
        text.set_rotation(90)
        text.set_ha("center")
        text.set_va("top")
        text.set_fontweight("bold")
    ax.set_xlim(-2.75, len(features) - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel("Top 20 features in shared global ranking", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=13)
    style_axes(ax, heatmap=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.15)
    # Large strategy bars and vertical group labels reproduce the hierarchy of
    # the supplied reference matrix while leaving the small row labels to carry
    # temperature and random seed.
    for start, color, group in [(0, STRATEGY_COLORS["Fine-tuning"], "Fine-tuning"), (6, STRATEGY_COLORS["Scratch"], "Scratch")]:
        ax.add_patch(Rectangle((-1.55, start - 0.45), 0.22, 5.9, facecolor=color, edgecolor="none", clip_on=False, zorder=4))
        ax.plot([-1.72, -1.84, -1.84, -1.72], [start - 0.45, start - 0.45, start + 5.45, start + 5.45], color=INK, linewidth=1.8, solid_capstyle="butt", clip_on=False, zorder=5)
        ax.text(-2.18, start + 2.5, group, rotation=90, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK, clip_on=False)
    ax.axhline(5.5, color="white", linewidth=3.0, zorder=5)
    cbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.035)
    cbar.set_label(r"Mean $|$SHAP$|$ (meV atom$^{-1}$)", fontsize=TYPE_PT["axis"], fontweight="bold", labelpad=9)
    cbar.ax.tick_params(labelsize=TYPE_PT["tick"], width=1.0, length=4.0, colors=INK)
    for text in cbar.ax.get_yticklabels():
        text.set_fontweight("bold")
    cbar.outline.set_linewidth(1.1)
    cbar.outline.set_edgecolor(GREY)
    add_header(fig, header_x, "b", "Top 20 feature importance by seed", "Large labels show strategy; rows show temperature / random seed; shared mean |SHAP| color scale")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=450 if path.suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_combined(aggregate: pd.DataFrame, seed_values: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig = plt.figure(figsize=(20.0, 9.1), constrained_layout=False)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.45], wspace=0.14)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.20, right=0.96, top=0.83, bottom=0.25, wspace=0.14)
    draw_top20_bars(ax_a, aggregate, features)
    add_header(fig, 0.20, "a", "Top 20 SHAP features", "Shared global feature ranking; bars are mean ± SD across seeds")
    draw_seed_heatmap(fig, ax_b, seed_values, features, header_x=0.57)
    add_panel_a_legends(fig, x_strategy=0.30, x_temperature=0.48, y=0.045)
    return fig


def make_bar_panel(aggregate: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.5, 9.1), constrained_layout=False)
    fig.subplots_adjust(left=0.27, right=0.98, top=0.83, bottom=0.25)
    draw_top20_bars(ax, aggregate, features)
    add_header(fig, 0.27, "a", "Top 20 SHAP features", "Shared global feature ranking; bars are mean ± SD across seeds")
    add_panel_a_legends(fig, x_strategy=0.38, x_temperature=0.66, y=0.045)
    return fig


def make_heatmap_panel(seed_values: pd.DataFrame, features: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16.8, 9.1), constrained_layout=False)
    fig.subplots_adjust(left=0.31, right=0.90, top=0.83, bottom=0.28)
    draw_seed_heatmap(fig, ax, seed_values, features, header_x=0.31)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure_matplotlib()
    aggregate = pd.read_csv(args.analysis / "shap_family_aggregate.csv")
    seed_values = pd.read_csv(args.analysis / "shap_seed_feature_importance.csv")
    features = select_features(aggregate, 20)
    variant_dir = args.output / "top20_bar_seed_heatmap_20261102"
    variant_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "shap_top20_bar_seed_heatmap": lambda: make_combined(aggregate, seed_values, features),
        "panel_a_top20_strategy_bars": lambda: make_bar_panel(aggregate, features),
        "panel_b_top20_seed_heatmap": lambda: make_heatmap_panel(seed_values, features),
    }
    for suffix in ("pdf", "png", "svg"):
        for name, builder in builders.items():
            save_figure(builder(), variant_dir / f"{name}.{suffix}")
    manifest = {
        "figure_id": "shap_full_campaign_20261101",
        "variant": "top20_bar_seed_heatmap_20261102",
        "description": "Shared Top-20 feature order; panel a strategy-colored open/filled bars; panel b 12-row model-seed heatmap with hierarchical strategy labels.",
        "top20_features_shared_order": features,
        "panel_a_encoding": {
            "strategy_color": STRATEGY_COLORS,
            "temperature": {"600 K": "open bar", "Multi-T": "filled bar"},
            "uncertainty": "mean ± sample SD across three seeds",
        },
        "panel_b_encoding": {
            "fill": "Mean |SHAP| shared continuous blue scale",
            "large_groups": ["Fine-tuning", "Scratch"],
            "small_rows": ["600 K / s123", "600 K / s234", "600 K / s345", "Multi-T / s123", "Multi-T / s234", "Multi-T / s345"],
            "temperature_or_seed_open_filled": False,
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
