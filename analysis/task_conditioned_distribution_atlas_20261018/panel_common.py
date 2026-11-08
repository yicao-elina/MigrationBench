#!/usr/bin/env python3
"""Shared panel recipes for the task-conditioned distribution figure.

Each public ``draw_panel_*`` function is independently callable from a
standalone renderer and from the final composition script.  The composition
script owns only placement and panel lettering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

PLOT_ATLAS = Path("/Users/alina/Documents/Codex/plot_library/src")
if str(PLOT_ATLAS) not in sys.path:
    sys.path.insert(0, str(PLOT_ATLAS))

from plot_atlas import apply_publication_style, get_palette, panel_label  # noqa: E402


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"

MODEL_ORDER = ["Foundation", "Scratch", "FT-600K", "FT-MultiT"]
PAL = get_palette("jhu_mace_models")
MODEL_COLORS = {
    "Foundation": PAL["roles"]["foundation_omat"],
    "Scratch": PAL["roles"]["from_scratch"],
    "FT-600K": PAL["roles"]["ft_600k"],
    "FT-MultiT": PAL["roles"]["ft_multit"],
}
SPLIT_ORDER = ["train", "valid", "test"]
SPLIT_COLORS = {"train": "#A8ADB5", "valid": "#5F7897", "test": "#D34F2D"}
INK = "#222222"
TRAIN_GREY = "#A8ADB5"


def configure_plot_style() -> None:
    """Apply the Plot Atlas house style with final-placement-safe typography."""

    apply_publication_style(font_size=12)
    plt.rcParams.update(
        {
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "#C9C9C9",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_reference() -> pd.DataFrame:
    return pd.read_csv(DERIVED / "grouped_reference_embedding.csv")


def load_static() -> pd.DataFrame:
    return pd.read_csv(DERIVED / "static_error_points.csv")


def load_md() -> pd.DataFrame:
    return pd.read_csv(DERIVED / "md_distribution_points.csv")


def style_axis(ax, *, xlabel=None, ylabel=None, title=None, title_pad=8):
    ax.grid(False)
    ax.tick_params(direction="out", width=1.25, length=5.0, labelsize=12.0)
    for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.25)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=13.0, fontweight="bold", labelpad=7)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=13.0, fontweight="bold", labelpad=7)
    if title is not None:
        ax.set_title(title, fontsize=14.0, fontweight="bold", pad=title_pad)
    return ax


def _kde_grid(x, y, bounds, n=180):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    xx = np.linspace(bounds[0], bounds[1], n)
    yy = np.linspace(bounds[2], bounds[3], n)
    X, Y = np.meshgrid(xx, yy)
    try:
        Z = gaussian_kde(np.vstack([x, y]))(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    except np.linalg.LinAlgError:
        return None
    return X, Y, Z


def _kde_line(values, bounds, n=260):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 8 or np.ptp(values) == 0:
        return None
    grid = np.linspace(bounds[0], bounds[1], n)
    try:
        density = gaussian_kde(values)(grid)
    except np.linalg.LinAlgError:
        return None
    return grid, density


def _soft_density_cmap(color, alpha):
    """Return a transparent-to-muted colormap for a filled KDE surface."""

    rgb = to_rgb(color)
    return LinearSegmentedColormap.from_list(
        f"soft_{color.lstrip('#')}",
        [
            (1.0, 1.0, 1.0, 0.0),
            (*rgb, alpha * 0.16),
            (*rgb, alpha * 0.52),
            (*rgb, alpha),
        ],
    )


def _draw_density(
    ax,
    x,
    y,
    color,
    *,
    xlim,
    ylim,
    fill_alpha=0.34,
    boundary_alpha=0.0,
    zorder=1,
):
    """Draw a soft filled density surface with at most one boundary.

    Filled levels communicate the shape of the configuration-space surface
    without the many nested contour lines that dominated the earlier atlas.
    ``boundary_alpha`` is deliberately zero by default; it is only used for
    the grouped reference envelope where one outer boundary helps orientation.
    """

    result = _kde_grid(x, y, (*xlim, *ylim))
    if result is None:
        return
    X, Y, Z = result
    finite = Z[np.isfinite(Z)]
    if finite.size == 0 or np.nanmax(finite) <= 0:
        return
    Z = Z / np.nanmax(finite)
    levels = np.array([0.14, 0.36, 0.62, 0.88, 0.97])
    if len(levels) >= 2:
        ax.contourf(
            X,
            Y,
            Z,
            levels=levels,
            cmap=_soft_density_cmap(color, fill_alpha),
            norm=Normalize(0.0, 1.0),
            antialiased=True,
            zorder=zorder,
        )
        if boundary_alpha > 0:
            ax.contour(
                X,
                Y,
                Z,
                levels=[0.18],
                colors=[color],
                linewidths=0.85,
                alpha=boundary_alpha,
                zorder=zorder + 1,
            )


def _draw_marginal(ax_top, ax_right, values_x, values_y, color, *, xlim, ylim):
    top = _kde_line(values_x, xlim)
    right = _kde_line(values_y, ylim)
    if top is not None:
        density = top[1].copy()
        density[density < np.nanmax(density) * 0.018] = np.nan
        ax_top.plot(top[0], density, color=color, linewidth=1.55, alpha=0.84, clip_on=True)
    if right is not None:
        density = right[1].copy()
        density[density < np.nanmax(density) * 0.018] = np.nan
        ax_right.plot(density, right[0], color=color, linewidth=1.55, alpha=0.84, clip_on=True)


def _joint_axes(fig, spec):
    nested = spec.subgridspec(
        4,
        4,
        width_ratios=[1, 1, 1, 0.18],
        height_ratios=[0.24, 1, 1, 1],
        wspace=0.0,
        hspace=0.0,
    )
    main = fig.add_subplot(nested[1:, :3])
    top = fig.add_subplot(nested[0, :3], sharex=main)
    right = fig.add_subplot(nested[1:, 3], sharey=main)
    top.tick_params(axis="x", labelbottom=False, bottom=False, labelleft=False, left=False)
    right.tick_params(axis="y", labelleft=False, left=False, labelbottom=False, bottom=False)
    top.set_yticks([])
    right.set_xticks([])
    for marginal in (top, right):
        marginal.grid(False)
        for spine in marginal.spines.values():
            spine.set_visible(False)
    return main, top, right


def _calibrate_marginals(top, right, *, xlim, ylim):
    """Align marginal support exactly to the joint axes and contain peaks."""

    top.set_xlim(*xlim)
    right.set_ylim(*ylim)
    top.relim()
    top.autoscale_view(scalex=False, scaley=True)
    right.relim()
    right.autoscale_view(scalex=True, scaley=False)
    top_ylim = top.get_ylim()
    right_xlim = right.get_xlim()
    top.set_ylim(0.0, max(top_ylim[1] * 1.06, 1e-12))
    right.set_xlim(0.0, max(right_xlim[1] * 1.06, 1e-12))
    top.margins(x=0)
    right.margins(y=0)


def _legend_handles_for_splits(ref):
    return [
        Line2D([0], [0], color=SPLIT_COLORS[split], linewidth=2.0, label=f"{split} ({len(ref.loc[ref.split.eq(split)]):,})")
        for split in SPLIT_ORDER
    ]


def draw_panel_a(fig, spec, *, add_title=True, label=None, legend=True):
    """Draw the grouped reference-space panel with contours and marginals."""

    ref = load_reference()
    main, top, right = _joint_axes(fig, spec)
    xlim = (-2.7, 3.25)
    ylim = (-2.65, 2.75)

    for split in SPLIT_ORDER:
        part = ref.loc[ref.split.eq(split)]
        color = SPLIT_COLORS[split]
        _draw_density(
            main,
            part.pca1,
            part.pca2,
            color,
            xlim=xlim,
            ylim=ylim,
            fill_alpha=0.42 if split != "train" else 0.30,
            boundary_alpha=0.28 if split != "train" else 0.18,
            zorder=1,
        )
        main.scatter(
            part.pca1,
            part.pca2,
            s=8 if split == "train" else 13,
            color=color,
            alpha=0.16 if split == "train" else 0.32,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        _draw_marginal(top, right, part.pca1, part.pca2, color, xlim=xlim, ylim=ylim)

    neb = ref.loc[ref.split.eq("neb")]
    main.scatter(
        neb.pca1,
        neb.pca2,
        s=62,
        marker="D",
        facecolor="white",
        edgecolor=INK,
        linewidth=1.25,
        zorder=6,
    )
    main.set_xlim(*xlim)
    main.set_ylim(*ylim)
    _calibrate_marginals(top, right, xlim=xlim, ylim=ylim)
    style_axis(main, xlabel="SOAP PCA 1", ylabel="SOAP PCA 2")
    if add_title:
        top.set_title("Grouped DFT reference space", fontsize=14.5, fontweight="bold", pad=7)
    if legend:
        handles = _legend_handles_for_splits(ref)
        handles.append(Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=INK, markersize=7, label="fixed-path images"))
        leg = main.legend(
            handles=handles,
            loc="upper left",
            ncol=2,
            fontsize=11.2,
            borderpad=0.45,
            handlelength=1.4,
            columnspacing=0.8,
            handletextpad=0.45,
            frameon=True,
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")
    if label:
        panel_label(main, label, x=-0.10, y=1.08)
    return {"main": main, "top": top, "right": right}


def draw_panel_b(fig, spec, *, add_title=True, label=None, legend=True):
    """Draw static test energy/force RMSE points in the four-family order."""

    static = load_static()
    ax = fig.add_subplot(spec)
    for family in MODEL_ORDER:
        part = static.loc[static.family.eq(family)].copy()
        if part.empty:
            continue
        x = part.energy_rmse_mev_atom.to_numpy(float)
        y = part.force_rmse_mev_A.to_numpy(float)
        marker = "D" if family == "FT-MultiT" else "o"
        ax.scatter(
            x,
            y,
            s=78 if family == "FT-MultiT" else 62,
            marker=marker,
            color=MODEL_COLORS[family],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.92,
            zorder=4,
            label=family,
        )
    ax.set_yscale("log")
    ax.set_xlim(0, 48)
    ax.set_ylim(35, 1400)
    ax.set_xticks([0, 10, 20, 30, 40])
    style_axis(
        ax,
        xlabel="test energy RMSE (meV atom$^{-1}$)",
        ylabel="test force RMSE (meV Å$^{-1}$)",
        title="Static pointwise test RMSE" if add_title else None,
        title_pad=10,
    )
    if legend:
        leg = ax.legend(
            loc="lower left",
            fontsize=11.2,
            borderpad=0.45,
            handletextpad=0.45,
            labelspacing=0.55,
            frameon=True,
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")
    if label:
        panel_label(ax, label, x=-0.10, y=1.08)
    return {"main": ax}


def draw_panel_c(fig, spec, *, add_title=True, label=None, legend=True):
    """Draw the MLIP-generated MD SOAP/PCA cloud over grouped-train density."""

    ref = load_reference()
    md = load_md()
    main, top, right = _joint_axes(fig, spec)
    xlim = (-28.0, 5.0)
    ylim = (-25.0, 10.0)

    train = ref.loc[ref.split.eq("train")]
    _draw_density(
        main,
        train.pca1,
        train.pca2,
        "#7A8088",
        xlim=xlim,
        ylim=ylim,
        fill_alpha=0.26,
        boundary_alpha=0.18,
        zorder=1,
    )
    main.scatter(train.pca1, train.pca2, s=5, color=TRAIN_GREY, alpha=0.08, linewidths=0, rasterized=True, zorder=1)

    for family in MODEL_ORDER:
        part = md.loc[md.family.eq(family)].dropna(subset=["pca1", "pca2"])
        if part.empty:
            continue
        color = MODEL_COLORS[family]
        _draw_density(
            main,
            part.pca1,
            part.pca2,
            color,
            xlim=xlim,
            ylim=ylim,
            fill_alpha=0.24,
            boundary_alpha=0.0,
            zorder=2,
        )
        main.scatter(
            part.pca1,
            part.pca2,
            s=4.5,
            color=color,
            alpha=0.23,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        _draw_marginal(top, right, part.pca1, part.pca2, color, xlim=xlim, ylim=ylim)

    main.set_xlim(*xlim)
    main.set_ylim(*ylim)
    _calibrate_marginals(top, right, xlim=xlim, ylim=ylim)
    main.set_xticks([-25, -15, -5, 5])
    main.set_yticks([-20, -10, 0, 10])
    # Keep the main axis in the exact GridSpec rectangle shared by the top and
    # right marginals.  ``equal/box`` would shrink only the joint axis to
    # enforce a square data box, leaving the marginal axes full-width/full-
    # height and visibly breaking their calibration.  The shared limits still
    # preserve the numerical coordinate mapping used by the marginals.
    main.set_aspect("auto")
    style_axis(main, xlabel="SOAP PCA 1", ylabel="SOAP PCA 2")
    if add_title:
        top.set_title("MLIP-generated 600 K MD configuration space", fontsize=14.5, fontweight="bold", pad=7)
    if legend:
        handles = [Patch(facecolor=TRAIN_GREY, edgecolor="none", alpha=0.75, label="grouped train")]
        handles.extend(
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[family], markeredgecolor="none", markersize=7, label=family)
            for family in MODEL_ORDER
        )
        leg = main.legend(
            handles=handles,
            loc="lower left",
            fontsize=10.9,
            borderpad=0.45,
            handlelength=1.1,
            handletextpad=0.45,
            labelspacing=0.48,
            frameon=True,
        )
        for text in leg.get_texts():
            text.set_fontweight("bold")
    if label:
        panel_label(main, label, x=-0.10, y=1.08)
    return {"main": main, "top": top, "right": right}


def save_figure_formats(fig, stem: Path, *, tight=True):
    stem.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"facecolor": "white", "metadata": {"Creator": "Plot Atlas / MigrationBench"}}
    if tight:
        kwargs.update({"bbox_inches": "tight", "pad_inches": 0.08})
    for ext in ("pdf", "png", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), format=ext, dpi=450, **kwargs)

