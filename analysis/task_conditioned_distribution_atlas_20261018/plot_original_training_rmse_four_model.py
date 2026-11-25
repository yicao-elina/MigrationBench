#!/usr/bin/env python3
"""Render original Rockfish validation RMSE curves as a four-model atlas."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from panel_common import MODEL_COLORS, MODEL_ORDER, configure_plot_style, save_figure_formats, style_axis


HERE = Path(__file__).resolve().parent
DERIVED = HERE / "derived"


def render(output_dir: Path = HERE / "figures") -> None:
    configure_plot_style()
    data = pd.read_csv(DERIVED / "rockfish_training_rmse_eval.csv")
    data = data.loc[data.plot_include.astype(bool)].copy()

    fig = plt.figure(figsize=(11.4, 6.7), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        left=0.085,
        right=0.995,
        bottom=0.13,
        top=0.86,
        wspace=0.28,
        hspace=0.36,
    )

    for col, family in enumerate(MODEL_ORDER):
        part = data.loc[data.family.eq(family)].sort_values("record_index")
        color = MODEL_COLORS[family]
        ax_e = fig.add_subplot(grid[0, col])
        ax_f = fig.add_subplot(grid[1, col])
        for ax, column, ylabel, linestyle in (
            (ax_e, "rmse_energy_meV_atom", "energy RMSE (meV atom$^{-1}$)", "-"),
            (ax_f, "rmse_force_meV_A", "force RMSE (meV Å$^{-1}$)", "--"),
        ):
            ax.plot(
                part.epoch,
                part[column],
                color=color,
                linewidth=2.0,
                linestyle=linestyle,
                marker="o",
                markersize=3.6,
                markeredgecolor="white",
                markeredgewidth=0.55,
                alpha=0.96,
                zorder=3,
            )
            ax.set_yscale("log")
            ax.set_xlim(-65, 1000)
            ax.set_xticks([0, 250, 500, 750, 1000])
            if col > 0:
                ax.set_yticklabels([])
            style_axis(
                ax,
                xlabel="training epoch" if ax is ax_f else None,
                ylabel=ylabel if column == 0 else None,
            )
            ax.tick_params(axis="x", labelsize=10.5)
            ax.tick_params(axis="y", labelsize=10.5)

        ax_e.set_ylim(0.1, 5000)
        ax_f.set_ylim(5, 600)
        ax_e.set_title(family, fontsize=13.0, fontweight="bold", color=color, pad=6)

    # The two rows identify the metric directly; omitting a separate legend
    # keeps the four-column comparison open and prevents title collisions.
    fig.text(0.5, 0.965, "Original Rockfish validation RMSE checkpoints", ha="center", va="center", fontsize=14.0, fontweight="bold")
    save_figure_formats(fig, output_dir / "fig_original_training_rmse_four_model", tight=False)
    plt.close(fig)


if __name__ == "__main__":
    from parse_rockfish_training_rmse import main as parse_main

    parse_main()
    render()
    print("Rendered original Rockfish four-model RMSE figure")
