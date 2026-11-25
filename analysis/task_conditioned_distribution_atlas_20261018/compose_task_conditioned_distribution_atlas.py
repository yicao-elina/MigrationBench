#!/usr/bin/env python3
"""Compose the independently rendered distribution panels into a readable atlas.

The original-training RMSE figure is intentionally kept as a separate
four-model figure so that the distribution atlas does not compress a distinct
training-history diagnostic into an unreadable middle panel.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from panel_common import (
    configure_plot_style,
    draw_panel_a,
    draw_panel_c,
    save_figure_formats,
)


HERE = Path(__file__).resolve().parent


def render(output_dir: Path = HERE / "figures") -> None:
    configure_plot_style()
    fig = plt.figure(figsize=(6.9, 9.7), facecolor="white")
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.0],
        hspace=0.34,
        left=0.15,
        right=0.98,
        bottom=0.025,
        top=0.985,
    )
    draw_panel_a(fig, grid[0], add_title=True, label="a", legend=True)
    draw_panel_c(fig, grid[1], add_title=True, label="b", legend=True)
    save_figure_formats(fig, output_dir / "fig_task_conditioned_distribution_atlas", tight=False)
    plt.close(fig)


if __name__ == "__main__":
    render()
    print("Rendered stacked task-conditioned distribution atlas")

