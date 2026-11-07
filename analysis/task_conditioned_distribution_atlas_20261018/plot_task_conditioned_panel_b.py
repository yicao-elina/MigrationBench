#!/usr/bin/env python3
"""Standalone Plot Atlas renderer for panel B: static pointwise RMSE."""

from pathlib import Path

import matplotlib.pyplot as plt

from panel_common import configure_plot_style, draw_panel_b, save_figure_formats


HERE = Path(__file__).resolve().parent


def render(output_dir: Path = HERE / "figures") -> None:
    configure_plot_style()
    fig = plt.figure(figsize=(6.2, 5.0), facecolor="white")
    spec = fig.add_gridspec(1, 1, left=0.18, right=0.98, bottom=0.17, top=0.91)
    draw_panel_b(fig, spec[0], add_title=True, label=None, legend=True)
    save_figure_formats(fig, output_dir / "fig_task_conditioned_panel_b", tight=True)
    plt.close(fig)


if __name__ == "__main__":
    render()
    print("Rendered standalone panel B")

