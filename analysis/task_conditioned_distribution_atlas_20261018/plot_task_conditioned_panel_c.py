#!/usr/bin/env python3
"""Standalone Plot Atlas renderer for panel C: MLIP-generated MD space."""

from pathlib import Path

import matplotlib.pyplot as plt

from panel_common import configure_plot_style, draw_panel_c, save_figure_formats


HERE = Path(__file__).resolve().parent


def render(output_dir: Path = HERE / "figures") -> None:
    configure_plot_style()
    fig = plt.figure(figsize=(6.6, 5.8), facecolor="white")
    spec = fig.add_gridspec(1, 1, left=0.15, right=0.98, bottom=0.14, top=0.93)
    draw_panel_c(fig, spec[0], add_title=True, label=None, legend=True)
    save_figure_formats(fig, output_dir / "fig_task_conditioned_panel_c", tight=True)
    plt.close(fig)


if __name__ == "__main__":
    render()
    print("Rendered standalone panel C")

