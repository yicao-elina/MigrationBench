#!/usr/bin/env python3
"""Render a visibly labelled no-data placeholder, never synthetic science."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_atlas.charts import clustered_heatmap
from plot_atlas.style import apply_publication_style, clean_axes, save_figure


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    apply_publication_style(font_size=10)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.0, 3.3), gridspec_kw={"width_ratios": [1.15, 1.0]})
    matrix = np.full((4, 4), np.nan)
    clustered_heatmap(ax0, matrix, ["Foundation", "Cr-FT", "Cr-Scratch", "Multi-FT"], ["Cu", "Li", "Na", "Cr"], cluster=False, colorbar=False)
    ax0.set_title("Model × element error heatmap", pad=10)
    ax0.text(1.5, 1.5, "NO DFT-GATED\nSCIENTIFIC DATA", ha="center", va="center", fontsize=12, fontweight="bold", color="#555555")
    ax0.set_xlabel("Element")
    ax0.set_ylabel("Model")
    ax1.axis("off")
    ax1.text(0.02, 0.88, "Task 2 health gate", fontsize=13, fontweight="bold", transform=ax1.transAxes)
    ax1.text(0.02, 0.66, "DFT paths accepted", fontsize=10, transform=ax1.transAxes)
    ax1.text(0.98, 0.66, "0 / 10", fontsize=16, fontweight="bold", ha="right", transform=ax1.transAxes)
    ax1.text(0.02, 0.58, "Reason", fontsize=10, transform=ax1.transAxes)
    ax1.text(0.02, 0.43, "SCF convergence warning\non at least one image\nfor every path", fontsize=10, transform=ax1.transAxes, color="#8B1E1E")
    ax1.text(0.02, 0.10, "Placeholder only — no model ranking, barrier,\nforce, transfer-gain, basin, or seed claim.", fontsize=9, transform=ax1.transAxes, color="#555555")
    clean_axes(ax0)
    fig.tight_layout(w_pad=2.0)
    out = ROOT / "figures/placeholder_dft_gate_status.png"
    save_figure(fig, out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", metadata={"Creator": "scientific-plot-atlas", "Date": None})
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
