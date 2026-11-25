#!/usr/bin/env python
"""Render 20 ps screening figures with the reusable Scientific Plot Atlas."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_atlas import apply_publication_style, save_figure
from plot_atlas.molecular_dynamics import steered_dynamics_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [row for row in json.loads(args.input.read_text()) if row.get("scientific_gate") != "blocked"]
    if not results:
        raise SystemExit("no nonblocked screening runs")
    grouped = defaultdict(list)
    for row in results:
        grouped[row["family"]].append(row)
    palette = "nature_steered_dynamics"
    series, errors, labels = {}, {}, {}
    for family, rows in grouped.items():
        x = np.asarray(rows[0]["msd_xy_A2"]["lag_ps"], float)
        curves = np.asarray([row["msd_xy_A2"]["values"] for row in rows], float)
        series[family] = curves.mean(axis=0)
        errors[family] = curves.std(axis=0, ddof=0) / np.sqrt(len(rows))
        labels[family] = family.replace("ablation_", "").replace("_", " ")
    xmax = max(float(np.max(row["msd_xy_A2"]["lag_ps"])) for row in results)
    ymax = max(float(np.max(values + errors[key])) for key, values in series.items()) * 1.08

    apply_publication_style(11)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    steered_dynamics_panel(
        axes[0],
        x,
        series,
        errors=errors,
        labels=labels,
        palette=palette,
        xlim=(0.0, xmax),
        ylim=(0.0, max(ymax, 0.01)),
        xticks=(0, 2, 4, 6, 8, 10),
        yticks=tuple(np.linspace(0.0, max(ymax, 0.01), 5)),
        xlabel="Lag time (ps)",
        ylabel="Cr MSD$_{xy}$ (A$^2$)",
    )

    block_values = defaultdict(lambda: defaultdict(list))
    for row in results:
        for block in row["blocks_5ps"]:
            value = block.get("Dxy_cm2_s")
            if value is not None:
                block_values[row["family"]][int(block["block"])].append(float(value))
    for family, blocks in block_values.items():
        xs = sorted(blocks)
        means = [np.mean(blocks[index]) for index in xs]
        errs = [np.std(blocks[index], ddof=0) / np.sqrt(len(blocks[index])) for index in xs]
        axes[1].errorbar(xs, means, yerr=errs, marker="o", lw=1.8, capsize=3, label=labels[family])
    axes[1].set(xlabel="5 ps block", ylabel="Screening D$_{xy}$ (cm$^2$ s$^{-1}$)")
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].legend(frameon=True, fontsize=8)
    for axis in axes:
        axis.tick_params(width=1.0, length=5)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        save_figure(fig, args.output / f"screening_plot_atlas_msd_dxy.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
