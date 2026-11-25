#!/usr/bin/env python3
"""Render a Plot Atlas diagnostic figure for incomplete unstable 1 ns runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_atlas import apply_publication_style, bold_tick_labels, get_palette, panel_label, save_figure


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def val(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(width=1.0, length=5, pad=3)
    bold_tick_labels(ax)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")


def plot_line(ax, rows: list[dict[str, str]], xkey: str, ykey: str, label: str, color: str) -> None:
    x = np.asarray([val(row, xkey) for row in rows])
    y = np.asarray([val(row, ykey) for row in rows])
    mask = np.isfinite(x) & np.isfinite(y)
    ax.plot(x[mask], y[mask], lw=1.35, color=color, label=label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--postprocess-root", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.inventory.open(newline="")))
    rows.sort(key=lambda row: (row["family"], int(row["training_seed"])))
    palette = get_palette("nature_steered_dynamics")["categorical"]
    colors = {row["run_id"]: palette[i % len(palette)] for i, row in enumerate(rows)}
    labels = {
        row["run_id"]: f"{row['family'].replace('ablation_', '')} seed {row['training_seed']}"
        for row in rows
    }
    data = {
        row["run_id"]: {
            "timeseries": read_csv(args.postprocess_root / row["run_id"] / "timeseries.csv"),
            "msd": read_csv(args.postprocess_root / row["run_id"] / "msd.csv"),
            "vacf": read_csv(args.postprocess_root / row["run_id"] / "vacf.csv"),
            "kappa": read_csv(args.postprocess_root / row["run_id"] / "kappa.csv"),
        }
        for row in rows
    }

    apply_publication_style(9.5)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), constrained_layout=True)
    axes = axes.ravel()

    for row in rows:
        run_id = row["run_id"]
        color = colors[run_id]
        label = labels[run_id]
        ts = data[run_id]["timeseries"]
        plot_line(axes[0], ts, "time_ps", "temperature_K", label, color)
        plot_line(axes[1], ts, "time_ps", "max_force_eV_A", label, color)
        plot_line(axes[2], ts, "time_ps", "min_distance_A", label, color)
        plot_line(axes[4], data[run_id]["vacf"], "lag_ps", "vacf_normalized", label, color)
        plot_line(axes[5], data[run_id]["kappa"], "time_ps", "kappa_W_mK", label, color)

    # Terminal runs have different lengths, so render each MSD trace against its
    # own lag axis.  This preserves the observed terminal extent instead of
    # forcing the registered equal-grid summary primitive to truncate data.
    for row in rows:
        run_id = row["run_id"]
        plot_line(axes[3], data[run_id]["msd"], "lag_ps", "msd_xy_A2", labels[run_id], colors[run_id])
    style(axes[3], "Lag time (ps)", "Cr MSD$_{xy}$ (Å$^2$)")

    style(axes[0], "Absolute time (ps)", "Temperature (K)")
    style(axes[1], "Absolute time (ps)", "Max |F| (eV Å$^{-1}$)")
    style(axes[2], "Absolute time (ps)", "Minimum distance (Å)")
    style(axes[4], "Lag time (ps)", "Normalized VACF")
    style(axes[5], "Integration time (ps)", "Running κ diagnostic (W m$^{-1}$ K$^{-1}$)")
    axes[0].axhline(600.0, color="#333333", lw=0.9, ls=":")
    axes[1].set_yscale("log")
    axes[4].axhline(0.0, color="#333333", lw=0.8, ls=":")
    axes[5].axhline(0.0, color="#333333", lw=0.8, ls=":")
    axes[0].legend(frameon=True, fontsize=7, ncol=1, loc="upper left")
    axes[5].set_ylim(bottom=min(-0.05, float(np.nanpercentile(np.concatenate([
        np.asarray([val(item, "kappa_W_mK") for item in data[row["run_id"]]["kappa"]]) for row in rows
    ]), 1.0))))

    titles = [
        "Thermal stability",
        "Force-instability diagnostic",
        "Geometry diagnostic",
        "Cr displacement (diagnostic)",
        "Cr velocity dynamics",
        "Running κ (diagnostic)",
    ]
    for ax, title, letter in zip(axes, titles, "abcdef"):
        ax.set_title(title, fontsize=10, fontweight="normal")
        panel_label(ax, letter)
    axes[1].text(
        0.98, 0.98, "ALL TERMINAL\nFAILURE_MODE", transform=axes[1].transAxes,
        ha="right", va="top", fontsize=8, fontweight="bold", color="#9b1c1c",
        bbox={"facecolor": "white", "edgecolor": "#9b1c1c", "linewidth": 0.8, "pad": 3.0},
    )
    fig.suptitle(
        "From-scratch ablation MD: terminal instability diagnostics — BLOCKED, no formal 1 ns transport",
        fontsize=12, fontweight="bold",
    )

    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        save_figure(fig, args.output / f"terminal_failure_diagnostic.{ext}")
    plt.close(fig)

    manifest = {
        "figure": "terminal_failure_diagnostic",
        "status": "BLOCKED",
        "scientific_boundary": "finite trajectories are retained for failure diagnostics; no formal Dxy, kappa, hop rate, or replica uncertainty claim",
        "source_postprocess_root": str(args.postprocess_root),
        "source_inventory": str(args.inventory),
        "runs": [row["run_id"] for row in rows],
        "panels": {
            "a": "temperature_vs_absolute_time",
            "b": "max_force_vs_absolute_time_log_scale",
            "c": "minimum_distance_vs_absolute_time",
            "d": "Cr_MSD_xy_vs_lag_time_direct_lines_preserving_each_terminal_extent",
            "e": "Cr_normalized_VACF",
            "f": "running_kappa_stress_based_diagnostic",
        },
        "atlas_palette": "nature_steered_dynamics",
        "render_formats": ["png", "svg", "pdf"],
    }
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "caption_and_status.md").write_text(
        "# Terminal failure diagnostic figure\n\n"
        "This Plot Atlas figure is exploratory/diagnostic and blocked for publication transport claims. "
        "All three selected 1 ns branches terminated at `instability_threshold` before 1,000,000 steps. "
        "Finite temperature, force, geometry, MSD, VACF, and stress-based running-kappa traces are shown "
        "to document failure onset and the available finite evidence; they are not converged transport observables.\n"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
