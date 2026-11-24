#!/usr/bin/env python3
"""Render the updated publication-style 1 ns MD summary figure.

The figure is deliberately evidence-bounded: the FT families have accepted
1 ns trajectories, Foundation is shown as a provisional comparator, and the
Scratch families are represented only by their observed failure horizons.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULTS = OUT / "results"
DATA_ROOT = ROOT / "analysis" / "mace_1ns_merged_20261010" / "corrected_v2"
OLD_ROOT = ROOT / "analysis" / "mace_1ns_publication_figure_20261005" / "data"

PALETTE = {
    "Foundation": "#CF4520",
    "FT-600K": "#002E71",
    "FT-MultiT": "#67ACE5",
    "Scratch-600K": "#FF9F1B",
    "Scratch-MultiT": "#F4C95D",
}

MODEL_FILES = {
    "FT-600K": [
        ("seed123", OLD_ROOT / "2-naive-fine-tuning"),
        ("seed234", DATA_ROOT / "FT-600K-seed234"),
        ("seed345", DATA_ROOT / "FT-600K-seed345"),
    ],
    "FT-MultiT": [
        ("seed123", OLD_ROOT / "3-multi_T-fine-tuning"),
        ("seed234", DATA_ROOT / "FT-MultiT-seed234"),
        ("seed345", DATA_ROOT / "FT-MultiT-seed345"),
    ],
    "Foundation*": [
        ("OMAT0-medium", DATA_ROOT / "Foundation-medium-mpa-0"),
        ("MP-0b3-medium", DATA_ROOT / "MP-0b3-medium"),
    ],
}

SCRATCH_FAILURES = {
    "Scratch-600K": {"seed123": 204.4, "seed234": 200.8, "seed345": 200.4},
    "Scratch-MultiT": {"seed123": 201.2, "seed234": 200.4, "seed345": 200.4},
}


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(rows, key):
    out = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
        else:
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def load_series(folder: Path, kind: str):
    if kind == "temperature":
        rows = read_csv(folder / ("timeseries.csv" if folder.parent == OLD_ROOT else "corrected_timeseries.csv"))
        x = numeric(rows, "time_ps")
        y = numeric(rows, "temperature_K")
    elif kind == "msd":
        rows = read_csv(folder / ("msd.csv" if folder.parent == OLD_ROOT else "corrected_msd.csv"))
        x = numeric(rows, "lag_ps")
        y = numeric(rows, "msd_A2")
    elif kind == "vacf":
        rows = read_csv(folder / ("vacf.csv" if folder.parent == OLD_ROOT else "corrected_vacf.csv"))
        x = numeric(rows, "lag_ps")
        y = numeric(rows, "vacf_normalized")
    else:
        raise ValueError(kind)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def common_grid(kind: str):
    if kind == "temperature":
        return np.linspace(0.0, 1000.0, 1001)
    return np.linspace(0.0, 800.0, 801)


def aggregate(family: str, kind: str):
    curves = []
    for _, folder in MODEL_FILES[family]:
        x, y = load_series(folder, kind)
        grid = common_grid(kind)
        curves.append(np.interp(grid, x, y))
    arr = np.vstack(curves)
    return grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), curves


def dxy_from_msd(folder: Path, lo: float, hi: float):
    x, y = load_series(folder, "msd")
    mask = (x >= lo) & (x <= hi)
    slope = np.polyfit(x[mask], y[mask], 1)[0]
    # MSD = 4 D_xy t; A^2/ps -> cm^2/s is 1e-4.
    return slope / 4.0 * 1e-4


def family_dxy(family: str, lo: float, hi: float):
    values = [dxy_from_msd(folder, lo, hi) for _, folder in MODEL_FILES[family]]
    return float(np.mean(values)), float(np.std(values, ddof=1) if len(values) > 1 else 0.0), values


def style_axes(ax, title, letter, xlabel=None, ylabel=None):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    ax.tick_params(axis="both", which="major", labelsize=9, width=1.2, length=5)
    ax.tick_params(axis="both", which="minor", width=0.9, length=3)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("#333333")
    ax.grid(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold", labelpad=5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", labelpad=5)


def line_with_band(ax, x, mean, sd, color, label, alpha=0.18):
    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=2.0, label=label)


def build():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 1.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.7), dpi=400)
    axes = axes.ravel()

    # A: thermal stability
    ax = axes[0]
    for family in ["FT-600K", "FT-MultiT", "Foundation*"]:
        x, mean, sd, _ = aggregate(family, "temperature")
        label = "Foundation" if family == "Foundation*" else family
        line_with_band(ax, x, mean, sd, PALETTE[label], label)
    ax.axhline(600, color="#666666", linestyle=(0, (3, 2)), linewidth=1.0, zorder=0)
    ax.set_xlim(0, 1000)
    ax.set_ylim(200, 720)
    style_axes(ax, "Temperature stability", "A", "Time (ps)", "Temperature (K)")

    # B: MSD ensemble
    ax = axes[1]
    for family in ["FT-600K", "FT-MultiT", "Foundation*"]:
        x, mean, sd, _ = aggregate(family, "msd")
        label = "Foundation" if family == "Foundation*" else family
        line_with_band(ax, x, mean, sd, PALETTE[label], label)
    ax.set_xlim(0, 800)
    ax.set_ylim(bottom=0)
    style_axes(ax, "Mean-square displacement", "B", "Lag time (ps)", r"MSD ($\AA^2$)")

    # C: VACF ensemble
    ax = axes[2]
    for family in ["FT-600K", "FT-MultiT", "Foundation*"]:
        x, mean, sd, _ = aggregate(family, "vacf")
        label = "Foundation" if family == "Foundation*" else family
        line_with_band(ax, x, mean, sd, PALETTE[label], label)
    ax.axhline(0, color="#666666", linestyle=(0, (3, 2)), linewidth=1.0, zorder=0)
    ax.set_xlim(0, 800)
    ax.set_ylim(-0.25, 1.05)
    style_axes(ax, "Velocity autocorrelation", "C", "Lag time (ps)", "Normalized VACF")

    # D: diffusivity windows
    ax = axes[3]
    families = ["FT-600K", "FT-MultiT", "Foundation*"]
    positions = np.arange(len(families))
    for i, family in enumerate(families):
        label = "Foundation" if family == "Foundation*" else family
        m1, s1, _ = family_dxy(family, 200, 500)
        m2, s2, _ = family_dxy(family, 500, 900)
        ax.errorbar(i - 0.10, m1, yerr=s1, fmt="o", markersize=6, color=PALETTE[label],
                    markerfacecolor="white", markeredgewidth=1.5, capsize=3, linewidth=1.2,
                    label="200–500 ps" if i == 0 else None)
        ax.errorbar(i + 0.10, m2, yerr=s2, fmt="D", markersize=5, color=PALETTE[label],
                    markerfacecolor=PALETTE[label], markeredgewidth=1.2, capsize=3, linewidth=1.2,
                    label="500–900 ps" if i == 0 else None)
    ax.set_xticks(positions, ["FT-600K", "FT-MultiT", "Foundation*"])
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e-2)
    style_axes(ax, r"$D_{xy}$ window comparison", "D", None, r"$D_{xy}$ (cm$^2$ s$^{-1}$)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.3)

    # E: observed stability horizon, including failure-mode-only Scratch runs.
    ax = axes[4]
    rows = [
        ("FT-600K", "seed123", 1000.0, True), ("FT-600K", "seed234", 1000.0, True), ("FT-600K", "seed345", 1000.0, True),
        ("FT-MultiT", "seed123", 1000.0, True), ("FT-MultiT", "seed234", 1000.0, True), ("FT-MultiT", "seed345", 1000.0, True),
        ("Foundation", "OMAT0-medium", 1000.0, False), ("Foundation", "MP-0b3-medium", 1000.0, False),
        ("Scratch-600K", "seed123", SCRATCH_FAILURES["Scratch-600K"]["seed123"], False),
        ("Scratch-600K", "seed234", SCRATCH_FAILURES["Scratch-600K"]["seed234"], False),
        ("Scratch-600K", "seed345", SCRATCH_FAILURES["Scratch-600K"]["seed345"], False),
        ("Scratch-MultiT", "seed123", SCRATCH_FAILURES["Scratch-MultiT"]["seed123"], False),
        ("Scratch-MultiT", "seed234", SCRATCH_FAILURES["Scratch-MultiT"]["seed234"], False),
        ("Scratch-MultiT", "seed345", SCRATCH_FAILURES["Scratch-MultiT"]["seed345"], False),
    ]
    y = np.arange(len(rows))
    for yi, (family, seed, horizon, accepted) in zip(y, rows):
        ax.plot([0, horizon], [yi, yi], color=PALETTE[family], linewidth=2.2, solid_capstyle="round")
        marker = "o" if accepted else "x"
        ax.plot(horizon, yi, marker=marker, color=PALETTE[family], markersize=6,
                markeredgewidth=1.8, markerfacecolor="white" if accepted else None)
    ax.axvline(1000, color="#222222", linestyle=(0, (4, 2)), linewidth=1.0)
    ax.set_yticks([1, 4, 6.5, 9.5, 12.0], ["FT-600K", "FT-MultiT", "Foundation*", "Scratch-600K", "Scratch-MultiT"])
    ax.set_xlim(0, 1040)
    ax.set_ylim(-1, len(rows))
    style_axes(ax, "Observed trajectory horizon", "E", "Time (ps)", "Model family")
    ax.tick_params(axis="y", length=0)

    # F: window-consistency ratio, a compact diagnostic rather than a ranking.
    ax = axes[5]
    for i, family in enumerate(families):
        label = "Foundation" if family == "Foundation*" else family
        m1, s1, _ = family_dxy(family, 200, 500)
        m2, s2, _ = family_dxy(family, 500, 900)
        ratio = m2 / m1
        rel = ratio * math.sqrt((s1 / m1) ** 2 + (s2 / m2) ** 2) if m1 and m2 else 0.0
        ax.errorbar(i, ratio, yerr=rel, fmt="o", color=PALETTE[label], markersize=7,
                    markerfacecolor="white" if family == "Foundation*" else PALETTE[label],
                    markeredgewidth=1.5, capsize=3, linewidth=1.2)
    ax.axhline(1.0, color="#666666", linestyle=(0, (3, 2)), linewidth=1.0)
    ax.set_xticks(positions, ["FT-600K", "FT-MultiT", "Foundation*"])
    ax.set_yscale("log")
    ax.set_ylim(0.05, 20)
    style_axes(ax, r"$D_{xy}$ window ratio", "F", None, r"500–900 / 200–500 ps")

    handles = []
    labels = []
    for label in ["Foundation", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]:
        handle, = axes[0].plot([], [], color=PALETTE[label], linewidth=2.4, label=label)
        handles.append(handle)
        labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.008), fontsize=10, handlelength=2.4, columnspacing=1.6)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.15, wspace=0.32, hspace=0.38)

    outputs = [RESULTS / "fig_md_1ns_updated_A_to_F.pdf", RESULTS / "fig_md_1ns_updated_A_to_F.png", RESULTS / "fig_md_1ns_updated_A_to_F.svg"]
    fig.savefig(outputs[0], bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[1], dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Machine-readable summary and provenance.
    summary_rows = []
    for family in ["FT-600K", "FT-MultiT", "Foundation*"]:
        for seed, folder in MODEL_FILES[family]:
            d1 = dxy_from_msd(folder, 200, 500)
            d2 = dxy_from_msd(folder, 500, 900)
            summary_rows.append({
                "family": family, "seed_or_variant": seed,
                "trajectory_horizon_ps": 1000.0,
                "status": "CONVERGED_1NS" if family != "Foundation*" else "PROVISIONAL_1NS",
                "dxy_200_500_cm2_s": d1, "dxy_500_900_cm2_s": d2,
            })
    for family, seeds in SCRATCH_FAILURES.items():
        for seed, horizon in seeds.items():
            summary_rows.append({
                "family": family, "seed_or_variant": seed,
                "trajectory_horizon_ps": horizon, "status": "SCIENTIFIC_FAILURE_NO_DXY",
                "dxy_200_500_cm2_s": "", "dxy_500_900_cm2_s": "",
            })
    with (OUT / "md_1ns_family_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader(); writer.writerows(summary_rows)

    source_paths = []
    for family, entries in MODEL_FILES.items():
        for _, folder in entries:
            for name in ("timeseries.csv", "msd.csv", "vacf.csv") if folder.parent == OLD_ROOT else ("corrected_timeseries.csv", "corrected_msd.csv", "corrected_vacf.csv"):
                source_paths.append(folder / name)
    manifest = {
        "figure": "fig_md_1ns_updated_A_to_F",
        "generated_utc_date": "2026-10-25",
        "style_reference": "FIG168_MODEL_STYLE_YELLOW_SCRATCH_R04.json",
        "palette": PALETTE,
        "scientific_boundaries": {
            "FT": "accepted 1 ns three-seed trajectories",
            "Foundation": "provisional comparator; current zero-shot continuation remains pending",
            "Scratch": "six 600K/MultiT smoke runs fail around 200 ps; no 1 ns Dxy reported",
        },
        "sources": [],
        "outputs": [],
    }
    for path in source_paths:
        manifest["sources"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in outputs + [OUT / "md_1ns_family_summary.csv"]:
        manifest["outputs"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    build()
