#!/usr/bin/env python3
"""Render the A2 grouped-retraining and overlap audit figure.

The left panel uses the Plot Atlas mean-plus-replicates grammar: individual
training-seed barrier errors are shown as open points, while the bars and
whiskers show the family mean and sample standard deviation.  The MPA0 row is
kept as a single-seed comparator and is deliberately drawn as a point without
an uncertainty bar.

The right panel reports each minimum distance divided by the declared
near-duplicate gate. A value above one is outside the near-duplicate region;
the RMSD values are restricted to species-composition-compatible local
environments, as recorded in the source JSON audits. Explanatory notes belong
in the caption, not as small text inside the publication figure.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_atlas import apply_publication_style, bar_with_points, get_palette, grouped_bar, save_figure


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def family_values(rows: list[dict[str, str]], family: str) -> list[float]:
    return [
        float(row["pathway_1_4_abs_error_eV"])
        for row in rows
        if row["family"] == family
    ]


def main() -> None:
    # Typography is chosen for the final manuscript placement. The figure is
    # embedded at roughly 0.9x in the response/SI, so the source sizes are
    # intentionally larger than a default Plot Atlas panel.
    apply_publication_style(13.5)
    mpl.rcParams.update({
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.2,
    })
    seed_rows = read_rows(DATA / "grouped_seed_comparison.csv")
    full_audit = read_json(DATA / "overlap_full_leakage_audit.json")
    scratch5_audit = read_json(DATA / "overlap_scratch5_leakage_audit.json")

    families = ["OMAT-FT", "Scratch", "Scratch-5%"]
    family_labels = ["OMAT-FT", "Scratch", "Scratch-5%", "MPA0\n(single seed)"]
    jhu = get_palette("jhu_mace_models")
    roles = jhu["roles"]
    family_colors = [roles["ft_600k"], roles["from_scratch"], roles["ft_multit"]]
    points = [family_values(seed_rows, family) for family in families]
    means = np.array([np.mean(values) for values in points])
    sds = np.array([np.std(values, ddof=1) for values in points])

    audits = [
        ("Full grouped", full_audit["summaries"]["train"]),
        ("Scratch-5%", scratch5_audit["summaries"]["train"]),
    ]
    threshold_soap = float(full_audit["thresholds"]["near_soap_cosine"])
    threshold_rmsd = float(full_audit["thresholds"]["near_rmsd_A"])
    ratio_rows = []
    ratio_matrix = []
    for label, summary in audits:
        soap_min = float(summary["soap_cosine_distance"]["min"])
        rmsd_min = float(summary["cr_local_species_rmsd_A"]["min"])
        ratio_rows.append(
            {
                "split": label,
                "min_soap_cosine": f"{soap_min:.10g}",
                "soap_threshold": f"{threshold_soap:.10g}",
                "soap_ratio": f"{soap_min / threshold_soap:.6g}",
                "min_comparable_rmsd_A": f"{rmsd_min:.10g}",
                "rmsd_threshold_A": f"{threshold_rmsd:.10g}",
                "rmsd_ratio": f"{rmsd_min / threshold_rmsd:.6g}",
                "rmsd_comparable_frames": str(summary["local_rmsd_comparable_frames"]),
            }
        )
        ratio_matrix.append([soap_min / threshold_soap, rmsd_min / threshold_rmsd])

    with (DATA / "overlap_ratio_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ratio_rows[0]))
        writer.writeheader()
        writer.writerows(ratio_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.75), gridspec_kw={"width_ratios": [1.2, 1.0]})

    # Panel a: seed-level barrier errors and grouped uncertainty.
    bar_with_points(
        axes[0],
        family_labels[:3],
        means,
        sds,
        family_colors,
        points=points,
        ylabel="Pathway 1--4 absolute barrier error (eV)",
    )
    for patch, color in zip(axes[0].patches[:3], family_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#3A3A3A")
        patch.set_linewidth(0.75)
    axes[0].axhline(
        0.160742,
        color="#555555",
        linestyle=(0, (4, 2)),
        linewidth=1.1,
        label="historical frame-level 0.16 eV",
        zorder=1,
    )
    axes[0].scatter(
        [3],
        [0.237235],
        marker="D",
        s=30,
        facecolor=roles["foundation_omat"],
        edgecolor="#222222",
        linewidth=1.0,
        zorder=4,
    )
    axes[0].set_xticks([0, 1, 2, 3], family_labels)
    axes[0].set_xlim(-0.55, 3.55)
    axes[0].set_ylim(0, 1.8)
    axes[0].set_xlabel("Model", fontweight="bold", labelpad=7)
    axes[0].set_ylabel("Pathway 1--4 absolute barrier error (eV)", fontweight="bold", labelpad=8)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="#555555", linestyle=(0, (4, 2)), linewidth=1.4,
                   label="Historical frame-level 0.16 eV"),
            Line2D([0], [0], marker="D", color=roles["foundation_omat"],
                   markerfacecolor=roles["foundation_omat"], markeredgecolor="#222222",
                   markersize=7, linestyle="None", label="MPA0 single seed"),
        ],
        frameon=False,
        loc="upper left",
        handlelength=2.2,
        handletextpad=0.6,
        labelspacing=0.5,
        borderpad=0.2,
    )
    axes[0].text(
        -0.12,
        1.04,
        "a",
        transform=axes[0].transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
    )

    # Panel b: distance ratios relative to the declared near-duplicate gates.
    grouped_bar(
        axes[1],
        [row[0] for row in audits],
        np.asarray(ratio_matrix),
        ["SOAP cosine", "Cr-local RMSD"],
        [jhu["categorical"][1], jhu["categorical"][2]],
        ylabel="Distance / declared near-duplicate gate",
        legend=False,
    )
    axes[1].axhline(1.0, color="#555555", linestyle=(0, (4, 2)), linewidth=1.3, zorder=1)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.8, 100)
    axes[1].set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    axes[1].set_xlabel("Training split", fontweight="bold", labelpad=7)
    axes[1].set_ylabel("Distance / declared near-duplicate gate", fontweight="bold", labelpad=8)
    axes[1].legend(
        handles=[
            Patch(facecolor=jhu["categorical"][1], edgecolor="#3A3A3A", linewidth=0.7,
                  label="SOAP cosine"),
            Patch(facecolor=jhu["categorical"][2], edgecolor="#3A3A3A", linewidth=0.7,
                  label="Cr-local RMSD"),
            Line2D([0], [0], color="#555555", linestyle=(0, (4, 2)), linewidth=1.3,
                   label="Declared gate (1)"),
        ],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        handlelength=2.2,
        handletextpad=0.6,
        labelspacing=0.5,
        borderpad=0.2,
        columnspacing=1.0,
    )
    axes[1].text(
        -0.12,
        1.04,
        "b",
        transform=axes[1].transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
    )

    for ax in axes:
        ax.tick_params(axis="both", which="major", width=1.1, length=5)
        ax.tick_params(axis="both", which="minor", width=0.8, length=3)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_linewidth(1.1)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontweight("bold")
    for label in axes[0].get_xticklabels():
        label.set_linespacing(1.05)
        label.set_rotation(25)
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.31, top=0.86, wspace=0.42)
    for suffix in ("png", "pdf", "svg"):
        output = FIGURES / f"a2_grouped_retraining_audit.{suffix}"
        if suffix == "png":
            save_figure(fig, output, dpi=300)
        else:
            fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
