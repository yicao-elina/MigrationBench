#!/usr/bin/env python3
"""Render three-seed Task 2 figures with Scientific Plot Atlas primitives."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_atlas.charts import clustered_heatmap, grouped_bar, timecourse
from plot_atlas.style import apply_publication_style, panel_label, save_figure


ROOT = Path(__file__).resolve().parents[1]
BR = ROOT / "results/broad_roster"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
COLORS = {
    "Foundation": "#6B7280",
    "Seed 123": "#4C78A8",
    "Seed 456": "#F28E2B",
    "Seed 789": "#59A14F",
}
SEEDS = ["broad_roster_ft_seed123", "broad_roster_ft_seed456", "broad_roster_ft_seed789"]
SEED_LABELS = {"broad_roster_ft_seed123": "Seed 123", "broad_roster_ft_seed456": "Seed 456", "broad_roster_ft_seed789": "Seed 789"}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_all(fig, stem: str, title: str):
    png = OUT / f"{stem}.png"
    save_figure(fig, png, dpi=300)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None, "Creator": "Scientific Plot Atlas"})
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata={"Creator": "Scientific Plot Atlas", "Title": title})
    plt.close(fig)


def main() -> None:
    apply_publication_style(font_size=9)
    summary = read_csv(BR / "table_3_diagnostic_profile_summary.csv")
    profiles = read_csv(ROOT / "results/validation/diagnostic_profiles.csv")
    qe = read_csv(BR / "qe_neb_path_profiles.csv")
    metrics = read_csv(BR / "seed_validation_metrics.csv")

    base = [r for r in summary if r["variant"] == "base"]
    display_ids = ["foundation", *SEEDS]
    display_labels = ["Foundation", "Seed 123", "Seed 456", "Seed 789"]

    # Figure 1: force scale and image/path seed spread.
    elements = ["Cu", "Li", "Na"]
    force_matrix = []
    for element in elements:
        force_matrix.append([
            np.mean([float(r["max_force_eV_A"]) for r in base if r["element"] == element and r["model_id"] == model_id])
            for model_id in display_ids
        ])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), gridspec_kw={"width_ratios": [1.0, 1.65]})
    panel_label(axes[0], "a")
    grouped_bar(axes[0], elements, np.asarray(force_matrix), display_labels, [COLORS[x] for x in display_labels], ylabel="Mean max force (eV Å$^{-1}$)")
    axes[0].set_title("Finite force scale across elements", pad=8)
    axes[0].tick_params(axis="x", rotation=0)

    path_ids = [r["path_id"] for r in base if r["model_id"] == "foundation"]
    path_ids = list(dict.fromkeys(path_ids))
    deviation = []
    for seed in SEEDS:
        row = []
        for path_id in path_ids:
            vals = [float(r["max_relative_energy_eV"]) for r in base if r["path_id"] == path_id and r["model_id"] in SEEDS]
            current = float(next(r["max_relative_energy_eV"] for r in base if r["path_id"] == path_id and r["model_id"] == seed))
            row.append(current - float(np.mean(vals)))
        deviation.append(row)
    short_paths = [p.replace("_neb-gap-diffuse", "-GD").replace("_neb-gap2QL", "-G2").replace("_neb-QLs", "-QL").replace("_neb", "") for p in path_ids]
    panel_label(axes[1], "b")
    clustered_heatmap(axes[1], np.asarray(deviation), ["Seed 123", "Seed 456", "Seed 789"], short_paths, low="#3B6EA8", mid="#FFFFFF", high="#C44E52", cluster=False, colorbar=True)
    axes[1].yaxis.tick_left()
    axes[1].tick_params(axis="y", pad=3)
    axes[1].set_title("Seed deviation from path-wise mean barrier", pad=8)
    axes[1].tick_params(axis="x", labelrotation=38, pad=1)
    axes[1].set_xlabel("Archived NEB path")
    axes[1].set_ylabel("")
    fig.suptitle("Task 2 — epoch-999 three-seed diagnostic stability", y=1.02, fontsize=11)
    fig.text(0.5, -0.08, "All curves are finite non-equilibrium snapshot diagnostics; the QE/NEB SCF gate remains failed.", ha="center", fontsize=8, color="#6B1D1D")
    fig.tight_layout(w_pad=2.0)
    save_all(fig, "task2_three_seed_stability", "Task 2 three-seed diagnostic stability")

    # Figure 2: QE/NEB provisional reference plus four model profiles.
    selected = ["Cu_neb-gap-diffuse", "Li_neb-gap-diffuse", "Na_neb-gap-diffuse"]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7), sharey=False)
    profile_labels = ["Seed 123", "Seed 456", "Seed 789"]
    for i, (ax, path_id) in enumerate(zip(axes, selected)):
        panel_label(ax, chr(ord("a") + i))
        qe_rows = sorted([r for r in qe if r["path_id"] == path_id], key=lambda r: int(r["image"]))
        qe_vals = np.asarray([float(r["qe_neb_energy_eV"]) for r in qe_rows])
        qe_vals -= qe_vals.min()
        ax.plot(range(1, 11), qe_vals, color="#222222", linestyle=(0, (4, 2)), marker="s", markersize=3.0, linewidth=1.1, label="QE/NEB last complete table", zorder=5)
        curves = []
        for seed in SEEDS:
            rows = sorted([r for r in profiles if r["path_id"] == path_id and r["model_id"] == seed and r["variant"] == "base"], key=lambda r: int(r["image"]))
            curves.append([float(r["relative_energy_eV"]) for r in rows])
        timecourse(ax, range(1, 11), curves, profile_labels, [COLORS[x] for x in profile_labels], xlabel="Image", ylabel="Relative energy (eV)" if i == 0 else None, markers=True, legend=False, line_width=1.5, marker_size=3.0)
        ax.set_title(path_id.replace("_", " "), fontsize=9, pad=8)
        ax.set_xticks([1, 5, 10])
    handles = [plt.Line2D([], [], color="#222222", linestyle=(0, (4, 2)), marker="s", linewidth=1.1, label="QE/NEB last complete table")]
    handles += [plt.Line2D([], [], color=COLORS[x], marker="o", linewidth=1.5, label=x) for x in profile_labels]
    fig.legend(handles, [h.get_label() for h in handles], frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=4)
    fig.suptitle("QE/NEB provisional reference and epoch-999 seed profiles", y=1.03, fontsize=11)
    fig.text(0.5, -0.19, "QE/NEB curves use last-complete neb.out energy tables; SCF warnings remain, so they are not strict DFT ground truth.", ha="center", fontsize=8, color="#6B1D1D")
    fig.tight_layout(w_pad=1.8)
    save_all(fig, "task2_three_seed_profiles", "Task 2 QE/NEB and three-seed profiles")

    # Figure 3: final validation errors, explicitly separate from path diagnostics.
    seeds = ["123", "456", "789"]
    energy = np.asarray([[float(r["validation_rmse_energy_meV_atom"])] for r in metrics])
    force = np.asarray([[float(r["validation_rmse_force_meV_A"])] for r in metrics])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    panel_label(axes[0], "a")
    grouped_bar(axes[0], seeds, energy, ["Validation RMSE energy"], ["#4C78A8"], ylabel="meV atom$^{-1}$")
    axes[0].set_title("Validation energy RMSE", pad=8)
    panel_label(axes[1], "b")
    grouped_bar(axes[1], seeds, force, ["Validation RMSE force"], ["#E45756"], ylabel="meV Å$^{-1}$")
    axes[1].set_title("Validation force RMSE", pad=8)
    fig.suptitle("Final epoch-999 training validation", y=1.04, fontsize=11)
    fig.text(0.5, -0.08, "Metrics are held-out validation diagnostics from the three independent training runs.", ha="center", fontsize=8)
    fig.tight_layout(w_pad=2.0)
    save_all(fig, "task2_three_seed_validation", "Three-seed epoch-999 validation")

    (ROOT / "results/figure_data_dictionary.json").write_text(json.dumps({
        "task2_three_seed_stability": {"source": "results/broad_roster/table_3_diagnostic_profile_summary.csv", "panels": {"a": "element-wise mean max force across archived paths", "b": "seed deviation from path-wise mean max relative energy"}},
        "task2_three_seed_profiles": {"source": ["results/validation/diagnostic_profiles.csv", "results/broad_roster/qe_neb_path_profiles.csv"], "panels": {"a-c": "QE/NEB last-complete table plus three epoch-999 seed base profiles"}},
        "task2_three_seed_validation": {"source": "results/broad_roster/seed_validation_metrics.csv", "panels": {"a": "energy RMSE", "b": "force RMSE"}},
        "scientific_boundary": "All path figures are diagnostic/provisional because archived QE/NEB paths retain SCF convergence warnings."
    }, indent=2) + "\n", encoding="utf-8")
    print("rendered three-seed figures in", OUT)


if __name__ == "__main__":
    main()
