#!/usr/bin/env python3
"""Render Phase-2 handoff figures with the reusable Plot Atlas grammar."""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from plot_atlas import apply_publication_style, bold_tick_labels, get_palette, save_figure, timecourse
import plot_atlas


MODELS = ["foundation-omat", "from-scratch", "naive-fine-tuning", "multi_T-fine-tuning"]
LABELS = {
    "foundation-omat": "Foundation-OMAT",
    "from-scratch": "From-scratch",
    "naive-fine-tuning": "Naive-FT",
    "multi_T-fine-tuning": "Multi-T",
}
# Reuse the Atlas steered-dynamics palette family while keeping the semantic
# model order fixed across all MigrationBench figures.
ATLAS_COLORS = get_palette("nature_steered_dynamics")["categorical"]
COLORS = {
    "foundation-omat": ATLAS_COLORS[2],
    "from-scratch": "#D55E00",
    "naive-fine-tuning": ATLAS_COLORS[3],
    "multi_T-fine-tuning": ATLAS_COLORS[1],
}


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--atlas", default="/Users/alina/Documents/Codex/plot_library")
    args = ap.parse_args()
    audit, output, atlas = Path(args.audit), Path(args.output), Path(args.atlas)
    output.mkdir(parents=True, exist_ok=True)
    apply_publication_style(font_size=8.2)
    rows = read_rows(audit / "frame_aligned_metrics.csv")
    by_run = {}
    for row in rows:
        by_run.setdefault((row["snapshot"], row["model"]), []).append(row)
    summary = json.loads((audit / "first_divergence_by_run.json").read_text())
    first = {(x["snapshot"], x["model"]): x.get("first_divergence") for x in summary}

    specs = [
        ("temperature_K", "Temperature (K)", "log", None),
        ("max_force_eV_A", "Maximum force (eV Å$^{-1}$)", "log", None),
        ("min_distance_A", "Minimum pair distance (Å)", "linear", None),
        ("max_unwrapped_displacement_A", "Max. unwrapped displacement (Å)", "symlog", 1),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.7), sharex=False)
    legend_done = False
    for ax, (field, ylabel, scale, linthresh) in zip(axes.ravel(), specs):
        for snapshot in ("461800", "462000"):
            for model in MODELS:
                run = by_run[(snapshot, model)]
                x = np.asarray([float(r["time_ps"]) - float(snapshot) / 1000.0 for r in run])
                y = np.asarray([float(r[field]) for r in run])
                label = LABELS[model] if snapshot == "461800" else "_nolegend_"
                timecourse(ax, x, [y], [label], [COLORS[model]], markers=False,
                           line_width=1.15, linestyles=["-" if snapshot == "461800" else "--"])
                d = first[(snapshot, model)]
                if d is not None and field in {"temperature_K", "max_force_eV_A", "max_unwrapped_displacement_A"}:
                    # One unobtrusive marker at the first gate crossing.
                    ax.axvline(d["time_ps"] - float(snapshot) / 1000.0, color=COLORS[model],
                               lw=0.35, alpha=0.18, zorder=0)
        if scale == "log":
            ax.set_yscale("log")
        elif scale == "symlog":
            ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Handoff time from input (ps)")
        ax.set_title({"temperature_K": "Thermal response", "max_force_eV_A": "Force growth",
                      "min_distance_A": "Contact geometry", "max_unwrapped_displacement_A": "Structural displacement"}[field])
        ax.set_xlim(left=0)
        bold_tick_labels(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.text(0.99, 0.01, "solid: 461800; dashed: 462000", ha="right", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("png", "svg", "pdf"):
        save_figure(fig, output / f"phase2_handoff_timeseries.{ext}")
    plt.close(fig)

    snap_order = ["461800", "462000"]
    code = {"runaway_or_nonfinite": 0, "bounded_with_structural_shift": 1, "stable_short_handoff": 2}
    mat = np.empty((len(MODELS), len(snap_order)), dtype=float)
    labels = [["runaway", "shift", "stable"][code[x["classification"]]] for x in summary]
    # Use explicit indexing rather than depending on JSON order.
    for i, model in enumerate(MODELS):
        for j, snap in enumerate(snap_order):
            item = next(x for x in summary if x["model"] == model and x["snapshot"] == snap)
            mat[i, j] = code[item["classification"]]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    cmap = ListedColormap(["#D55E00", "#E69F00", "#009E73"])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(2), ["Input step 461800", "Input step 462000"])
    ax.set_yticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
    ax.set_title("2 ps model-stability gate")
    for i, model in enumerate(MODELS):
        for j, snap in enumerate(snap_order):
            item = next(x for x in summary if x["model"] == model and x["snapshot"] == snap)
            text = {0: "runaway", 1: "bounded / shift", 2: "stable"}[code[item["classification"]]]
            d = item.get("first_divergence")
            if d is not None:
                text += f"\nstep {d['step']}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=7.2, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        save_figure(fig, output / f"phase2_stability_gate.{ext}")
    plt.close(fig)

    env = {
        "renderer": "Plot Atlas timecourse primitive + Atlas publication style",
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "plot_atlas_module": str(Path(plot_atlas.__file__).resolve()),
        "plot_atlas_git_head": git_head(atlas),
        "atlas_palette": "nature_steered_dynamics",
        "model_color_mapping": COLORS,
        "command": f"MPLBACKEND=Agg PYTHONPATH=/Users/alina/Documents/Codex/plot_library/src {sys.executable} analysis/phase2_handoff_md_20261002/audit_20261003/render_phase2_atlas.py --audit analysis/phase2_handoff_md_20261002/audit_20261003/results --output analysis/phase2_handoff_md_20261002/audit_20261003/figures",
    }
    (output / "plot_env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    manifest = {
        "data": str(audit / "frame_aligned_metrics.csv"),
        "summary": str(audit / "first_divergence_by_run.json"),
        "outputs": sorted(str(p) for p in output.glob("phase2_*")),
        "styles": ["plot_atlas.apply_publication_style", "plot_atlas.timecourse", "plot_atlas.save_figure"],
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
