#!/usr/bin/env python3
"""Render the local-cell MACE diagnostic with the established Plot Atlas style."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_atlas import apply_publication_style, bold_tick_labels, get_palette, save_figure
import plot_atlas


MODELS = ["foundation-omat", "from-scratch", "naive-fine-tuning", "multi_T-fine-tuning"]
LABELS = {
    "foundation-omat": "Foundation-OMAT",
    "from-scratch": "From-scratch",
    "naive-fine-tuning": "Naive-FT",
    "multi_T-fine-tuning": "Multi-T",
}
CASES = [
    "normal_461800_initial",
    "bounded_multiT_462000_step464000",
    "runaway_foundation_461800_step462120",
    "runaway_scratch_461800_step462060",
]
CASE_LABELS = {
    "normal_461800_initial": "Normal\n461800",
    "bounded_multiT_462000_step464000": "Bounded / shift\nMulti-T 464000",
    "runaway_foundation_461800_step462120": "Runaway onset\nFoundation 462120",
    "runaway_scratch_461800_step462060": "Runaway onset\nScratch 462060",
}


def git_head(path: Path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path(__file__).parent / "mace_results" / "mace_within_model_comparison.csv")
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "mace_results" / "figures")
    ap.add_argument("--atlas", type=Path, default=Path("/Users/alina/Documents/Codex/plot_library"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with args.input.open() as fh:
        rows = list(csv.DictReader(fh))
    by_key = {(r["case"], r["model"]): r for r in rows}
    palette = get_palette("nature_steered_dynamics")["categorical"]
    colors = {"foundation-omat": palette[2], "from-scratch": "#D55E00", "naive-fine-tuning": palette[3], "multi_T-fine-tuning": palette[1]}

    apply_publication_style(font_size=8.2)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.55), constrained_layout=True)
    x = np.arange(len(CASES))
    width = 0.19
    for i, model in enumerate(MODELS):
        forces = [float(by_key[(case, model)]["max_force_eV_A"]) for case in CASES]
        delta_e = [float(by_key[(case, model)]["delta_energy_eV_within_model"]) for case in CASES]
        axes[0].bar(x + (i - 1.5) * width, forces, width, label=LABELS[model], color=colors[model])
        axes[1].bar(x + (i - 1.5) * width, delta_e, width, color=colors[model])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Local-cell maximum force (eV Å$^{-1}$)")
    axes[1].set_ylabel("Within-model ΔE from normal (eV)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xticks(x, [CASE_LABELS[c] for c in CASES])
        ax.tick_params(axis="x", labelsize=7)
        bold_tick_labels(ax)
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_title("Force response")
    axes[1].set_title("Energy change within each model")
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("Local-cell MACE reconstruction diagnostic", fontsize=9.5)
    fig.text(0.99, -0.03, "RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED; not DFT validation", ha="right", fontsize=7)
    for ext in ("png", "svg", "pdf"):
        save_figure(fig, args.output / f"mace_reconstruction_diagnostic.{ext}")
    plt.close(fig)

    env = {
        "renderer": "Plot Atlas publication style + save_figure",
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "plot_atlas_module": str(Path(plot_atlas.__file__).resolve()),
        "plot_atlas_git_head": git_head(args.atlas),
        "atlas_palette": "nature_steered_dynamics",
        "command": f"MPLBACKEND=Agg PYTHONPATH=/Users/alina/Documents/Codex/plot_library/src {sys.executable} {Path(__file__)}",
    }
    (args.output / "plot_env.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    (args.output / "figure_manifest.json").write_text(json.dumps({"input": str(args.input), "outputs": sorted(str(p) for p in args.output.glob("mace_reconstruction_diagnostic.*")), "scope": "RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
