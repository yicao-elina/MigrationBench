#!/usr/bin/env python3
"""Plot Atlas-style standalone time-series and model x snapshot stability panels."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
try:
    from plot_atlas import apply_publication_style, bold_tick_labels, get_palette, save_figure, timecourse
    ATLAS_STATUS = "plot_atlas_api"
except ImportError:
    ATLAS_STATUS = "self_contained_style_compatible_fallback"
    def apply_publication_style(size=12):
        plt.rcParams.update({"font.family": "Arial", "font.size": size, "axes.linewidth": 1.25,
                             "xtick.labelsize": size - 2, "ytick.labelsize": size - 2,
                             "svg.fonttype": "none"})
    def save_figure(fig, output):
        fig.savefig(output, dpi=300, bbox_inches="tight")
    def bold_tick_labels(ax):
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()): label.set_fontweight("bold")
    def get_palette(name):
        return {"categorical": ["#9B9B9B", "#98449F", "#337FB0", "#48A947"]}
    def timecourse(ax, x, ys, labels, colors, **kwargs):
        for i, y in enumerate(ys):
            ax.plot(x, y, color=colors[i], label=labels[i], **{k: v for k, v in kwargs.items() if k in {"linestyle", "linewidth"}})


MODELS = ["foundation-omat", "from-scratch", "naive-fine-tuning", "multi_T-fine-tuning"]
LABELS = {"foundation-omat": "Foundation", "from-scratch": "Scratch", "naive-fine-tuning": "FT-600K", "multi_T-fine-tuning": "FT-MultiT"}
COLORS = {"foundation-omat": "#0072B2", "from-scratch": "#D55E00", "naive-fine-tuning": "#009E73", "multi_T-fine-tuning": "#CC79A7"}


def read_csv(p):
    with open(p) as f: return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); args = ap.parse_args()
    root = Path(args.root); apply_publication_style(8.2)
    palette = get_palette("nature_steered_dynamics")
    # Registered MD palette: foundation/from-scratch/FT-MultiT/FT-600K.
    model_colors = dict(zip(MODELS, palette["categorical"]))
    # Independent time-series panel: four diagnostics, one line per model and snapshot.
    metrics = {}
    for p in sorted(root.glob("snapshot_*/model_*/metrics.csv")):
        metrics[(p.parent.parent.name, p.parent.name.removeprefix("model_"))] = read_csv(p)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6), sharex=True)
    specs = [("temperature_K", "K", "Kinetic temperature"), ("max_force_eV_A", "eV A$^{-1}$", "Maximum force"),
             ("min_distance_A", "Å", "Minimum pair distance"), ("max_unwrapped_displacement_A", "Å", "Max unwrapped displacement")]
    for ax, (field, unit, title) in zip(axes.ravel(), specs):
        for (snap, model), rows in metrics.items():
            x = np.asarray([float(r["time_ps"]) for r in rows]); y = np.asarray([float(r[field]) for r in rows])
            ls = "-" if snap.endswith("461800") else "--"
            timecourse(ax, x, [y], [LABELS[model] if snap.endswith("461800") else "_nolegend_"], [model_colors[model]], markers=False,
                       line_width=1.4, linestyles=[ls])
        ax.set_title(title, fontsize=9); ax.set_ylabel(unit); ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False); bold_tick_labels(ax)
    axes[1, 0].set_xlabel("Handoff time (ps)"); axes[1, 1].set_xlabel("Handoff time (ps)")
    # Keep the legend outside the data region so curves cannot run through it.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.985))
    fig.text(0.99, 0.02, f"solid: step 461800; dashed: step 462000 | {ATLAS_STATUS}", ha="right", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91)); save_figure(fig, root / "handoff_timeseries.png"); save_figure(fig, root / "handoff_timeseries.svg"); plt.close(fig)
    # Stability summary heatmap; stable=1, unstable=0, missing=-1.
    summary = json.loads((root / "stability_summary.json").read_text())
    snaps = sorted({s["snapshot"] for s in summary}); mat = np.full((len(MODELS), len(snaps)), np.nan)
    for s in summary: mat[MODELS.index(s["model"]), snaps.index(s["snapshot"])] = 1 if s["classification"] == "stable_short_handoff" else 0
    fig, ax = plt.subplots(figsize=(5.2, 3.1)); cmap = ListedColormap(["#D55E00", "#E69F00", "#009E73"])
    # 0 runaway, 1 bounded but structurally shifted, 2 stable.
    mat = np.full((len(MODELS), len(snaps)), np.nan)
    code = {"runaway_or_nonfinite": 0, "bounded_with_structural_shift": 1, "stable_short_handoff": 2}
    for s in summary: mat[MODELS.index(s["model"]), snaps.index(s["snapshot"])] = code[s["classification"]]
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(snaps)), [x.replace("snapshot_", "step ") for x in snaps]); ax.set_yticks(range(len(MODELS)), [LABELS[x] for x in MODELS])
    ax.set_title("Short-handoff stability gate", fontsize=11); ax.tick_params(labelsize=9)
    for i in range(len(MODELS)):
        for j in range(len(snaps)):
            val = mat[i, j]; txt = {0: "runaway", 1: "shift", 2: "stable"}.get(int(val), "missing")
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout(); save_figure(fig, root / "handoff_stability_heatmap.png"); save_figure(fig, root / "handoff_stability_heatmap.svg"); plt.close(fig)


if __name__ == "__main__": main()
