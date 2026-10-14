#!/usr/bin/env python3
"""Render the provenance-linked from-scratch failure-attribution composite.

The figure is intentionally regenerated from compact, committed tables rather
than copied from an exploratory notebook.  Panel (a) is a conceptual map of
the sampling and ranking operator; panels (b--d) are quantitative summaries
of the paired 639 x 3 static MLIP evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch


GROUPS = [
    "stable_early_control",
    "stable_pre_onset",
    "precursor_near_onset",
    "threshold_or_terminal",
    "terminal_stable_target",
]
GROUP_LABELS = {
    "stable_early_control": "early\nstable",
    "stable_pre_onset": "pre-onset\nstable",
    "precursor_near_onset": "near-onset\nprecursor",
    "threshold_or_terminal": "threshold /\nterminal",
    "terminal_stable_target": "1 ns\nstable",
}
GROUP_COLORS = {
    "stable_early_control": "#4c78a8",
    "stable_pre_onset": "#72b7b2",
    "precursor_near_onset": "#f2a541",
    "threshold_or_terminal": "#d1495b",
    "terminal_stable_target": "#7a5195",
}
MODEL_ORDER = [
    "ablation_lmax_2_seed123",
    "ablation_lmax_2_seed234",
    "baseline_seed345",
]
MODEL_LABELS = {
    "ablation_lmax_2_seed123": r"$l_{\max}=2$, seed 123",
    "ablation_lmax_2_seed234": r"$l_{\max}=2$, seed 234",
    "baseline_seed345": "baseline, seed 345",
}
MODEL_COLORS = {
    "ablation_lmax_2_seed123": "#6f6f6f",
    "ablation_lmax_2_seed234": "#756bb1",
    "baseline_seed345": "#2c7fb8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_axes_style(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", width=0.7, length=3, labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#222222")
    ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.09,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def draw_periodic_cell(ax: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    """Draw a deliberately schematic periodic local environment, not data."""
    ax.add_patch(
        patches.Rectangle(
            (x, y), width, height, fill=False, linewidth=0.8, linestyle="--", edgecolor="#555555"
        )
    )
    atoms = [
        (0.16, 0.27, "#9e9e9e", 0.028),
        (0.33, 0.24, "#9e9e9e", 0.025),
        (0.53, 0.28, "#9e9e9e", 0.028),
        (0.70, 0.22, "#9e9e9e", 0.024),
        (0.26, 0.57, "#bdbdbd", 0.024),
        (0.47, 0.61, "#bdbdbd", 0.026),
        (0.70, 0.58, "#bdbdbd", 0.024),
        (0.49, 0.43, "#d1495b", 0.038),
    ]
    for dx, dy, color, radius in atoms:
        ax.add_patch(plt.Circle((x + dx * width, y + dy * height), radius * width, color=color, ec="white", lw=0.35))
    ax.add_patch(
        patches.Circle(
            (x + 0.49 * width, y + 0.43 * height),
            0.18 * width,
            fill=False,
            ec="#d1495b",
            lw=0.8,
            linestyle=":",
        )
    )


def render_selection_schematic(ax: plt.Axes, counts: dict[str, int]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "a")
    ax.text(0.02, 0.96, "Sampling and ranking operator", fontsize=9, fontweight="bold", va="top")

    draw_periodic_cell(ax, 0.03, 0.62, 0.18, 0.22)
    ax.text(0.12, 0.57, "same 2050-atom\nPBC snapshot", ha="center", va="top", fontsize=6.5)
    ax.annotate("", xy=(0.31, 0.73), xytext=(0.22, 0.73), arrowprops=dict(arrowstyle="->", lw=0.9))

    ax.text(0.31, 0.87, "639 saved frames", fontsize=7.5, fontweight="bold", ha="left")
    ax.text(0.31, 0.825, "classified by trajectory stage", fontsize=6.5, ha="left", color="#444444")
    x0, y0, w, h, gap = 0.30, 0.69, 0.115, 0.075, 0.008
    short_labels = ["early stable", "pre-onset", "precursor", "threshold", "1 ns stable"]
    for i, group in enumerate(GROUPS):
        x = x0 + i * (w + gap)
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y0), w, h,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=GROUP_COLORS[group], edgecolor="white", linewidth=0.5, alpha=0.95,
            )
        )
        ax.text(x + w / 2, y0 + h / 2 + 0.006, short_labels[i], ha="center", va="center", fontsize=4.6, color="white", fontweight="bold")
        ax.text(x + w / 2, y0 - 0.022, f"n={counts[group]}", ha="center", va="top", fontsize=5.7)

    ax.annotate("", xy=(0.68, 0.55), xytext=(0.61, 0.65), arrowprops=dict(arrowstyle="->", lw=0.9, connectionstyle="arc3,rad=-0.15"))
    ax.text(0.715, 0.625, "paired static\nevaluation", ha="center", va="center", fontsize=6.1, fontweight="bold")
    model_short = ["lmax2 / 123", "lmax2 / 234", "baseline / 345"]
    for i, model in enumerate(MODEL_ORDER):
        yy = 0.53 - i * 0.075
        ax.add_patch(patches.FancyBboxPatch((0.62, yy), 0.19, 0.048, boxstyle="round,pad=0.006,rounding_size=0.008", facecolor=MODEL_COLORS[model], edgecolor="white", lw=0.5))
        ax.text(0.715, yy + 0.024, model_short[i], ha="center", va="center", fontsize=5.2, color="white", fontweight="bold")

    ax.annotate("", xy=(0.87, 0.39), xytext=(0.79, 0.48), arrowprops=dict(arrowstyle="->", lw=0.9, connectionstyle="arc3,rad=-0.1"))
    ax.add_patch(patches.FancyBboxPatch((0.81, 0.26), 0.17, 0.13, boxstyle="round,pad=0.01,rounding_size=0.012", facecolor="#f3f3f3", edgecolor="#555555", lw=0.7))
    ax.text(0.895, 0.355, "rank by paired\nforce spread", ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax.text(0.895, 0.285, r"$\sigma_F$ and $\Delta F_{max}$", ha="center", va="center", fontsize=6.0, color="#444444")
    ax.annotate("", xy=(0.67, 0.20), xytext=(0.81, 0.26), arrowprops=dict(arrowstyle="->", lw=0.9, connectionstyle="arc3,rad=0.05"))
    ax.add_patch(patches.FancyBboxPatch((0.42, 0.08), 0.27, 0.13, boxstyle="round,pad=0.01,rounding_size=0.012", facecolor="#fff4df", edgecolor="#d18900", lw=0.8))
    ax.text(0.555, 0.16, "20 DFT-priority rows", ha="center", va="center", fontsize=7, fontweight="bold", color="#7a4d00")
    ax.text(0.555, 0.105, "prepared-only; no DFT parity", ha="center", va="center", fontsize=5.9, color="#7a4d00")
    ax.text(0.04, 0.04, "Panels b–d map the same candidate groups to the three-model force diagnostics.", fontsize=6.0, color="#444444")


def render_force_boxplot(ax: plt.Axes, results: pd.DataFrame) -> None:
    panel_label(ax, "b")
    ax.set_title("Static maximum force across stages", fontsize=8.5, fontweight="bold", pad=6)
    positions = []
    data = []
    colors = []
    ticks = []
    labels = []
    for gi, group in enumerate(GROUPS):
        base = gi * 4
        for mi, model in enumerate(MODEL_ORDER):
            values = results.loc[
                (results["candidate_group"] == group) & (results["model_id"] == model),
                "max_force_eV_A",
            ].dropna().to_numpy()
            if len(values) == 0:
                continue
            data.append(values)
            positions.append(base + mi + 1)
            colors.append(MODEL_COLORS[model])
        ticks.append(base + 2)
        labels.append(GROUP_LABELS[group])
    box = ax.boxplot(data, positions=positions, widths=0.64, patch_artist=True, showfliers=False, zorder=2)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(0.55)
    for key in ("whiskers", "caps", "medians"):
        for item in box[key]:
            item.set_color("#222222")
            item.set_linewidth(0.55)
    ax.set_yscale("log")
    ax.set_ylabel(r"max $|F|$ (eV Å$^{-1}$)", fontsize=7.5)
    ax.set_xticks(ticks, labels, fontsize=6.4)
    ax.axhline(100, color="#d1495b", lw=0.7, ls="--", alpha=0.8)
    ax.text(0.99, 100, " instability-scale guide", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.6, color="#a12f40")
    ax.set_xlim(0, (len(GROUPS) - 1) * 4 + 4)
    handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="#222222", alpha=0.78, label=MODEL_LABELS[m]) for m in MODEL_ORDER]
    ax.legend(handles=handles, loc="upper left", fontsize=5.5, frameon=False, handlelength=1.2, borderpad=0.2)
    apply_axes_style(ax)


def render_force_spread(ax: plt.Axes, spread: pd.DataFrame) -> None:
    panel_label(ax, "c")
    ax.set_title("Paired force disagreement", fontsize=8.5, fontweight="bold", pad=6)
    for group in GROUPS:
        sub = spread[spread["candidate_group"] == group]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["max_force_range_eV_A"],
            sub["ensemble_force_std_rms_eV_A"],
            s=9,
            alpha=0.58,
            linewidths=0,
            color=GROUP_COLORS[group],
            label=GROUP_LABELS[group].replace("\n", " "),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"range of max $|F|$ across models (eV Å$^{-1}$)", fontsize=7.2)
    ax.set_ylabel(r"ensemble force spread $\sigma_F$ (eV Å$^{-1}$)", fontsize=7.2)
    ax.legend(loc="upper left", fontsize=5.4, frameon=False, handletextpad=0.3, borderpad=0.2, markerscale=1.2)
    apply_axes_style(ax)


def render_heatmap(ax: plt.Axes, summary: pd.DataFrame) -> None:
    panel_label(ax, "d")
    ax.set_title("Stage-wise median force by checkpoint", fontsize=8.5, fontweight="bold", pad=6)
    table = summary.pivot(index="candidate_group", columns="model_id", values="max_force_eV_A_median").reindex(index=GROUPS, columns=MODEL_ORDER)
    values = table.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="magma", norm=LogNorm(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values))))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=6.1, color="#111111" if im.norm(val) > 0.68 else "white")
    ax.set_yticks(range(len(GROUPS)), [GROUP_LABELS[g].replace("\n", " ") for g in GROUPS], fontsize=6.2)
    ax.set_xticks(range(len(MODEL_ORDER)), ["123", "234", "345"], fontsize=6.5)
    ax.set_xlabel("training seed", fontsize=7.2)
    ax.set_ylabel("source stage", fontsize=7.2)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cbar.set_label(r"median max $|F|$ (eV Å$^{-1}$)", fontsize=6.2)
    apply_axes_style(ax)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    data_dir = repo / "data/raw_compact/revision1/from_scratch_failure_attribution_20261013/mlip_results"
    output_dir = (args.output_dir or repo / "paper/revision1/figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = data_dir / "mlip_candidate_results_all.csv"
    spread_path = data_dir / "candidate_model_spread.csv"
    summary_path = data_dir / "model_stage_summary.csv"
    results = pd.read_csv(results_path)
    spread = pd.read_csv(spread_path)
    summary = pd.read_csv(summary_path)
    results = results[results["evaluation_status"].eq("success")].copy()
    counts = spread.groupby("candidate_group")["candidate_id"].nunique().reindex(GROUPS).fillna(0).astype(int).to_dict()

    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titleweight": "bold", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=False)
    grid = fig.add_gridspec(2, 2, left=0.075, right=0.98, bottom=0.08, top=0.95, wspace=0.28, hspace=0.38)
    render_selection_schematic(fig.add_subplot(grid[0, 0]), counts)
    render_force_boxplot(fig.add_subplot(grid[0, 1]), results)
    render_force_spread(fig.add_subplot(grid[1, 0]), spread)
    render_heatmap(fig.add_subplot(grid[1, 1]), summary)

    stem = output_dir / "figS_from_scratch_failure_attribution"
    for suffix, kwargs in [("pdf", {"bbox_inches": "tight"}), ("png", {"dpi": 600, "bbox_inches": "tight"}), ("svg", {"bbox_inches": "tight"})]:
        fig.savefig(stem.with_suffix(f".{suffix}"), **kwargs)
    plt.close(fig)

    manifest = {
        "schema": "migrationbench_from_scratch_failure_attribution_figure_v1",
        "status": "VERIFIED",
        "figure": stem.name,
        "source_inputs": {p.name: sha256(p) for p in (results_path, spread_path, summary_path)},
        "n_successful_model_rows": int(len(results)),
        "n_candidate_rows": int(spread["candidate_id"].nunique()),
        "n_models": int(results["model_id"].nunique()),
        "candidate_group_counts": {k: int(v) for k, v in counts.items()},
        "outputs": {f"{stem.name}.{s}": sha256(stem.with_suffix(f".{s}")) for s in ("pdf", "png", "svg")},
        "interpretation_boundary": "paired static MLIP evidence; not DFT correctness or causal proof",
    }
    (output_dir / f"{stem.name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
