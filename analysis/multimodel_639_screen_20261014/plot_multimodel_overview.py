from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from plot_atlas import apply_publication_style, get_palette, save_figure


MODEL_ORDER = [
    "ablation_lmax_2_seed123", "ablation_lmax_2_seed234", "baseline_seed345",
    "ft600k_seed123", "ft600k_seed234", "ft600k_seed345",
    "ftmultit_seed123", "ftmultit_seed234", "ftmultit_seed345",
    "foundation_omat_legacy", "foundation_mpa0_medium", "foundation_mp0b3_medium",
]
MODEL_LABELS = {
    "ablation_lmax_2_seed123": "scratch lmax2\n123", "ablation_lmax_2_seed234": "scratch lmax2\n234",
    "baseline_seed345": "scratch baseline\n345", "ft600k_seed123": "FT-600K\n123",
    "ft600k_seed234": "FT-600K\n234", "ft600k_seed345": "FT-600K\n345",
    "ftmultit_seed123": "FT-MultiT\n123", "ftmultit_seed234": "FT-MultiT\n234",
    "ftmultit_seed345": "FT-MultiT\n345", "foundation_omat_legacy": "OMAT\nlegacy",
    "foundation_mpa0_medium": "MPA-0\nmedium", "foundation_mp0b3_medium": "MP-0b3\nmedium",
}
GROUP_ORDER = ["stable_early_control", "stable_pre_onset", "precursor_near_onset", "threshold_or_terminal", "terminal_stable_target"]
GROUP_LABELS = {"stable_early_control": "early stable", "stable_pre_onset": "pre-onset", "precursor_near_onset": "precursor", "threshold_or_terminal": "threshold", "terminal_stable_target": "stable terminal"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def style(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.05)
        spine.set_color("#1b1b1b")
    ax.tick_params(width=1.0, length=4)


def colors():
    pal = get_palette("nature_steered_dynamics")["categorical"]
    return {m: pal[i % len(pal)] for i, m in enumerate(MODEL_ORDER)}


def family_colors():
    return {"ablation_lmax_2": "#C44E52", "baseline": "#8172B2", "FT-600K": "#4C72B0", "FT-MultiT": "#55A868", "Foundation-OMAT": "#DD8452", "Foundation-MPA0": "#937860", "Foundation-MP-0b3": "#DA8BC3"}


def save_all(fig, out: Path, stem: str):
    save_figure(fig, out / f"{stem}.png", dpi=350)
    fig.savefig(out / f"{stem}.pdf", dpi=350, bbox_inches="tight", metadata={"Creator": "Plot Atlas", "Date": None})
    fig.savefig(out / f"{stem}.svg", dpi=350, bbox_inches="tight", metadata={"Creator": "Plot Atlas", "Date": None})
    plt.close(fig)


def panel_heatmap(ax, results):
    med = results[results.evaluation_status == "success"].groupby(["candidate_group", "model_id"], observed=True).max_force_eV_A.median().unstack()
    med = med.reindex(index=[x for x in GROUP_ORDER if x in med.index], columns=[x for x in MODEL_ORDER if x in med.columns])
    arr = med.to_numpy(float)
    vals = arr[np.isfinite(arr)]
    vmax = max(100.0, float(np.nanpercentile(vals, 95))) if len(vals) else 100.0
    im = ax.imshow(arr, aspect="auto", cmap="magma", norm=LogNorm(vmin=max(0.1, np.nanmin(vals) if len(vals) else 0.1), vmax=vmax))
    ax.set_xticks(np.arange(len(med.columns)), [MODEL_LABELS.get(x, x) for x in med.columns], rotation=55, ha="right", fontsize=6.5)
    ax.set_yticks(np.arange(len(med.index)), [GROUP_LABELS.get(x, x) for x in med.index], fontsize=7)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i,j]:.1f}", ha="center", va="center", fontsize=6.1, color="white" if arr[i, j] < vmax ** .7 else "#111111")
    ax.set_xlabel("MLIP checkpoint", labelpad=4)
    ax.set_ylabel("Source trajectory stage")
    cbar = ax.figure.colorbar(im, ax=ax, pad=.02, fraction=.046)
    cbar.set_label("median max |F| (eV Å$^{-1}$)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    style(ax)


def panel_stage_box(ax, results):
    fam = family_colors()
    order = [x for x in GROUP_ORDER if x in set(results.candidate_group.dropna())]
    positions, labels = [], []
    for gi, group in enumerate(order):
        for fi, family in enumerate(["ablation_lmax_2", "baseline", "FT-600K", "FT-MultiT", "Foundation-OMAT", "Foundation-MPA0", "Foundation-MP-0b3"]):
            vals = pd.to_numeric(results.loc[(results.candidate_group == group) & (results.model_family == family) & (results.evaluation_status == "success"), "max_force_eV_A"], errors="coerce").dropna().to_numpy()
            if not len(vals):
                continue
            pos = gi * 8 + fi + 1
            ax.boxplot(vals, positions=[pos], widths=.7, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": fam[family], "edgecolor": "#222", "linewidth": .6},
                       medianprops={"color": "#111", "linewidth": 1.0}, whiskerprops={"color": "#222", "linewidth": .6}, capprops={"color": "#222", "linewidth": .6})
            positions.append(pos)
        labels.append(GROUP_LABELS.get(group, group))
    centers = [gi * 8 + 4 for gi in range(len(order))]
    ax.set_xticks(centers, labels, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(bottom=.2)
    ax.axhline(100, color="#222", linestyle=(0, (3, 2)), linewidth=.9)
    ax.set_ylabel("static max |F| (eV Å$^{-1}$)")
    ax.set_xlabel("source stage")
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=c, markeredgecolor="#222", markersize=6, label=k) for k, c in fam.items()]
    handles.append(Line2D([0], [0], color="#222", linestyle=(0, (3, 2)), label="100 eV Å$^{-1}$ gate"))
    ax.legend(handles=handles, frameon=True, fontsize=6.2, ncol=2, loc="upper left")
    style(ax)


def panel_embedding(ax, frame, reference):
    ref_colors = {"train": "#BDBDBD", "valid": "#969696", "test": "#636363", "neb": "#1B9E77"}
    for name, group in reference.groupby("reference"):
        ax.scatter(group.pca1, group.pca2, s=12 if name != "neb" else 26, c=ref_colors.get(name, "#777"), alpha=.5, linewidths=0, label=name)
    for name, group in frame.groupby("candidate_group"):
        ax.scatter(group.pca1, group.pca2, s=8, alpha=.55, linewidths=0, label=GROUP_LABELS.get(name, name))
    ax.set_xlabel("SOAP PCA 1")
    ax.set_ylabel("SOAP PCA 2")
    ax.legend(frameon=True, fontsize=6.2, ncol=2, loc="best")
    style(ax)


def panel_novelty(ax, spread, manifest):
    d = spread.merge(manifest[["candidate_id", "distance_to_train"]].drop_duplicates(), on="candidate_id", how="left")
    palette = {"stable_early_control": "#4C72B0", "stable_pre_onset": "#55A868", "precursor_near_onset": "#C44E52", "threshold_or_terminal": "#DD8452", "terminal_stable_target": "#8172B2"}
    for name, group in d.groupby("candidate_group"):
        ax.scatter(group.distance_to_train, group.max_force_range_eV_A, s=14, alpha=.55, color=palette.get(name, "#333"), edgecolor="white", linewidth=.25, label=GROUP_LABELS.get(name, name))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance to DFT training distribution")
    ax.set_ylabel("12-model range of max |F| (eV Å$^{-1}$)")
    ax.legend(frameon=True, fontsize=6.2, loc="best")
    style(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--frame-features", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    apply_publication_style(font_size=8.5)
    analysis = Path(args.analysis); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(analysis / "combined_results_12models.csv")
    spread = pd.read_csv(analysis / "candidate_force_spread_12models.csv")
    manifest = pd.read_csv(args.manifest)
    frame = pd.read_csv(args.frame_features)
    reference = pd.read_csv(args.reference)
    frame = frame.rename(columns={"snapshot_path": "source_snapshot"})
    frame = manifest[["candidate_id", "source_snapshot", "candidate_group"]].merge(frame, on=["source_snapshot"], how="left")
    frame = frame.drop_duplicates("candidate_id")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.3), constrained_layout=True)
    panel_heatmap(axes[0, 0], results)
    panel_stage_box(axes[0, 1], results)
    panel_embedding(axes[1, 0], frame.dropna(subset=["pca1", "pca2"]), reference)
    panel_novelty(axes[1, 1], spread, manifest)
    for ax, label in zip(axes.flat, "abcd"):
        ax.text(-.15, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    save_all(fig, out, "fig_multimodel_639_overview")
    manifest_out = {"analysis_sha256": sha256(Path(args.analysis) / "combined_analysis_manifest.json"), "plot_script_sha256": sha256(Path(__file__)), "figure": "fig_multimodel_639_overview", "status": "RECONSTRUCTION"}
    (out / "figure_manifest.json").write_text(pd.Series(manifest_out).to_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
