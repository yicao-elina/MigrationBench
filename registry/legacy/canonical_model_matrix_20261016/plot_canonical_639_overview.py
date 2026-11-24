#!/usr/bin/env python3
"""Plot the canonical accepted-model 639-candidate overview with Plot Atlas style."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_atlas import apply_publication_style, get_palette


GROUP_ORDER = [
    "stable_early_control",
    "stable_pre_onset",
    "precursor_near_onset",
    "threshold_or_terminal",
    "terminal_stable_target",
]
GROUP_LABELS = {
    "stable_early_control": "early stable",
    "stable_pre_onset": "pre-onset",
    "precursor_near_onset": "precursor",
    "threshold_or_terminal": "threshold",
    "terminal_stable_target": "stable terminal",
}
FAMILY_ORDER = [
    "Foundation",
    "Scratch / 600 K",
    "Scratch / Multi-T",
    "FT / 600 K",
    "FT / Multi-T",
]
FAMILY_COLORS = {
    "Foundation": "#D55E00",
    "Scratch / 600 K": "#E69F00",
    "Scratch / Multi-T": "#F0B323",
    "FT / 600 K": "#234B87",
    "FT / Multi-T": "#009E73",
}
TICK_SIZE = 14.0
LABEL_SIZE = 16.0
LEGEND_SIZE = 12.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family_label(value: str) -> str:
    if value == "foundation_zero_shot":
        return "Foundation"
    if value == "scratch_600k_grouped_v1":
        return "Scratch / 600 K"
    if value == "from_scratch_all_data_v1":
        return "Scratch / Multi-T"
    if value in {"ft_600k_grouped_A2", "ft_600k_5pct_grouped_v1"}:
        return "FT / 600 K"
    if value in {"ft_multit_grouped_A2", "ft_multit_5pct_grouped_v1"}:
        return "FT / Multi-T"
    return value


def style(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.05)
        spine.set_color("#1b1b1b")
    ax.tick_params(width=1.35, length=5.5, labelsize=TICK_SIZE)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontweight("bold")
        tick.set_fontsize(TICK_SIZE)
    ax.xaxis.label.set_fontsize(LABEL_SIZE)
    ax.yaxis.label.set_fontsize(LABEL_SIZE)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def save_all(fig, output: Path, stem: str) -> list[Path]:
    paths = []
    for extension, metadata in (("png", {"Software": "Plot Atlas"}), ("pdf", {"Creator": "Plot Atlas", "Date": None}), ("svg", {"Creator": "Plot Atlas", "Date": None})):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, format=extension, dpi=350, metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def panel_heatmap(ax, results: pd.DataFrame) -> None:
    data = results[results["evaluation_status"].eq("success")].copy()
    data["family_label"] = data["model_family"].map(family_label)
    med = data.groupby(["candidate_group", "family_label"], observed=True)["max_force_eV_A"].median().unstack()
    rows = [group for group in GROUP_ORDER if group in med.index]
    cols = [family for family in FAMILY_ORDER if family in med.columns]
    med = med.reindex(index=rows, columns=cols)
    values = med.to_numpy(float)
    finite = values[np.isfinite(values)]
    lower = max(0.1, float(np.nanmin(finite))) if len(finite) else 0.1
    upper = max(100.0, float(np.nanpercentile(finite, 95))) if len(finite) else 100.0
    image = ax.imshow(values, aspect="auto", cmap="magma", norm=LogNorm(vmin=lower, vmax=upper))
    ax.set_xticks(np.arange(len(cols)), cols, rotation=32, ha="right")
    ax.set_yticks(np.arange(len(rows)), [GROUP_LABELS.get(row, row) for row in rows])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                color = "white" if values[i, j] < upper ** 0.7 else "#111111"
                ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=12, color=color, fontweight="bold")
    ax.set_xlabel("accepted model family")
    ax.set_ylabel("source trajectory stage")
    colorbar = ax.figure.colorbar(image, ax=ax, pad=0.02, fraction=0.046)
    colorbar.set_label("median max |F| (eV Å$^{-1}$)", fontsize=14, fontweight="bold")
    colorbar.ax.tick_params(labelsize=12)
    style(ax)


def panel_stage_box(ax, results: pd.DataFrame) -> None:
    data = results[results["evaluation_status"].eq("success")].copy()
    data["family_label"] = data["model_family"].map(family_label)
    groups = [group for group in GROUP_ORDER if group in set(data["candidate_group"])]
    families = [family for family in FAMILY_ORDER if family in set(data["family_label"])]
    width = len(families) + 1
    for group_index, group in enumerate(groups):
        for family_index, family in enumerate(families):
            values = data.loc[(data["candidate_group"] == group) & (data["family_label"] == family), "max_force_eV_A"].dropna().to_numpy(float)
            if not len(values):
                continue
            position = group_index * width + family_index + 1
            ax.boxplot(
                values,
                positions=[position],
                widths=0.72,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": FAMILY_COLORS[family], "edgecolor": "#222", "linewidth": 0.8},
                medianprops={"color": "#111", "linewidth": 1.2},
                whiskerprops={"color": "#222", "linewidth": 0.7},
                capprops={"color": "#222", "linewidth": 0.7},
            )
            medians = data.loc[(data["candidate_group"] == group) & (data["family_label"] == family)].groupby("training_seed")["max_force_eV_A"].median()
            ax.scatter(np.full(len(medians), position), medians.to_numpy(float), s=28, color="#111", edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_xticks([group_index * width + (len(families) + 1) / 2 for group_index in range(len(groups))], [GROUP_LABELS.get(group, group) for group in groups])
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.2)
    ax.axhline(100, color="#222", linestyle=(0, (3, 2)), linewidth=0.9)
    ax.set_xlabel("source stage")
    ax.set_ylabel("static max |F| (eV Å$^{-1}$)")
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=FAMILY_COLORS[family], markeredgecolor="#222", markersize=9, label=family) for family in families]
    handles.append(Line2D([0], [0], color="#222", linestyle=(0, (3, 2)), label="100 eV Å$^{-1}$ gate"))
    ax.legend(handles=handles, frameon=True, fontsize=LEGEND_SIZE, ncol=2, loc="upper left", handlelength=1.7, borderpad=0.4)
    style(ax)


def panel_embedding(ax, frame: pd.DataFrame, reference: pd.DataFrame) -> None:
    ref_colors = {"train": "#BDBDBD", "valid": "#969696", "test": "#636363", "neb": "#1B9E77"}
    group_colors = dict(zip(GROUP_ORDER, get_palette("nature_steered_dynamics")["categorical"]))

    def draw(target, candidate_frame, reference_frame, small=False):
        for name, group in reference_frame.groupby("reference"):
            target.scatter(group["pca1"], group["pca2"], s=9 if small else (15 if name == "neb" else 10), c=ref_colors.get(name, "#777"), alpha=0.5, linewidths=0, label=name, zorder=2)
        for name, group in candidate_frame.groupby("candidate_group"):
            target.scatter(group["pca1"], group["pca2"], s=11 if small else 9, alpha=0.6, linewidths=0, color=group_colors.get(name, "#333"), label=GROUP_LABELS.get(name, name), zorder=3)

    draw(ax, frame, reference)
    ax.set_xlabel("SOAP PCA 1")
    ax.set_ylabel("SOAP PCA 2")
    ax.legend(frameon=True, fontsize=LEGEND_SIZE, ncol=2, loc="lower left", borderpad=0.4, labelspacing=0.3)
    style(ax)
    local_reference = reference[reference["reference"] != "neb"].copy()
    inset = inset_axes(ax, width="43%", height="45%", loc="upper right", borderpad=1.0)
    draw(inset, frame, local_reference, small=True)
    inset.set_xlim(-1.8, 3.2)
    inset.set_ylim(-5.35, 1.0)
    inset.set_xlabel("PCA 1", fontsize=11.5, fontweight="bold", labelpad=1.5)
    inset.set_ylabel("PCA 2", fontsize=11.5, fontweight="bold", labelpad=1.5)
    inset.tick_params(width=1.0, length=3.5, labelsize=10.5, pad=1.5)
    for tick in (*inset.get_xticklabels(), *inset.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#1b1b1b")
    inset.set_facecolor("white")
    inset.patch.set_alpha(0.96)
    inset.text(0.04, 0.96, "local reference + candidate view", transform=inset.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5})


def panel_novelty(ax, spread: pd.DataFrame, manifest: pd.DataFrame) -> None:
    data = spread.merge(manifest[["candidate_id", "distance_to_train"]].drop_duplicates(), on="candidate_id", how="left")
    group_colors = dict(zip(GROUP_ORDER, get_palette("nature_steered_dynamics")["categorical"]))
    for group, subset in data.groupby("candidate_group"):
        ax.scatter(subset["distance_to_train"], subset["max_force_range_eV_A"], s=16, alpha=0.58, color=group_colors.get(group, "#333"), edgecolor="white", linewidth=0.25, label=GROUP_LABELS.get(group, group))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance to DFT training distribution")
    ax.set_ylabel("accepted-model range of max |F| (eV Å$^{-1}$)")
    ax.legend(frameon=True, fontsize=LEGEND_SIZE, loc="best", borderpad=0.4, labelspacing=0.3)
    style(ax)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-features", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    apply_publication_style(font_size=14.0)
    analysis = Path(args.analysis)
    manifest_path = Path(args.manifest)
    frame_path = Path(args.frame_features)
    reference_path = Path(args.reference)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(analysis / "mlip_candidate_results_all.csv")
    spread = pd.read_csv(analysis / "candidate_model_spread.csv")
    manifest = pd.read_csv(manifest_path)
    frame = pd.read_csv(frame_path).rename(columns={"snapshot_path": "source_snapshot"})
    frame = manifest[["candidate_id", "source_snapshot", "candidate_group"]].merge(frame, on="source_snapshot", how="left").drop_duplicates("candidate_id")
    reference = pd.read_csv(reference_path)

    figure, axes = plt.subplots(2, 2, figsize=(16.0, 11.0), constrained_layout=True)
    panel_heatmap(axes[0, 0], results)
    panel_stage_box(axes[0, 1], results)
    panel_embedding(axes[1, 0], frame.dropna(subset=["pca1", "pca2"]), reference)
    panel_novelty(axes[1, 1], spread, manifest)
    for axis, label in zip(axes.flat, "abcd"):
        axis.text(-0.15, 1.08, label, transform=axis.transAxes, fontsize=20, fontweight="bold", va="top")
    stem = "fig_canonical_639_overview"
    figure_paths = save_all(figure, output, stem)
    manifest_out = {
        "schema": "migrationbench_canonical_639_figure_v1",
        "status": "RECONSTRUCTION",
        "analysis_manifest_sha256": sha256(analysis / "CANONICAL_ANALYSIS_MANIFEST.json"),
        "screen_analysis_root": str(analysis),
        "candidate_manifest_sha256": sha256(manifest_path),
        "frame_features_sha256": sha256(frame_path),
        "reference_embedding_sha256": sha256(reference_path),
        "plot_script_sha256": sha256(Path(__file__)),
        "figure": stem,
        "outputs": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in figure_paths},
        "panel_contract": {
            "a": "family-by-stage median heatmap; individual seed values remain in canonical_model_stage_summary.tsv",
            "b": "family stage distributions with seed medians; no vertical separator lines",
            "c": "global PCA geometry with local inset; NEB reference retains global scale",
            "d": "candidate distance versus accepted-model max-force range",
        },
        "interpretation_boundary": "paired static MLIP evidence; not DFT correctness or causal proof",
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    print(json.dumps({"status": manifest_out["status"], "outputs": [path.name for path in figure_paths]}, indent=2))


if __name__ == "__main__":
    main()
