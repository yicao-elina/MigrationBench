from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from plot_atlas import apply_publication_style, get_palette


MODEL_ORDER = [
    # Keep the four manuscript-level families contiguous.  The order within
    # each family is arm 1, arm 2, arm 3.
    "ablation_lmax_2_seed123", "ablation_lmax_2_seed234", "baseline_seed345",
    "foundation_omat_legacy", "foundation_mpa0_medium", "foundation_mp0b3_medium",
    "ft600k_seed123", "ft600k_seed234", "ft600k_seed345",
    "ftmultit_seed123", "ftmultit_seed234", "ftmultit_seed345",
]
MODEL_LABELS = {
    "ablation_lmax_2_seed123": "Lmax2\n123", "ablation_lmax_2_seed234": "Lmax2\n234",
    "baseline_seed345": "baseline\n345", "foundation_omat_legacy": "OMAT\nlegacy",
    "foundation_mpa0_medium": "MPA-0\nmedium", "foundation_mp0b3_medium": "MP-0b3\nmedium",
    "ft600k_seed123": "FT-600K\n123", "ft600k_seed234": "FT-600K\n234",
    "ft600k_seed345": "FT-600K\n345", "ftmultit_seed123": "FT-MultiT\n123",
    "ftmultit_seed234": "FT-MultiT\n234", "ftmultit_seed345": "FT-MultiT\n345",
}
FAMILY_ORDER = ["From Scratch", "Foundation / zero-shot", "FT-600K", "FT-MultiT"]
FAMILY_COLORS = {
    "From Scratch": "#E69F00",
    "Foundation / zero-shot": "#D55E00",
    "FT-600K": "#234B87",
    "FT-MultiT": "#009E73",
}
ARM_SLOTS = {
    "ablation_lmax_2_seed123": "arm 1",
    "ablation_lmax_2_seed234": "arm 2",
    "baseline_seed345": "arm 3",
    "foundation_omat_legacy": "arm 1",
    "foundation_mpa0_medium": "arm 2",
    "foundation_mp0b3_medium": "arm 3",
    "ft600k_seed123": "arm 1",
    "ft600k_seed234": "arm 2",
    "ft600k_seed345": "arm 3",
    "ftmultit_seed123": "arm 1",
    "ftmultit_seed234": "arm 2",
    "ftmultit_seed345": "arm 3",
}
ARM_HATCH = {"arm 1": "", "arm 2": "///", "arm 3": "..."}
ARM_MARKER = {"arm 1": "o", "arm 2": "s", "arm 3": "^"}
GROUP_ORDER = ["stable_early_control", "stable_pre_onset", "precursor_near_onset", "threshold_or_terminal", "terminal_stable_target"]
GROUP_LABELS = {"stable_early_control": "early stable", "stable_pre_onset": "pre-onset", "precursor_near_onset": "precursor", "threshold_or_terminal": "threshold", "terminal_stable_target": "stable terminal"}

# The original rendering was legible only at full-screen zoom.  Keep the
# figure dimensions publication-compatible, but make the plotted typography
# comfortably readable at the intended manuscript size.
TICK_FONTSIZE = 14.0
AXIS_LABEL_FONTSIZE = 16.0
LEGEND_FONTSIZE = 12.0
PANEL_LETTER_FONTSIZE = 20.0


def family_for_model(model_id: str) -> str:
    if model_id.startswith("ablation_") or model_id.startswith("baseline_"):
        return "From Scratch"
    if model_id.startswith("foundation_"):
        return "Foundation / zero-shot"
    if model_id.startswith("ft600k_"):
        return "FT-600K"
    if model_id.startswith("ftmultit_"):
        return "FT-MultiT"
    raise KeyError(f"Unmapped model ID: {model_id}")


def arm_label(model_id: str) -> str:
    labels = {
        "ablation_lmax_2_seed123": "Lmax2 / seed 123",
        "ablation_lmax_2_seed234": "Lmax2 / seed 234",
        "baseline_seed345": "baseline / seed 345",
        "foundation_omat_legacy": "OMAT legacy",
        "foundation_mpa0_medium": "MPA-0 medium",
        "foundation_mp0b3_medium": "MP-0b3 medium",
    }
    if model_id in labels:
        return labels[model_id]
    return MODEL_LABELS[model_id].replace("\n", " / ")


def arm_semantics(model_id: str) -> str:
    family = family_for_model(model_id)
    if family in {"FT-600K", "FT-MultiT"}:
        return "same-family training-seed replica"
    if family == "Foundation / zero-shot":
        return "zero-shot checkpoint variant; training seed not applicable"
    if model_id == "baseline_seed345":
        return "different from-scratch baseline branch; not Lmax2 replica"
    return "Lmax2 from-scratch training-seed branch"


def write_model_grouping_manifest() -> Path:
    registry_path = Path(__file__).resolve().parent / "MODEL_REGISTRY.tsv"
    registry = pd.read_csv(registry_path, sep="\t")
    by_id = registry.set_index("canonical_model_id")
    missing = [model_id for model_id in MODEL_ORDER if model_id not in by_id.index]
    if missing:
        raise ValueError(f"MODEL_REGISTRY.tsv is missing plotted models: {missing}")
    rows = []
    for model_id in MODEL_ORDER:
        row = by_id.loc[model_id]
        family = family_for_model(model_id)
        rows.append({
            "canonical_model_id": model_id,
            "major_group": family,
            "arm_slot": ARM_SLOTS[model_id],
            "arm_label": arm_label(model_id),
            "model_family_raw": row["model_family"],
            "training_seed": row["training_seed"],
            "training_seed_interpretation": "not applicable" if family == "Foundation / zero-shot" else "registered training seed",
            "arm_semantics": arm_semantics(model_id),
            "lineage_status": row["lineage_status"],
            "model_sha256": row["model_sha256"],
        })
    out = Path(__file__).resolve().parent / "MODEL_GROUPING.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    return out


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
    ax.tick_params(width=1.35, length=5.5, labelsize=TICK_FONTSIZE)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontweight("bold")
        tick.set_fontsize(TICK_FONTSIZE)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONTSIZE)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def colors():
    pal = get_palette("nature_steered_dynamics")["categorical"]
    return {m: pal[i % len(pal)] for i, m in enumerate(MODEL_ORDER)}


def family_colors():
    return FAMILY_COLORS


def save_all(fig, out: Path, stem: str):
    # Plot Atlas' generic save_figure helper performs an additional
    # bbox_inches="tight" SVG pass.  That pass can stall on a composite that
    # contains inset_axes and mark_inset; constrained_layout has already
    # reserved the required whitespace here, so write all three formats
    # directly and deterministically.
    # constrained_layout already reserves space for the enlarged typography.
    # Write to temporary siblings and atomically replace the published files;
    # this avoids transient file-provider locks on the existing SVG/PDF while
    # keeping the figure release self-contained.
    for ext, metadata in (
        ("png", {"Software": "Plot Atlas"}),
        ("pdf", {"Creator": "Plot Atlas", "Date": None}),
        ("svg", {"Creator": "Plot Atlas", "Date": None}),
    ):
        target = out / f"{stem}.{ext}"
        temporary = out / f".{stem}.{ext}.tmp"
        fig.savefig(temporary, format=ext, dpi=350, metadata=metadata)
        os.replace(temporary, target)
    plt.close(fig)


def panel_heatmap(ax, results):
    med = results[results.evaluation_status == "success"].groupby(["candidate_group", "model_id"], observed=True).max_force_eV_A.median().unstack()
    med = med.reindex(index=[x for x in GROUP_ORDER if x in med.index], columns=[x for x in MODEL_ORDER if x in med.columns])
    arr = med.to_numpy(float)
    vals = arr[np.isfinite(arr)]
    vmax = max(100.0, float(np.nanpercentile(vals, 95))) if len(vals) else 100.0
    im = ax.imshow(arr, aspect="auto", cmap="magma", norm=LogNorm(vmin=max(0.1, np.nanmin(vals) if len(vals) else 0.1), vmax=vmax))
    ax.set_xticks(np.arange(len(med.columns)), [MODEL_LABELS.get(x, x) for x in med.columns], rotation=48, ha="right", fontsize=12.5)
    ax.set_yticks(np.arange(len(med.index)), [GROUP_LABELS.get(x, x) for x in med.index], fontsize=14.0)
    for tick, model_id in zip(ax.get_xticklabels(), med.columns):
        tick.set_color(FAMILY_COLORS[family_for_model(model_id)])
        tick.set_fontweight("bold")
    # White separators and a coloured family strip make the nested grouping
    # visible even though the cell colour itself remains the quantitative
    # force-scale colour.
    for boundary in (2.5, 5.5, 8.5):
        ax.axvline(boundary, color="white", linewidth=2.8, zorder=3)
    for start, end, family in ((0, 2, "From Scratch"), (3, 5, "Foundation / zero-shot"),
                               (6, 8, "FT-600K"), (9, 11, "FT-MultiT")):
        center = (start + end) / 2
        ax.plot([start - .42, end + .42], [1.015, 1.015], transform=ax.get_xaxis_transform(),
                color=FAMILY_COLORS[family], linewidth=5.0, solid_capstyle="butt", clip_on=False)
        family_label = "Foundation /\nzero-shot" if family == "Foundation / zero-shot" else family
        ax.text(center, 1.055, family_label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=12.5, fontweight="bold", color=FAMILY_COLORS[family], clip_on=False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i,j]:.1f}", ha="center", va="center", fontsize=10.8, color="white" if arr[i, j] < vmax ** .7 else "#111111")
    ax.set_xlabel("MLIP checkpoint", labelpad=4)
    ax.set_ylabel("Source trajectory stage")
    cbar = ax.figure.colorbar(im, ax=ax, pad=.02, fraction=.046)
    cbar.set_label("median max |F| (eV Å$^{-1}$)", fontsize=14.0, fontweight="bold")
    cbar.ax.tick_params(labelsize=12.5, width=1.1)
    style(ax)


def panel_stage_box(ax, results):
    order = [x for x in GROUP_ORDER if x in set(results.candidate_group.dropna())]
    positions, labels = [], []
    for gi, group in enumerate(order):
        for mi, model_id in enumerate(MODEL_ORDER):
            family = family_for_model(model_id)
            vals = pd.to_numeric(results.loc[(results.candidate_group == group) & (results.model_id == model_id) & (results.evaluation_status == "success"), "max_force_eV_A"], errors="coerce").dropna().to_numpy()
            if not len(vals):
                continue
            pos = gi * 14 + mi + 1
            arm = ARM_SLOTS[model_id]
            ax.boxplot(vals, positions=[pos], widths=.7, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": FAMILY_COLORS[family], "edgecolor": "#222", "linewidth": .7, "hatch": ARM_HATCH[arm]},
                       medianprops={"color": "#111", "linewidth": 1.0}, whiskerprops={"color": "#222", "linewidth": .6}, capprops={"color": "#222", "linewidth": .6})
            positions.append(pos)
        labels.append(GROUP_LABELS.get(group, group))
        # Keep the panel visually uncluttered: family identity is already
        # encoded by box colour and arm identity by hatching.  The horizontal
        # 100 eV A^-1 gate remains the only reference line in this panel.
    centers = [gi * 14 + 6.5 for gi in range(len(order))]
    ax.set_xticks(centers, labels, fontsize=14.0)
    ax.set_yscale("log")
    ax.set_ylim(bottom=.2)
    ax.axhline(100, color="#222", linestyle=(0, (3, 2)), linewidth=.9)
    ax.set_ylabel("static max |F| (eV Å$^{-1}$)")
    ax.set_xlabel("source stage")
    family_handles = [Patch(facecolor=color, edgecolor="#222", label=family) for family, color in FAMILY_COLORS.items()]
    arm_handles = [Patch(facecolor="white", edgecolor="#222", hatch=ARM_HATCH[arm], label=arm) for arm in ("arm 1", "arm 2", "arm 3")]
    handles = family_handles + arm_handles + [Line2D([0], [0], color="#222", linestyle=(0, (3, 2)), label="100 eV Å$^{-1}$ gate")]
    ax.legend(handles=handles, frameon=True, fontsize=12.0, ncol=2, loc="upper left", handlelength=1.8,
              columnspacing=.9, handletextpad=.4, borderpad=.45)
    style(ax)


def panel_embedding(ax, frame, reference):
    ref_colors = {"train": "#BDBDBD", "valid": "#969696", "test": "#636363", "neb": "#1B9E77"}

    def draw_points(target, local_frame, local_reference, *, inset=False):
        ref_size = 9 if inset else 12
        neb_size = 15 if inset else 26
        candidate_size = 12 if inset else 8
        for name, group in local_reference.groupby("reference"):
            target.scatter(
                group.pca1,
                group.pca2,
                s=ref_size if name != "neb" else neb_size,
                c=ref_colors.get(name, "#777"),
                alpha=.5,
                linewidths=0,
                label=name,
                zorder=2,
            )
        for name, group in local_frame.groupby("candidate_group"):
            target.scatter(
                group.pca1,
                group.pca2,
                s=candidate_size,
                alpha=.58,
                linewidths=0,
                label=GROUP_LABELS.get(name, name),
                zorder=3,
            )

    draw_points(ax, frame, reference)
    ax.set_xlabel("SOAP PCA 1")
    ax.set_ylabel("SOAP PCA 2")
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, ncol=2, loc="lower left", handlelength=1.6,
              borderpad=.45, labelspacing=.35)
    style(ax)

    # The NEB reference branch is separated from the training/candidate cloud
    # by roughly two orders of magnitude in PCA2.  A single linear axis is
    # retained for the global geometry, while this inset exposes the local
    # structure that would otherwise be compressed against y=0.
    local_reference = reference[reference.reference != "neb"].copy()
    axins = inset_axes(ax, width="43%", height="45%", loc="upper right", borderpad=1.0)
    draw_points(axins, frame, local_reference, inset=True)
    axins.set_xlim(-1.8, 3.2)
    axins.set_ylim(-5.35, 1.0)
    axins.set_xlabel("PCA 1", fontsize=11.5, fontweight="bold", labelpad=1.5)
    axins.set_ylabel("PCA 2", fontsize=11.5, fontweight="bold", labelpad=1.5)
    axins.tick_params(width=1.0, length=3.5, labelsize=10.5, pad=1.5)
    for tick in (*axins.get_xticklabels(), *axins.get_yticklabels()):
        tick.set_fontweight("bold")
    for spine in axins.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#1b1b1b")
    axins.set_facecolor("white")
    axins.patch.set_alpha(.96)
    axins.text(
        .04,
        .96,
        "local reference + candidate view",
        transform=axins.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": 1.5},
    )
    # A local rectangle is less distracting than long connector lines across
    # the global NEB scale, while still making the zoomed region explicit.
    ax.add_patch(Rectangle(
        (-1.8, -5.35),
        5.0,
        6.35,
        fill=False,
        edgecolor="#444444",
        linewidth=.9,
        linestyle=(0, (3, 2)),
        zorder=4,
    ))


def panel_novelty(ax, spread, manifest):
    d = spread.merge(manifest[["candidate_id", "distance_to_train"]].drop_duplicates(), on="candidate_id", how="left")
    palette = {"stable_early_control": "#4C72B0", "stable_pre_onset": "#55A868", "precursor_near_onset": "#C44E52", "threshold_or_terminal": "#DD8452", "terminal_stable_target": "#8172B2"}
    for name, group in d.groupby("candidate_group"):
        ax.scatter(group.distance_to_train, group.max_force_range_eV_A, s=14, alpha=.55, color=palette.get(name, "#333"), edgecolor="white", linewidth=.25, label=GROUP_LABELS.get(name, name))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance to DFT training distribution")
    ax.set_ylabel("12-model range of max |F| (eV Å$^{-1}$)")
    ax.legend(frameon=True, fontsize=LEGEND_FONTSIZE, loc="best", handlelength=1.6,
              borderpad=.45, labelspacing=.35)
    style(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--frame-features", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    grouping_path = write_model_grouping_manifest()
    apply_publication_style(font_size=14.0)
    analysis = Path(args.analysis); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(analysis / "combined_results_12models.csv")
    spread = pd.read_csv(analysis / "candidate_force_spread_12models.csv")
    manifest = pd.read_csv(args.manifest)
    frame = pd.read_csv(args.frame_features)
    reference = pd.read_csv(args.reference)
    frame = frame.rename(columns={"snapshot_path": "source_snapshot"})
    frame = manifest[["candidate_id", "source_snapshot", "candidate_group"]].merge(frame, on=["source_snapshot"], how="left")
    frame = frame.drop_duplicates("candidate_id")
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0), constrained_layout=True)
    panel_heatmap(axes[0, 0], results)
    panel_stage_box(axes[0, 1], results)
    panel_embedding(axes[1, 0], frame.dropna(subset=["pca1", "pca2"]), reference)
    panel_novelty(axes[1, 1], spread, manifest)
    for ax, label in zip(axes.flat, "abcd"):
        ax.text(-.15, 1.08, label, transform=ax.transAxes, fontsize=PANEL_LETTER_FONTSIZE, fontweight="bold", va="top")
    output_stem = "fig_multimodel_639_overview_family_grouped"
    save_all(fig, out, output_stem)
    manifest_out = {
        "analysis_sha256": sha256(Path(args.analysis) / "combined_analysis_manifest.json"),
        "plot_script_sha256": sha256(Path(__file__)),
        "figure": output_stem,
        "status": "RECONSTRUCTION",
        "embedding_inset": {
            "purpose": "preserve global NEB PCA scale while resolving local training/reference/candidate structure",
            "main_axis_limits": "data-driven global limits",
            "local_xlim": [-1.8, 3.2],
            "local_ylim": [-5.35, 1.0],
            "neb_in_main_only": True,
        },
        "typography_pt": {
            "tick": TICK_FONTSIZE,
            "axis_label": AXIS_LABEL_FONTSIZE,
            "legend": LEGEND_FONTSIZE,
            "panel_letter": PANEL_LETTER_FONTSIZE,
        },
        "model_grouping_sha256": sha256(grouping_path),
        "model_grouping": {
            "primary_channel": "family colour",
            "secondary_channel": "within-family arm hatch/label",
            "families": FAMILY_ORDER,
            "arm_semantics": {
                "From Scratch": "Lmax2 seed123, Lmax2 seed234, baseline seed345; not three identical parameterizations",
                "Foundation / zero-shot": "OMAT legacy, MPA-0 medium, MP-0b3 medium; checkpoint variants, not random seeds",
                "FT-600K": "training seeds 123, 234, 345",
                "FT-MultiT": "training seeds 123, 234, 345",
            },
        },
    }
    figure_paths = [out / f"{output_stem}.{ext}" for ext in ("png", "pdf", "svg")]
    manifest_out["outputs"] = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in figure_paths
    }
    # Keep the older paper/Obsidian filename as an atomic compatibility alias
    # so existing links resolve to the repaired inset version as well.
    legacy_stem = "fig_multimodel_639_overview"
    legacy_paths = []
    for ext in ("png", "pdf", "svg"):
        source = out / f"{output_stem}.{ext}"
        target = out / f"{legacy_stem}.{ext}"
        temporary = out / f".{legacy_stem}.{ext}.tmp"
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
        legacy_paths.append(target)
    manifest_out["compatibility_alias"] = {
        "figure": legacy_stem,
        "manifest": "figure_manifest.json",
        "outputs": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in legacy_paths
        },
    }
    (out / "figure_manifest_family_grouped.json").write_text(pd.Series(manifest_out).to_json(indent=2) + "\n")
    legacy_manifest = dict(manifest_out)
    legacy_manifest["figure"] = legacy_stem
    legacy_manifest["alias_of"] = output_stem
    (out / "figure_manifest.json").write_text(pd.Series(legacy_manifest).to_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
