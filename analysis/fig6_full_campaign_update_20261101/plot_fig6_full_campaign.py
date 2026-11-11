#!/usr/bin/env python3
"""Plot the versioned four-panel Figure 6 update from the 12-model static screen.

The source campaign is the 639-candidate / 12-model force-evaluation package from
MigrationBench PR #11. This script deliberately keeps the original four-panel
diagnostic structure while applying the Plot Atlas publication contract:

* each panel is independently exportable;
* the composite owns layout only;
* Arial-first typography, bold ticks, no decorative grid, and explicit provenance;
* the PCA panel uses an inset so the NEB reference does not hide the candidate
  cloud near the origin;
* the singleton stable-terminal point is rendered as a point, not as a fake box.

The local Plot Atlas package is not imported here because the recovered package
is currently an unhydrated cloud placeholder. The style constants below are the
same Atlas contract used by the historical script and are recorded in the
generated manifest as a local-contract fallback.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
OUT = ROOT / "figures"
QA = ROOT / "qa"

MODEL_GROUPING = SOURCE / "MODEL_GROUPING.tsv"
RESULTS = SOURCE / "combined_results_12models.csv"
MANIFEST = SOURCE / "candidate_manifest_639.csv"
FRAME_FEATURES = SOURCE / "frame_features.csv"
REFERENCE = SOURCE / "reference_embedding.csv"
SPREAD = SOURCE / "candidate_force_spread_12models.csv"

MODEL_ORDER = [
    "ablation_lmax_2_seed123",
    "ablation_lmax_2_seed234",
    "baseline_seed345",
    "foundation_omat_legacy",
    "foundation_mpa0_medium",
    "foundation_mp0b3_medium",
    "ft600k_seed123",
    "ft600k_seed234",
    "ft600k_seed345",
    "ftmultit_seed123",
    "ftmultit_seed234",
    "ftmultit_seed345",
]

FAMILY_ORDER = [
    "From Scratch",
    "Foundation / zero-shot",
    "FT-600K",
    "FT-MultiT",
]
FAMILY_COLORS = {
    "From Scratch": "#E69F00",
    "Foundation / zero-shot": "#D55E00",
    "FT-600K": "#234B87",
    "FT-MultiT": "#009E73",
}
STAGE_ORDER = [
    "stable_early_control",
    "stable_pre_onset",
    "precursor_near_onset",
    "threshold_or_terminal",
    "terminal_stable_target",
]
STAGE_LABELS = {
    "stable_early_control": "Stable early\n(n=207)",
    "stable_pre_onset": "Stable pre-onset\n(n=216)",
    "precursor_near_onset": "Precursor\n(n=208)",
    "threshold_or_terminal": "Threshold / terminal\n(n=7)",
    "terminal_stable_target": "Stable terminal\n(n=1)",
}
STAGE_COLORS = {
    "stable_early_control": "#4C78A8",
    "stable_pre_onset": "#72B7B2",
    "precursor_near_onset": "#F2CF5B",
    "threshold_or_terminal": "#F58518",
    "terminal_stable_target": "#E45756",
}
ARM_HATCHES = {"arm 1": "", "arm 2": "///", "arm 3": "\\\\\\"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atlas_rc() -> dict:
    try:
        fm.findfont("Arial", fallback_to_default=False)
        font = "Arial"
    except ValueError:
        font = "DejaVu Sans"
    mpl.rcParams.update(
        {
            "font.family": font,
            "font.size": 11.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    return {"font_requested": "Arial", "font_selected": font}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    grouping = pd.read_csv(MODEL_GROUPING, sep="\t", dtype=str)
    grouping = grouping.set_index("canonical_model_id").loc[MODEL_ORDER].reset_index()
    results = pd.read_csv(RESULTS)
    results = results[results["evaluation_status"].eq("success")].copy()
    results["max_force_eV_A"] = pd.to_numeric(results["max_force_eV_A"], errors="coerce")
    results = results[np.isfinite(results["max_force_eV_A"])].copy()
    manifest = pd.read_csv(MANIFEST)
    features = pd.read_csv(FRAME_FEATURES)
    if len(features) != len(manifest):
        raise ValueError("frame_features and candidate_manifest row counts differ")
    # The feature table is keyed by source/step/seed rather than candidate_id.
    join_keys = ["source", "step", "training_seed", "window"]
    feature_subset = features[join_keys + ["pca1", "pca2"]].copy()
    candidates = manifest.merge(feature_subset, on=join_keys, how="left", validate="one_to_one")
    if candidates[["pca1", "pca2"]].isna().any().any():
        raise ValueError("unmatched PCA feature rows")
    spread = pd.read_csv(SPREAD)
    if results["model_id"].nunique() != len(MODEL_ORDER):
        raise ValueError("the source screen does not contain exactly 12 models")
    if results["candidate_id"].nunique() != 639:
        raise ValueError("the source screen does not contain exactly 639 candidates")
    if len(results) != 7668:
        raise ValueError(f"expected 7668 successful rows, found {len(results)}")
    if results.duplicated(["candidate_id", "model_id"]).any():
        raise ValueError("duplicate candidate/model pairs")
    if not np.isfinite(results["max_force_eV_A"]).all():
        raise ValueError("non-finite max-force values")
    if set(results["model_id"]) != set(MODEL_ORDER):
        raise ValueError("model order mismatch")
    return results, candidates, spread, grouping, {
        "n_candidates": int(results["candidate_id"].nunique()),
        "n_models": int(results["model_id"].nunique()),
        "n_successful_rows": int(len(results)),
        "n_duplicate_candidate_model_pairs": int(results.duplicated(["candidate_id", "model_id"]).sum()),
    }


def apply_axes(ax):
    ax.tick_params(direction="out", length=4, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")
    ax.grid(False)


def panel_title(ax, label: str, title: str):
    ax.set_title(f"{label}  {title}", loc="left", pad=8)


def family_for(grouping: pd.DataFrame, model_id: str) -> str:
    return grouping.loc[grouping["canonical_model_id"].eq(model_id), "major_group"].iloc[0]


def panel_heatmap(results, grouping, standalone: bool = False):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=False)
    matrix = np.full((len(STAGE_ORDER), len(MODEL_ORDER)), np.nan)
    for iy, stage in enumerate(STAGE_ORDER):
        for ix, model in enumerate(MODEL_ORDER):
            values = results.loc[
                results["candidate_group"].eq(stage) & results["model_id"].eq(model),
                "max_force_eV_A",
            ].to_numpy()
            if values.size:
                matrix[iy, ix] = np.nanmedian(values)
    finite = matrix[np.isfinite(matrix)]
    norm = LogNorm(vmin=max(0.5, float(np.nanmin(finite))), vmax=float(np.nanmax(finite)))
    image = ax.imshow(matrix, aspect="auto", cmap="magma", norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels([grouping.set_index("canonical_model_id").loc[m, "arm_label"] for m in MODEL_ORDER], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(STAGE_ORDER)))
    ax.set_yticklabels([STAGE_LABELS[s] for s in STAGE_ORDER])
    ax.set_xlabel("Model arm")
    ax.set_ylabel("Candidate stage")
    panel_title(ax, "(a)", "Median max force across 639 candidates")
    for iy in range(matrix.shape[0]):
        for ix in range(matrix.shape[1]):
            value = matrix[iy, ix]
            if np.isfinite(value):
                color = "white" if norm(value) < 0.65 else "black"
                ax.text(ix, iy, f"{value:.2g}", ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")
    for boundary in [2.5, 5.5, 8.5]:
        ax.axvline(boundary, color="white", linewidth=1.5)
    strip = ax.inset_axes([0, 1.015, 1, 0.04])
    colors = [FAMILY_COLORS[family_for(grouping, m)] for m in MODEL_ORDER]
    strip.imshow([np.arange(len(MODEL_ORDER))], aspect="auto", cmap=ListedColormap(colors))
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, pad=0.03, fraction=0.045)
    cbar.set_label("Median max force (eV/A)", fontweight="bold")
    cbar.ax.tick_params(labelsize=9)
    apply_axes(ax)
    fig.subplots_adjust(left=0.20, right=0.88, top=0.82, bottom=0.28)
    return fig, ax


def panel_stage_box(results, grouping, standalone: bool = False):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=False)
    x_centers = np.arange(len(STAGE_ORDER))
    offsets = np.linspace(-0.36, 0.36, len(MODEL_ORDER))
    width = 0.055
    grouping_index = grouping.set_index("canonical_model_id")
    for im, model in enumerate(MODEL_ORDER):
        family = grouping_index.loc[model, "major_group"]
        face = FAMILY_COLORS[family]
        hatch = ARM_HATCHES[grouping_index.loc[model, "arm_slot"]]
        for istage, stage in enumerate(STAGE_ORDER):
            vals = results.loc[
                results["candidate_group"].eq(stage) & results["model_id"].eq(model),
                "max_force_eV_A",
            ].dropna().to_numpy()
            x = x_centers[istage] + offsets[im]
            if vals.size == 1:
                ax.scatter([x], vals, s=28, marker="D", facecolor=face, edgecolor="#111111", linewidth=0.6, zorder=4)
            elif vals.size > 1:
                bp = ax.boxplot(
                    vals,
                    positions=[x],
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    whis=(5, 95),
                    manage_ticks=False,
                    medianprops={"color": "#111111", "linewidth": 1.0},
                    whiskerprops={"color": "#222222", "linewidth": 0.65},
                    capprops={"color": "#222222", "linewidth": 0.65},
                    boxprops={"facecolor": face, "edgecolor": "#222222", "linewidth": 0.55},
                )
                bp["boxes"][0].set_hatch(hatch)
                bp["boxes"][0].set_alpha(0.85)
    ax.set_yscale("log")
    ax.set_xlim(-0.55, len(STAGE_ORDER) - 0.45)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER])
    ax.set_ylabel("Max force (eV/A)")
    ax.set_xlabel("Candidate stage")
    ax.axhline(100, color="#333333", linewidth=1.0, linestyle=(0, (4, 2)), zorder=1)
    panel_title(ax, "(b)", "Model force distributions by candidate stage")
    family_handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor="#222222", label=f) for f in FAMILY_ORDER]
    arm_handles = [Patch(facecolor="white", edgecolor="#222222", hatch=ARM_HATCHES[h], label=label) for h, label in [("arm 1", "arm 1"), ("arm 2", "arm 2"), ("arm 3", "arm 3")]]
    gate_handle = Line2D([0], [0], color="#333333", linestyle=(0, (4, 2)), linewidth=1.0, label="100 eV/A")
    legend1 = ax.legend(handles=family_handles, loc="upper left", bbox_to_anchor=(0, -0.19), ncol=2, fontsize=9, handlelength=1.3, columnspacing=1.2)
    ax.add_artist(legend1)
    ax.legend(handles=arm_handles + [gate_handle], loc="upper right", bbox_to_anchor=(1, -0.19), ncol=2, fontsize=9, handlelength=1.5, columnspacing=1.2)
    apply_axes(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.31)
    return fig, ax


def panel_embedding(candidates, standalone: bool = False):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=False)
    ref = pd.read_csv(REFERENCE)
    ref_styles = {
        "train": ("#111111", "o", "Train"),
        "valid": ("#666666", "s", "Valid"),
        "test": ("#AAAAAA", "^", "Test"),
        "neb": ("#7A3E9D", "D", "NEB reference"),
    }
    ref_handles = []
    for name in ["train", "valid", "test", "neb"]:
        sub = ref[ref["reference"].eq(name)]
        color, marker, label = ref_styles[name]
        ax.scatter(sub["pca1"], sub["pca2"], s=16 if name != "neb" else 28, c=color, marker=marker, alpha=0.75, linewidths=0.3, edgecolors="white", label=label)
        ref_handles.append(Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor="white", markersize=6, label=label))
    for stage in STAGE_ORDER:
        sub = candidates[candidates["candidate_group"].eq(stage)]
        ax.scatter(sub["pca1"], sub["pca2"], s=15 if len(sub) > 1 else 42, c=STAGE_COLORS[stage], marker="o" if len(sub) > 1 else "D", alpha=0.78, linewidths=0.25, edgecolors="white")
    stage_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=STAGE_COLORS[s], markeredgecolor="white", markersize=6, label=STAGE_LABELS[s].replace("\n", " ")) for s in STAGE_ORDER]
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    panel_title(ax, "(c)", "Candidate and reference embedding")
    apply_axes(ax)
    # Inset excludes the remote NEB cluster and shows the candidate cloud at usable scale.
    non_neb = ref[ref["reference"].ne("neb")]
    inset = ax.inset_axes([0.57, 0.12, 0.38, 0.40])
    for name in ["train", "valid", "test"]:
        sub = ref[ref["reference"].eq(name)]
        color, marker, _ = ref_styles[name]
        inset.scatter(sub["pca1"], sub["pca2"], s=8, c=color, marker=marker, alpha=0.55, linewidths=0.2, edgecolors="white")
    for stage in STAGE_ORDER:
        sub = candidates[candidates["candidate_group"].eq(stage)]
        inset.scatter(sub["pca1"], sub["pca2"], s=7 if len(sub) > 1 else 24, c=STAGE_COLORS[stage], marker="o" if len(sub) > 1 else "D", alpha=0.7, linewidths=0.15, edgecolors="white")
    x = np.concatenate([non_neb["pca1"].to_numpy(), candidates["pca1"].to_numpy()])
    y = np.concatenate([non_neb["pca2"].to_numpy(), candidates["pca2"].to_numpy()])
    inset.set_xlim(float(x.min()) - 0.4, float(x.max()) + 0.4)
    inset.set_ylim(float(y.min()) - 0.4, float(y.max()) + 0.4)
    inset.tick_params(labelsize=7, length=2)
    inset.set_title("Near-origin detail", fontsize=8.5, fontweight="bold", pad=3)
    apply_axes(inset)
    ax.legend(handles=ref_handles + stage_handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8.5, handlelength=1.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.26)
    return fig, ax


def panel_novelty(candidates, spread, standalone: bool = False):
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=False)
    plot = candidates[["candidate_id", "candidate_group", "distance_to_train"]].merge(spread[["candidate_id", "max_force_range_eV_A"]], on="candidate_id", how="left", validate="one_to_one")
    for stage in STAGE_ORDER:
        sub = plot[plot["candidate_group"].eq(stage)]
        ax.scatter(sub["distance_to_train"], sub["max_force_range_eV_A"], s=22 if len(sub) > 1 else 48, marker="o" if len(sub) > 1 else "D", color=STAGE_COLORS[stage], edgecolors="white", linewidths=0.35, alpha=0.8, label=STAGE_LABELS[stage].replace("\n", " "))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance to training set (feature units)")
    ax.set_ylabel("Across-model max-force range (eV/A)")
    panel_title(ax, "(d)", "Distribution shift versus model disagreement")
    ax.axhline(100, color="#333333", linewidth=1.0, linestyle=(0, (4, 2)))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=3, fontsize=8.8, handlelength=1.2, columnspacing=1.0)
    apply_axes(ax)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.27)
    return fig, ax


def save_figure(fig, stem: str):
    paths = []
    for ext in ["png", "pdf", "svg"]:
        path = OUT / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def composite(results, candidates, spread, grouping):
    fig = plt.figure(figsize=(15.2, 10.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.055, right=0.985, top=0.965, bottom=0.065, wspace=0.25, hspace=0.38)
    # Recreate each primitive on its own axes so the composite owns layout only.
    axa = fig.add_subplot(gs[0, 0])
    matrix = np.full((len(STAGE_ORDER), len(MODEL_ORDER)), np.nan)
    for iy, stage in enumerate(STAGE_ORDER):
        for ix, model in enumerate(MODEL_ORDER):
            vals = results.loc[results["candidate_group"].eq(stage) & results["model_id"].eq(model), "max_force_eV_A"].to_numpy()
            if vals.size:
                matrix[iy, ix] = np.nanmedian(vals)
    finite = matrix[np.isfinite(matrix)]
    norm = LogNorm(vmin=max(0.5, float(np.nanmin(finite))), vmax=float(np.nanmax(finite)))
    im = axa.imshow(matrix, aspect="auto", cmap="magma", norm=norm, interpolation="nearest")
    axa.set_xticks(range(len(MODEL_ORDER)))
    axa.set_xticklabels([grouping.set_index("canonical_model_id").loc[m, "arm_label"] for m in MODEL_ORDER], rotation=45, ha="right")
    axa.set_yticks(range(len(STAGE_ORDER)))
    axa.set_yticklabels([STAGE_LABELS[s] for s in STAGE_ORDER])
    axa.set_xlabel("Model arm")
    axa.set_ylabel("Candidate stage")
    panel_title(axa, "(a)", "Median max force across 639 candidates")
    for iy in range(matrix.shape[0]):
        for ix in range(matrix.shape[1]):
            value = matrix[iy, ix]
            if np.isfinite(value):
                color = "white" if norm(value) < 0.65 else "black"
                axa.text(ix, iy, f"{value:.2g}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    for boundary in [2.5, 5.5, 8.5]:
        axa.axvline(boundary, color="white", linewidth=1.3)
    strip = axa.inset_axes([0, 1.015, 1, 0.04])
    colors = [FAMILY_COLORS[family_for(grouping, m)] for m in MODEL_ORDER]
    strip.imshow([np.arange(len(MODEL_ORDER))], aspect="auto", cmap=ListedColormap(colors))
    strip.set_xticks([])
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axa, pad=0.025, fraction=0.045)
    cbar.set_label("Median max force (eV/A)", fontweight="bold")
    cbar.ax.tick_params(labelsize=8.5)
    apply_axes(axa)

    axb = fig.add_subplot(gs[0, 1])
    x_centers = np.arange(len(STAGE_ORDER))
    offsets = np.linspace(-0.36, 0.36, len(MODEL_ORDER))
    grouping_index = grouping.set_index("canonical_model_id")
    for imodel, model in enumerate(MODEL_ORDER):
        family = grouping_index.loc[model, "major_group"]
        face = FAMILY_COLORS[family]
        hatch = ARM_HATCHES[grouping_index.loc[model, "arm_slot"]]
        for istage, stage in enumerate(STAGE_ORDER):
            vals = results.loc[results["candidate_group"].eq(stage) & results["model_id"].eq(model), "max_force_eV_A"].dropna().to_numpy()
            x = x_centers[istage] + offsets[imodel]
            if vals.size == 1:
                axb.scatter([x], vals, s=28, marker="D", facecolor=face, edgecolor="#111111", linewidth=0.6, zorder=4)
            elif vals.size > 1:
                bp = axb.boxplot(vals, positions=[x], widths=0.055, patch_artist=True, showfliers=False, whis=(5, 95), manage_ticks=False, medianprops={"color": "#111111", "linewidth": 1.0}, whiskerprops={"color": "#222222", "linewidth": 0.65}, capprops={"color": "#222222", "linewidth": 0.65}, boxprops={"facecolor": face, "edgecolor": "#222222", "linewidth": 0.55})
                bp["boxes"][0].set_hatch(hatch)
                bp["boxes"][0].set_alpha(0.85)
    axb.set_yscale("log")
    axb.set_xlim(-0.55, len(STAGE_ORDER) - 0.45)
    axb.set_xticks(x_centers)
    axb.set_xticklabels([STAGE_LABELS[s] for s in STAGE_ORDER])
    axb.set_ylabel("Max force (eV/A)")
    axb.set_xlabel("Candidate stage")
    axb.axhline(100, color="#333333", linewidth=1.0, linestyle=(0, (4, 2)))
    panel_title(axb, "(b)", "Model force distributions by candidate stage")
    family_handles = [Patch(facecolor=FAMILY_COLORS[f], edgecolor="#222222", label=f) for f in FAMILY_ORDER]
    arm_handles = [Patch(facecolor="white", edgecolor="#222222", hatch=ARM_HATCHES[h], label=label) for h, label in [("arm 1", "arm 1"), ("arm 2", "arm 2"), ("arm 3", "arm 3")]]
    gate_handle = Line2D([0], [0], color="#333333", linestyle=(0, (4, 2)), linewidth=1.0, label="100 eV/A")
    legend1 = axb.legend(handles=family_handles, loc="upper left", bbox_to_anchor=(0, -0.17), ncol=2, fontsize=8.2, handlelength=1.2, columnspacing=0.9)
    axb.add_artist(legend1)
    axb.legend(handles=arm_handles + [gate_handle], loc="upper right", bbox_to_anchor=(1, -0.17), ncol=2, fontsize=8.2, handlelength=1.4, columnspacing=0.9)
    apply_axes(axb)

    axc = fig.add_subplot(gs[1, 0])
    ref = pd.read_csv(REFERENCE)
    ref_styles = {"train": ("#111111", "o", "Train"), "valid": ("#666666", "s", "Valid"), "test": ("#AAAAAA", "^", "Test"), "neb": ("#7A3E9D", "D", "NEB reference")}
    ref_handles = []
    for name in ["train", "valid", "test", "neb"]:
        sub = ref[ref["reference"].eq(name)]
        color, marker, label = ref_styles[name]
        axc.scatter(sub["pca1"], sub["pca2"], s=14 if name != "neb" else 25, c=color, marker=marker, alpha=0.75, linewidths=0.3, edgecolors="white")
        ref_handles.append(Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor="white", markersize=5.5, label=label))
    for stage in STAGE_ORDER:
        sub = candidates[candidates["candidate_group"].eq(stage)]
        axc.scatter(sub["pca1"], sub["pca2"], s=14 if len(sub) > 1 else 36, c=STAGE_COLORS[stage], marker="o" if len(sub) > 1 else "D", alpha=0.78, linewidths=0.25, edgecolors="white")
    stage_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=STAGE_COLORS[s], markeredgecolor="white", markersize=5.5, label=STAGE_LABELS[s].replace("\n", " ")) for s in STAGE_ORDER]
    axc.set_xlabel("PCA 1")
    axc.set_ylabel("PCA 2")
    panel_title(axc, "(c)", "Candidate and reference embedding")
    inset = axc.inset_axes([0.57, 0.12, 0.38, 0.40])
    non_neb = ref[ref["reference"].ne("neb")]
    for name in ["train", "valid", "test"]:
        sub = ref[ref["reference"].eq(name)]
        color, marker, _ = ref_styles[name]
        inset.scatter(sub["pca1"], sub["pca2"], s=7, c=color, marker=marker, alpha=0.55, linewidths=0.2, edgecolors="white")
    for stage in STAGE_ORDER:
        sub = candidates[candidates["candidate_group"].eq(stage)]
        inset.scatter(sub["pca1"], sub["pca2"], s=6 if len(sub) > 1 else 20, c=STAGE_COLORS[stage], marker="o" if len(sub) > 1 else "D", alpha=0.7, linewidths=0.15, edgecolors="white")
    x = np.concatenate([non_neb["pca1"].to_numpy(), candidates["pca1"].to_numpy()])
    y = np.concatenate([non_neb["pca2"].to_numpy(), candidates["pca2"].to_numpy()])
    inset.set_xlim(float(x.min()) - 0.4, float(x.max()) + 0.4)
    inset.set_ylim(float(y.min()) - 0.4, float(y.max()) + 0.4)
    inset.tick_params(labelsize=6.8, length=2)
    inset.set_title("Near-origin detail", fontsize=8.2, fontweight="bold", pad=3)
    apply_axes(inset)
    axc.legend(handles=ref_handles + stage_handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7.8, handlelength=1.1, columnspacing=0.8)
    apply_axes(axc)

    axd = fig.add_subplot(gs[1, 1])
    plot = candidates[["candidate_id", "candidate_group", "distance_to_train"]].merge(spread[["candidate_id", "max_force_range_eV_A"]], on="candidate_id", how="left", validate="one_to_one")
    for stage in STAGE_ORDER:
        sub = plot[plot["candidate_group"].eq(stage)]
        axd.scatter(sub["distance_to_train"], sub["max_force_range_eV_A"], s=20 if len(sub) > 1 else 42, marker="o" if len(sub) > 1 else "D", color=STAGE_COLORS[stage], edgecolors="white", linewidths=0.35, alpha=0.8, label=STAGE_LABELS[stage].replace("\n", " "))
    axd.set_xscale("log")
    axd.set_yscale("log")
    axd.set_xlabel("Distance to training set (feature units)")
    axd.set_ylabel("Across-model max-force range (eV/A)")
    panel_title(axd, "(d)", "Distribution shift versus model disagreement")
    axd.axhline(100, color="#333333", linewidth=1.0, linestyle=(0, (4, 2)))
    axd.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=3, fontsize=7.8, handlelength=1.1, columnspacing=0.8)
    apply_axes(axd)
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    style_info = atlas_rc()
    results, candidates, spread, grouping, audit = load_data()

    outputs = []
    for name, builder in [
        ("fig6_panel_a_heatmap", lambda: panel_heatmap(results, grouping)),
        ("fig6_panel_b_stage_distributions", lambda: panel_stage_box(results, grouping)),
        ("fig6_panel_c_embedding_inset", lambda: panel_embedding(candidates)),
        ("fig6_panel_d_novelty_disagreement", lambda: panel_novelty(candidates, spread)),
    ]:
        fig, _ = builder()
        outputs.extend(save_figure(fig, name))
    composite_fig = composite(results, candidates, spread, grouping)
    outputs.extend(save_figure(composite_fig, "fig6_full_campaign_family_grouped_v2"))

    source_hashes = {path.name: sha256(path) for path in sorted(SOURCE.iterdir()) if path.is_file()}
    output_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in outputs}
    manifest = {
        "artifact": "Figure 6 full-campaign four-panel update",
        "status": "RECONSTRUCTION",
        "claim_scope": "639 candidate structures x 12 MLIP checkpoints; static energy/force evaluation",
        "not_claimed": [
            "not a 1 ns MD campaign",
            "not a strict 4-family x 3 exchangeable-seed design",
            "not a DFT correctness or causal attribution result",
        ],
        "source_provenance": {
            "repository": "https://github.com/yicao-elina/MigrationBench",
            "pr": "https://github.com/yicao-elina/MigrationBench/pull/11",
            "commit": "476ca16",
            "campaign_directory": "analysis/multimodel_639_screen_20261014",
            "source_note": "2026-10-14 - Twelve-Model 639-Candidate Static Screen and Distribution-Shift Diagnostic",
        },
        "data_audit": audit,
        "model_order": MODEL_ORDER,
        "family_order": FAMILY_ORDER,
        "family_colors": FAMILY_COLORS,
        "stage_order": STAGE_ORDER,
        "style": {
            "contract": "scientific-plot-atlas",
            "runtime": "local-contract fallback; recovered Plot Atlas package was not hydrated",
            **style_info,
            "dpi_png": 350,
            "editable_exports": ["pdf", "svg"],
            "panel_c_inset": "near-origin candidate/reference detail; NEB retained in main axes",
            "panel_b_singleton": "stable terminal rendered as a diamond point",
        },
        "source_sha256": source_hashes,
        "outputs_sha256": output_hashes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "matplotlib": mpl.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    manifest_path = ROOT / "figure_manifest_family_grouped_v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "outputs": [str(p) for p in outputs], "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
