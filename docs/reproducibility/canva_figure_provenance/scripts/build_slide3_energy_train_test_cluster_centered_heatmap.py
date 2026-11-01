#!/usr/bin/env python3
"""Complete train/test energy parity after shared cluster-centered alignment."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_slide3_nature_density as base  # noqa: E402

OUT_ROOT = base.OUT_ROOT
FONT = base.FONT
MIN_TEXT = base.MIN_TEXT
TITLE_SIZE = base.TITLE_SIZE
PANEL_TITLE = base.PANEL_TITLE
AXIS_LABEL = base.AXIS_LABEL
TICK_SIZE = base.TICK_SIZE
ANNOTATION = base.ANNOTATION
LEGEND_SIZE = base.LEGEND_SIZE
COLORBAR_SIZE = base.COLORBAR_SIZE
COLORS = base.COLORS
FAMILIES = base.FAMILIES
SEEDS = base.SEEDS
SPLITS = ("train", "test")

FAMILY_PREFIX = {
    "Scratch-600K": "scratch_600k",
    "FT-600K": "ft_600k",
    "Scratch-MultiT": "scratch_multit",
    "FT-MultiT": "ft_multit",
}


def style_axis(ax: plt.Axes) -> None:
    base.style_axis(ax)
    ax.tick_params(axis="both", which="major", direction="out", length=7, width=1.35, pad=6, labelsize=TICK_SIZE)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontfamily(FONT)
        tick.set_fontsize(TICK_SIZE)
        tick.set_fontweight("bold")


def load_family_split(family: str, split: str) -> dict[str, np.ndarray | list[str]]:
    refs = []
    preds = []
    labels: list[str] = []
    prefix = FAMILY_PREFIX[family]
    for seed in SEEDS:
        stem = f"{prefix}_seed{seed}_{split}"
        npz_path = base.RAW_ROOT / f"{stem}__parity.npz"
        csv_path = base.RAW_ROOT / f"{stem}__per_frame.csv"
        with np.load(npz_path, allow_pickle=False) as data:
            ref = np.asarray(data["energy_ref_eV_per_atom"], dtype=float)
            pred = np.asarray(data["energy_pred_eV_per_atom"], dtype=float)
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != ref.size:
            raise RuntimeError(f"CSV/NPZ frame mismatch: {npz_path.name}")
        csv_ref = np.asarray([float(row["energy_ref_eV_per_atom"]) for row in rows], dtype=float)
        if not np.allclose(csv_ref, ref, rtol=0.0, atol=2e-4):
            raise RuntimeError(f"CSV/NPZ reference-energy mismatch: {npz_path.name}")
        for row, value in zip(rows, ref, strict=True):
            trajectory = row["trajectory"]
            if trajectory == "isolated_atoms_SG15":
                label = f"isolated_atom_{value:.6f}"
            else:
                label = trajectory
            labels.append(label)
        refs.append(ref)
        preds.append(pred)
    return {"ref": np.concatenate(refs), "pred": np.concatenate(preds), "labels": labels}


def shared_cluster_center(
    train: dict[str, np.ndarray | list[str]],
    test: dict[str, np.ndarray | list[str]],
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, int], list[str]]:
    train_ref = np.asarray(train["ref"], dtype=float)
    test_ref = np.asarray(test["ref"], dtype=float)
    train_pred = np.asarray(train["pred"], dtype=float)
    test_pred = np.asarray(test["pred"], dtype=float)
    train_labels = list(train["labels"])
    test_labels = list(test["labels"])
    all_ref = np.concatenate((train_ref, test_ref))
    all_labels = train_labels + test_labels
    members: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(all_labels):
        members[label].append(index)
    centers = {label: float(np.mean(all_ref[indices])) for label, indices in members.items()}
    sizes = {label: len(indices) for label, indices in members.items()}
    train_offsets = np.asarray([centers[label] for label in train_labels], dtype=float)
    test_offsets = np.asarray([centers[label] for label in test_labels], dtype=float)
    normalized = {
        "train_x": train_ref - train_offsets,
        "train_y": train_pred - train_offsets,
        "test_x": test_ref - test_offsets,
        "test_y": test_pred - test_offsets,
    }
    train_only = sorted(set(train_labels) - set(test_labels))
    test_only = sorted(set(test_labels) - set(train_labels))
    return normalized, centers, sizes, test_only


def actual_hexbin_counts(x: np.ndarray, y: np.ndarray, extent: tuple[float, float, float, float], gridsize: int) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    collection = ax.hexbin(x, y, gridsize=gridsize, extent=extent, mincnt=1)
    counts = np.asarray(collection.get_array(), dtype=float).copy()
    plt.close(fig)
    return counts[np.isfinite(counts) & (counts > 0)]


def add_vector_colorbar(fig: plt.Figure, reference_ax: plt.Axes, norm: LogNorm) -> None:
    bbox = reference_ax.get_position()
    cax = fig.add_axes([bbox.x1 + 0.014, bbox.y0, 0.018, bbox.height])
    cax.set_yscale("log")
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(norm.vmin, norm.vmax)
    cmap = mpl.colormaps["magma"]
    edges = np.geomspace(norm.vmin, norm.vmax, 72)
    for lower, upper in zip(edges[:-1], edges[1:]):
        cax.add_patch(mpl.patches.Rectangle((0.0, lower), 1.0, upper - lower, facecolor=cmap(norm(np.sqrt(lower * upper))), edgecolor="none", transform=cax.transData))
    cax.xaxis.set_visible(False)
    cax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    cax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    cax.yaxis.tick_right()
    cax.tick_params(axis="y", which="major", labelsize=COLORBAR_SIZE, width=1.0, length=5, pad=5)
    cax.set_ylabel("Count per hexbin", fontsize=COLORBAR_SIZE, fontweight="bold", labelpad=13, rotation=270)
    cax.yaxis.set_label_position("right")
    for tick in cax.get_yticklabels():
        tick.set_fontfamily(FONT)
        tick.set_fontsize(COLORBAR_SIZE)
        tick.set_fontweight("bold")
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
        spine.set_color("#111111")


def render(panels: dict[str, dict[str, np.ndarray]], cluster_counts: dict[str, int]) -> tuple[dict[str, str], tuple[float, float, float, float], LogNorm]:
    merged = []
    for panel in panels.values():
        merged.extend((panel["train_x"], panel["train_y"], panel["test_x"], panel["test_y"]))
    all_values = np.concatenate(merged)
    lo = float(np.min(all_values))
    hi = float(np.max(all_values))
    span = hi - lo
    pad = 0.08 * span if span > 0 else 0.02
    extent = (lo - pad, hi + pad, lo - pad, hi + pad)
    max_count = 1.0
    for panel in panels.values():
        x = np.concatenate((panel["train_x"], panel["test_x"]))
        y = np.concatenate((panel["train_y"], panel["test_y"]))
        counts = actual_hexbin_counts(x, y, extent, gridsize=48)
        if counts.size:
            max_count = max(max_count, float(np.max(counts)))
    norm = LogNorm(vmin=1.0, vmax=max(2.0, max_count))

    fig, axes = plt.subplots(1, 4, figsize=(25.5, 9.0), squeeze=False)
    for col, family in enumerate(FAMILIES):
        ax = axes[0, col]
        panel = panels[family]
        train_x, train_y = panel["train_x"], panel["train_y"]
        test_x, test_y = panel["test_x"], panel["test_y"]
        all_x = np.concatenate((train_x, test_x))
        all_y = np.concatenate((train_y, test_y))
        ax.hexbin(all_x, all_y, gridsize=48, extent=extent, mincnt=1, cmap="magma", norm=norm, linewidths=0.0, edgecolors="none", rasterized=False)
        ax.scatter(train_x, train_y, marker="o", s=8.0, facecolors="#9a9a9a", edgecolors="none", alpha=0.28, zorder=4)
        ax.scatter(test_x, test_y, marker="o", s=9.0, facecolors="#111111", edgecolors="none", alpha=0.78, zorder=5)
        ax.plot([extent[0], extent[1]], [extent[0], extent[1]], linestyle=(0, (6, 4)), color="#cf2e2e", linewidth=2.2, zorder=3)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_title(f"{chr(65 + col)}  {family}", loc="left", color=COLORS[family], fontsize=PANEL_TITLE, fontweight="bold", pad=12)
        annotation = (
            f"train: ρ={base.spearman(train_x, train_y):.2f}, r={base.pearson(train_x, train_y):.2f}\n"
            f"test:  ρ={base.spearman(test_x, test_y):.2f}, r={base.pearson(test_x, test_y):.2f}\n"
            f"n_train={train_x.size:,}, n_test={test_x.size:,}\n"
            f"k = {cluster_counts[family]} clusters"
        )
        ax.text(0.045, 0.95, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=ANNOTATION, linespacing=1.12, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.80, "pad": 3.0})
        style_axis(ax)
        if col > 0:
            ax.tick_params(labelleft=False)

    fig.text(0.5, 0.095, r"DFT $E/N-\langle E/N\rangle_{\mathrm{cluster}}$ (eV atom$^{-1}$)", ha="center", va="center", fontsize=AXIS_LABEL, fontweight="bold")
    fig.text(0.018, 0.50, r"MLIP $E/N-\langle E/N\rangle_{\mathrm{cluster}}$ (eV atom$^{-1}$)", ha="center", va="center", rotation=90, fontsize=AXIS_LABEL, fontweight="bold")
    fig.text(0.5, 0.985, "DFT reference vs MLIP prediction | cluster-centered complete train and held-out test energy", ha="center", va="top", fontsize=TITLE_SIZE, fontweight="bold")
    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#9a9a9a", markeredgecolor="none", alpha=0.28, markersize=8, label="train"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#111111", markeredgecolor="none", alpha=0.78, markersize=8, label="held-out test"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=2, frameon=False, fontsize=LEGEND_SIZE, handletextpad=0.55, columnspacing=1.6)
    fig.subplots_adjust(left=0.075, right=0.925, bottom=0.21, top=0.87, wspace=0.12)
    add_vector_colorbar(fig, axes[0, -1], norm)

    stem = "slide3_energy_train_test_cluster_centered_heatmap"
    paths = {ext: OUT_ROOT / "figures" / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")}
    paths["png"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["png"], dpi=420, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.18, facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    return {key: str(value) for key, value in paths.items()}, extent, norm


def main() -> None:
    base.setup_style()
    panels = {}
    cluster_centers = {}
    cluster_sizes = {}
    test_only_clusters = {}
    cluster_counts = {}
    for family in FAMILIES:
        train = load_family_split(family, "train")
        test = load_family_split(family, "test")
        normalized, centers, sizes, test_only = shared_cluster_center(train, test)
        panels[family] = normalized
        cluster_centers[family] = centers
        cluster_sizes[family] = sizes
        test_only_clusters[family] = test_only
        cluster_counts[family] = len(centers)
    outputs, extent, norm = render(panels, cluster_counts)
    manifest = {
        "data_root": str(base.DATA_ROOT),
        "raw_root": str(base.RAW_ROOT),
        "input_matrix": "24 parity NPZ outputs plus matching per-frame CSV labels: 4 families x 3 seeds x train/test",
        "normalization": "within each family and trajectory cluster, subtract the pooled train+test DFT E/N mean from both train and test DFT/MLIP E/N",
        "cluster_identity": "per-frame CSV trajectory; isolated_atoms_SG15 is split into three singleton clusters by its three distinct absolute reference energies",
        "split_alignment": "shared trajectory clusters use one common E0; test-only trajectories use a test-derived E0 and are explicitly recorded",
        "complete_data": "all finite train and held-out test frames are used in the heatmap and point overlays; no point sampling or clipping",
        "heatmap": "pooled train+test magma hexbin count density with shared LogNorm across family panels",
        "overlay": "train points are gray with alpha 0.28; held-out test points are black with alpha 0.78",
        "cluster_counts": cluster_counts,
        "test_only_clusters": test_only_clusters,
        "cluster_centers_eV_per_atom": cluster_centers,
        "cluster_sizes": cluster_sizes,
        "limits": list(extent),
        "count_norm": {"vmin": float(norm.vmin), "vmax": float(norm.vmax)},
        "min_text_size_pt": MIN_TEXT,
        "outputs": outputs,
    }
    manifest_path = OUT_ROOT / "slide3_energy_train_test_cluster_centered_heatmap_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"outputs": outputs, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
