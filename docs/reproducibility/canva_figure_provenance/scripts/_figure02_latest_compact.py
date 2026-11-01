#!/usr/bin/env python3
"""Build the compact, square-ready reference layout for Figure 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _figure02_latest_seed as v9
import _figure02_latest_layout as v12


WORKTREE = Path("/Users/alina/.codex/worktrees/canva-figure-provenance")
OUT_DEFAULT = WORKTREE / (
    "docs/reproducibility/canva_figure_provenance/"
    "R03_typography18pt/figures/figure02_temperature_reference_layout_v13"
)

# The v12 layout used a presentation-wide horizontal spacing.  v13 keeps the
# reference grammar but uses a compact gap between adjacent temperature rules.
TEMP_CATEGORIES = v9.TEMP_CATEGORIES
TEMP_CENTERS = {category: index * 2.15 for index, category in enumerate(TEMP_CATEGORIES)}
TEMP_FAMILY_OFFSETS = {"Scratch-MultiT": -0.53, "FT-MultiT": 0.53}
GROUP_HALF_WIDTH = 0.90


def _temperature_figure(records: list[dict[str, object]], out_dir: Path) -> tuple[dict[str, str], dict[str, object]]:
    fig, axes = plt.subplots(2, 1, figsize=(16.0, 9.0), sharex=True, squeeze=False)
    comparison = {
        "energy": v12._draw_temperature_strip(
            axes[0, 0], records, "energy", 0, show_xlabels=False, show_top_labels=True
        ),
        "force": v12._draw_temperature_strip(
            axes[1, 0], records, "force", 1, show_xlabels=True, show_top_labels=False
        ),
    }
    handles = v9._family_handles(v9.MULTIT_FAMILIES) + v9._seed_handles()
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        prop={"weight": "bold", "size": 19},
        columnspacing=1.25,
        handletextpad=0.5,
    )
    fig.suptitle("Figure 2 revision | temperature-resolved MultiT RMSE", y=1.06, fontsize=24, fontweight="bold")
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.145, top=0.80, hspace=0.30)
    return v9.base.save(fig, out_dir, "variant04_by_temperature_class_reference_grouped_seed_boxclusters_compact"), comparison


def _overview_figure(records: list[dict[str, object]], out_dir: Path) -> tuple[dict[str, str], dict[str, object]]:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 10.5), squeeze=False)
    comparison = v12.v11._draw_overview_axes(fig, axes[:, 0], records)
    fig.legend(
        handles=v9._family_handles(v9.FAMILIES),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        prop={"weight": "bold", "size": 18},
        columnspacing=1.1,
        handletextpad=0.5,
    )
    fig.suptitle("Figure 2 overview | Scratch vs FT across all physical trajectory classes", y=1.035, fontsize=23, fontweight="bold")
    fig.subplots_adjust(left=0.155, right=0.98, bottom=0.11, top=0.84, hspace=0.28)
    return v9.base.save(fig, out_dir, "variant06_scratch_vs_ft_overview_shared_legend_square"), comparison


def _composite_figure(records: list[dict[str, object]], out_dir: Path) -> dict[str, str]:
    # A square canvas is intentional: it gives the combined board a stable
    # Canva/manuscript placement while preserving the requested side-by-side view.
    fig = plt.figure(figsize=(16.0, 16.0))
    grid = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.0, 1.0, 1.2, 1.2], wspace=0.24, hspace=0.30)
    temp_axes = np.asarray([
        fig.add_subplot(grid[0, :3]),
        fig.add_subplot(grid[1, :3]),
    ], dtype=object)
    overview_axes = np.asarray([
        fig.add_subplot(grid[0, 3:]),
        fig.add_subplot(grid[1, 3:]),
    ], dtype=object)
    v12._draw_temperature_strip(temp_axes[0], records, "energy", 0, show_xlabels=False, show_top_labels=True)
    v12._draw_temperature_strip(temp_axes[1], records, "force", 1, show_xlabels=True, show_top_labels=False)
    v12.v11._draw_overview_axes(fig, overview_axes, records, add_labels=False)
    overview_axes[1].set_xlabel("model lineage · 600K / MultiT", labelpad=18)
    fig.text(0.285, 0.985, "Temperature-resolved MultiT", ha="center", va="top", fontsize=23, fontweight="bold")
    fig.text(0.75, 0.985, "Overall Scratch vs FT", ha="center", va="top", fontsize=23, fontweight="bold")
    shared_handles = v9._family_handles(v9.FAMILIES) + v9._seed_handles()
    fig.legend(
        handles=shared_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=7,
        prop={"weight": "bold", "size": 18},
        columnspacing=0.95,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.115, top=0.84)
    return v9.base.save(fig, out_dir, "variant04_temperature_plus_scratch_ft_shared_legend_square")


def build(out_dir: Path) -> dict[str, object]:
    v9._house_style()
    v9.BRACKET_STEP = 0.60
    # Override the v12 geometry without changing the already-committed v12 output.
    v12.TEMP_CENTERS = TEMP_CENTERS
    v12.TEMP_FAMILY_OFFSETS = TEMP_FAMILY_OFFSETS
    v12.GROUP_HALF_WIDTH = GROUP_HALF_WIDTH
    out_dir.mkdir(parents=True, exist_ok=True)
    records = v12._physical_records(v12._records())
    temperature, temperature_stats = _temperature_figure(records, out_dir)
    overview, overview_stats = _overview_figure(records, out_dir)
    composite = _composite_figure(records, out_dir)
    manifest = {
        "status": "RECONSTRUCTION",
        "previous_version": "e44cf33",
        "purpose": "compact reference-style temperature groups and square standalone/composite overview",
        "layout": {
            "temperature_centers": TEMP_CENTERS,
            "group_half_width": GROUP_HALF_WIDTH,
            "composite_canvas_inches": [16.0, 16.0],
            "overview_canvas_inches": [10.5, 10.5],
        },
        "data": {
            "source_mode": "readable_packaged_per_frame_csv",
            "frame_root": str(v9.FRAME_ROOT),
            "splits_merged": v9.base.SPLITS,
            "isolated_atoms_excluded": True,
            "temperature_families": v9.MULTIT_FAMILIES,
            "overview_families": v9.FAMILIES,
            "seeds": v9.SEEDS,
            "overview_scope": "all physical trajectory classes pooled within family and seed; train + test merged",
            "significance": {
                "temperature": "Scratch-MultiT vs FT-MultiT",
                "overview": ["Scratch vs FT · 600K", "Scratch vs FT · MultiT", "600K vs MultiT · FT"],
                "test": "paired two-sided t-test on three seed-level aggregates; exploratory n=3; no multiplicity correction",
            },
        },
        "outputs": {"temperature": temperature, "overview": overview, "composite": composite},
        "statistics": {"temperature": temperature_stats, "overview": overview_stats},
    }
    manifest_path = out_dir / "figure02_temperature_reference_layout_v13_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_temperature_reference_layout_v13_caption.md"
    caption_path.write_text(
        "# Figure 2 v13 compact reference-style layout\n\n"
        "This revision preserves the reference-style broken-axis temperature grammar while reducing the inter-temperature gap. The standalone overview and the combined temperature-plus-overview board are square canvases. All frame-level points and three seed-specific boxplots are retained, with train and held-out test records merged.\n"
    )
    return {"manifest": str(manifest_path), "caption": str(caption_path), "outputs": manifest["outputs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
