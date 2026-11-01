#!/usr/bin/env python3
"""Build the reference-style grouped temperature layout for Figure 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import build_figure02_seed_boxclusters_v9 as v9
import build_figure02_temperature_panels_and_overview_v11 as v11


WORKTREE = Path("/Users/alina/.codex/worktrees/canva-figure-provenance")
OUT_DEFAULT = WORKTREE / (
    "docs/reproducibility/canva_figure_provenance/"
    "R03_typography18pt/figures/figure02_temperature_reference_layout_v12"
)

TEMP_FAMILIES = v9.MULTIT_FAMILIES
TEMP_CATEGORIES = v9.TEMP_CATEGORIES
TEMP_CENTERS = {category: index * 3.55 for index, category in enumerate(TEMP_CATEGORIES)}
TEMP_FAMILY_OFFSETS = {"Scratch-MultiT": -0.58, "FT-MultiT": 0.58}
SEED_OFFSETS = np.asarray([-0.105, 0.0, 0.105])
TEMP_BOX_WIDTH = 0.085
GROUP_HALF_WIDTH = 1.02
BRACKET_STEP = 0.60


def _records() -> list[dict[str, object]]:
    v9.base.FRAME_ROOT = v9.FRAME_ROOT
    return v9._apply_atom_based_cr_classes(v9.base.read_frame_records())


def _physical_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if str(record["trajectory"]) != "isolated_atoms_SG15"]


def _reference_axis(ax) -> None:
    v9._style_axis(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(2.2)
    ax.spines["left"].set_color("#111111")
    ax.tick_params(axis="both", which="major", direction="out", length=7, width=1.8, labelsize=19)
    ax.tick_params(axis="both", which="minor", direction="out", length=3.5, width=1.2)
    ax.tick_params(axis="x", bottom=True, top=False, length=6, width=1.8)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontsize(19)
        label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(21)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(21)
    ax.yaxis.label.set_fontweight("bold")


def _temperature_raw(records: list[dict[str, object]], category: str, metric: str) -> dict[tuple[str, str], np.ndarray]:
    return {
        (family, seed): v9._raw_for(
            records,
            family=family,
            seed=seed,
            group_key="temperature",
            category=category,
            metric=metric,
        )
        for family in TEMP_FAMILIES
        for seed in v9.SEEDS
    }


def _draw_temperature_group_headers(ax, low: float, categories: list[str], *, show_top_labels: bool) -> None:
    for category in categories:
        center = TEMP_CENTERS[category]
        start = center - GROUP_HALF_WIDTH
        end = center + GROUP_HALF_WIDTH
        if show_top_labels:
            ax.plot(
                [start, end],
                [1.025, 1.025],
                transform=ax.get_xaxis_transform(),
                color="#111111",
                linewidth=4.2,
                solid_capstyle="butt",
                clip_on=False,
                zorder=10,
            )
            ax.text(
                center,
                1.055,
                v9.TEMP_LABELS[category],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=21,
                fontweight="bold",
                color="#111111",
                clip_on=False,
                zorder=10,
            )
        ax.plot(
            [start, end],
            [low, low],
            color="#111111",
            linewidth=2.2,
            solid_capstyle="butt",
            clip_on=False,
            zorder=9,
        )


def _draw_temperature_strip(
    ax,
    records: list[dict[str, object]],
    metric: str,
    row_index: int,
    *,
    show_xlabels: bool,
    show_top_labels: bool,
) -> dict[str, object]:
    categories = TEMP_CATEGORIES
    aggregate = v9.merged_aggregate(records, "temperature", categories)
    all_raw: dict[tuple[str, str, str], np.ndarray] = {}
    raw_high_by_category: dict[str, float] = {}
    for category in categories:
        raw = _temperature_raw(records, category, metric)
        for key, values in raw.items():
            all_raw[(category, key[0], key[1])] = values
        raw_high_by_category[category] = float(np.log10(np.nanmax(np.concatenate([values for values in raw.values() if values.size]))))
    low, _, high = v9._collect_limits(all_raw, aggregate[metric], categories, TEMP_FAMILIES)
    comparison_manifest: dict[str, object] = {}
    for category in categories:
        raw = _temperature_raw(records, category, metric)
        center = TEMP_CENTERS[category]
        for family in TEMP_FAMILIES:
            family_center = center + TEMP_FAMILY_OFFSETS[family]
            for seed_index, seed in enumerate(v9.SEEDS):
                v9._box_and_points(
                    ax,
                    raw[(family, seed)],
                    family_center + SEED_OFFSETS[seed_index],
                    color=v9.COLORS[family],
                    seed=seed,
                    family=family,
                    category=category,
                    metric=metric,
                    box_width=TEMP_BOX_WIDTH,
                )

        values_a = [aggregate[metric].get(category, {}).get((TEMP_FAMILIES[0], seed), np.nan) for seed in v9.SEEDS]
        values_b = [aggregate[metric].get(category, {}).get((TEMP_FAMILIES[1], seed), np.nan) for seed in v9.SEEDS]
        pvalue = v9._paired_p(values_a, values_b)
        marker = v9._sig_label(pvalue)
        comparison_manifest[category] = {
            "Scratch vs FT · MultiT": {"p_value": pvalue, "marker": marker, "n_seed_pairs": 3}
        }
        v9._draw_bracket(
            ax,
            center + TEMP_FAMILY_OFFSETS[TEMP_FAMILIES[0]],
            center + TEMP_FAMILY_OFFSETS[TEMP_FAMILIES[1]],
            raw_high_by_category[category] + 0.18,
            marker,
        )

    v9._set_log_exponent_axis(ax, (low, high))
    ax.set_xlim(TEMP_CENTERS[TEMP_CATEGORIES[0]] - 1.28, TEMP_CENTERS[TEMP_CATEGORIES[-1]] + 1.28)
    tick_positions = [
        TEMP_CENTERS[category] + TEMP_FAMILY_OFFSETS[family]
        for category in categories
        for family in TEMP_FAMILIES
    ]
    tick_labels = ["Scratch\nMultiT", "FT\nMultiT"] * len(categories)
    ax.set_xticks(tick_positions, tick_labels)
    if not show_xlabels:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylim(low, high)
    ax.set_ylabel("Energy RMSE (meV atom$^{-1}$)" if metric == "energy" else "Force RMSE (meV Å$^{-1}$)")
    if show_xlabels:
        ax.set_xlabel("model", labelpad=18)
    _reference_axis(ax)
    _draw_temperature_group_headers(ax, low, categories, show_top_labels=show_top_labels)
    return comparison_manifest


def _temperature_figure(records: list[dict[str, object]], out_dir: Path) -> tuple[dict[str, str], dict[str, object]]:
    fig, axes = plt.subplots(2, 1, figsize=(20.0, 10.5), sharex=True, squeeze=False)
    comparison = {
        "energy": _draw_temperature_strip(axes[0, 0], records, "energy", 0, show_xlabels=False, show_top_labels=True),
        "force": _draw_temperature_strip(axes[1, 0], records, "force", 1, show_xlabels=True, show_top_labels=False),
    }
    handles = v9._family_handles(TEMP_FAMILIES) + v9._seed_handles()
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, prop={"weight": "bold", "size": 19}, columnspacing=1.45, handletextpad=0.55)
    fig.suptitle("Figure 2 revision | temperature-resolved MultiT RMSE", y=1.065, fontsize=24, fontweight="bold")
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.14, top=0.80, hspace=0.34)
    return v9.base.save(fig, out_dir, "variant04_by_temperature_class_reference_grouped_seed_boxclusters"), comparison


def _overview_figure(records: list[dict[str, object]], out_dir: Path) -> tuple[dict[str, str], dict[str, object]]:
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 12.0), squeeze=False)
    comparison = v11._draw_overview_axes(fig, axes[:, 0], records)
    fig.legend(handles=v9._family_handles(v9.FAMILIES), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, prop={"weight": "bold", "size": 18}, columnspacing=1.25, handletextpad=0.55)
    fig.suptitle("Figure 2 overview | Scratch vs FT across all physical trajectory classes", y=1.035, fontsize=24, fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.12, top=0.86, hspace=0.30)
    return v9.base.save(fig, out_dir, "variant06_scratch_vs_ft_overview_shared_legend"), comparison


def _composite_figure(records: list[dict[str, object]], out_dir: Path) -> dict[str, str]:
    fig = plt.figure(figsize=(34.0, 15.0))
    grid = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.0, 1.0, 1.35, 1.35], wspace=0.32, hspace=0.36)
    temp_axes = np.asarray([
        fig.add_subplot(grid[0, :3]),
        fig.add_subplot(grid[1, :3]),
    ], dtype=object)
    overview_axes = np.asarray([
        fig.add_subplot(grid[0, 3:]),
        fig.add_subplot(grid[1, 3:]),
    ], dtype=object)
    _draw_temperature_strip(temp_axes[0], records, "energy", 0, show_xlabels=False, show_top_labels=True)
    _draw_temperature_strip(temp_axes[1], records, "force", 1, show_xlabels=True, show_top_labels=False)
    v11._draw_overview_axes(fig, overview_axes, records, add_labels=False)
    fig.text(0.285, 0.985, "Temperature-resolved MultiT", ha="center", va="top", fontsize=24, fontweight="bold")
    fig.text(0.755, 0.985, "Overall Scratch vs FT", ha="center", va="top", fontsize=24, fontweight="bold")
    shared_handles = v9._family_handles(v9.FAMILIES) + v9._seed_handles()
    fig.legend(handles=shared_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=7, prop={"weight": "bold", "size": 18}, columnspacing=1.15, handletextpad=0.45)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.84)
    return v9.base.save(fig, out_dir, "variant04_temperature_plus_scratch_ft_shared_legend")


def build(out_dir: Path) -> dict[str, object]:
    v9._house_style()
    v9.BRACKET_STEP = BRACKET_STEP
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _physical_records(_records())
    temperature, temperature_stats = _temperature_figure(records, out_dir)
    overview, overview_stats = _overview_figure(records, out_dir)
    composite = _composite_figure(records, out_dir)
    manifest = {
        "status": "RECONSTRUCTION",
        "previous_version": "a863abb",
        "purpose": "reference-style grouped temperature strips with broken x-axis segments, thick temperature headers, and a shared legend in the composite",
        "data": {
            "source_mode": "readable_packaged_per_frame_csv",
            "frame_root": str(v9.FRAME_ROOT),
            "splits_merged": v9.base.SPLITS,
            "isolated_atoms_excluded": True,
            "temperature_families": TEMP_FAMILIES,
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
    manifest_path = out_dir / "figure02_temperature_reference_layout_v12_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_temperature_reference_layout_v12_caption.md"
    caption_path.write_text(
        "# Figure 2 v12 reference-style temperature layout\n\n"
        "The temperature view uses a grouped reference-style layout rather than six boxed panels. Energy and force each occupy one horizontal strip with a shared within-row y scale across 300 K, 600 K, and 1200 K. Each temperature group is separated by whitespace, identified by a thick upper rule, and retains all three seed-resolved boxplots and all frame-level points. The composite uses one shared family/seed legend for the temperature strip and Scratch-vs-FT overview. Train and held-out test records are merged; isolated-atom reference records are excluded.\n"
    )
    return {"manifest": str(manifest_path), "caption": str(caption_path), "outputs": manifest["outputs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
