#!/usr/bin/env python3
"""Build square temperature panels and a Scratch-vs-FT overview for Figure 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import _figure02_latest_seed as v9


WORKTREE = Path("/Users/alina/.codex/worktrees/canva-figure-provenance")
OUT_DEFAULT = WORKTREE / (
    "docs/reproducibility/canva_figure_provenance/"
    "R03_typography18pt/figures/figure02_temperature_panels_v11"
)
TEMP_FAMILIES = v9.MULTIT_FAMILIES
TEMP_CATEGORIES = v9.TEMP_CATEGORIES
SEED_OFFSETS = np.asarray([-0.105, 0.0, 0.105])
TEMP_FAMILY_OFFSETS = {"Scratch-MultiT": -0.30, "FT-MultiT": 0.30}
TEMP_BOX_WIDTH = 0.085
OVERVIEW_FAMILY_X = {
    "Scratch-600K": -0.20,
    "Scratch-MultiT": 0.20,
    "FT-600K": 1.20,
    "FT-MultiT": 1.60,
}
OVERVIEW_BOX_WIDTH = 0.24
BRACKET_STEP = 0.60


def _records() -> list[dict[str, object]]:
    v9.base.FRAME_ROOT = v9.FRAME_ROOT
    return v9._apply_atom_based_cr_classes(v9.base.read_frame_records())


def _physical_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [record for record in records if str(record["trajectory"]) != "isolated_atoms_SG15"]


def _square_axis(ax) -> None:
    v9._style_axis(ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.2)
        spine.set_color("#111111")
    ax.tick_params(axis="both", which="major", direction="out", length=7, width=1.8, labelsize=19)
    ax.tick_params(axis="both", which="minor", direction="out", length=3.5, width=1.2)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontsize(19)
        label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(21)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontsize(21)
    ax.yaxis.label.set_fontweight("bold")
    ax.title.set_fontsize(21)
    ax.title.set_fontweight("bold")


def _temperature_raw(records: list[dict[str, object]], category: str, metric: str) -> dict[tuple[str, str, str], np.ndarray]:
    return {
        (category, family, seed): v9._raw_for(
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


def _draw_temperature_grid(fig, axes, records: list[dict[str, object]], *, add_labels: bool = True) -> dict[str, object]:
    categories = TEMP_CATEGORIES
    aggregate = v9.merged_aggregate(records, "temperature", categories)
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    comparison_manifest: dict[str, object] = {}
    for row_index, (metric, ylabel) in enumerate(metrics):
        all_raw: dict[tuple[str, str, str], np.ndarray] = {}
        raw_high_by_category: dict[str, float] = {}
        for category in categories:
            raw = _temperature_raw(records, category, metric)
            all_raw.update(raw)
            raw_high_by_category[category] = float(np.log10(np.nanmax(np.concatenate([values for values in raw.values() if values.size]))))
        low, _, high = v9._collect_limits(all_raw, aggregate[metric], categories, TEMP_FAMILIES)
        comparison_manifest[metric] = {}
        for col_index, category in enumerate(categories):
            ax = axes[row_index, col_index]
            raw = _temperature_raw(records, category, metric)
            for family in TEMP_FAMILIES:
                family_center = TEMP_FAMILY_OFFSETS[family]
                for seed_index, seed in enumerate(v9.SEEDS):
                    v9._box_and_points(
                        ax,
                        raw[(category, family, seed)],
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
            comparison_manifest[metric][category] = {"Scratch vs FT · MultiT": {"p_value": pvalue, "marker": marker, "n_seed_pairs": 3}}
            v9._draw_bracket(ax, TEMP_FAMILY_OFFSETS[TEMP_FAMILIES[0]], TEMP_FAMILY_OFFSETS[TEMP_FAMILIES[1]], raw_high_by_category[category] + 0.18, marker)

            v9._set_log_exponent_axis(ax, (low, high))
            ax.set_xlim(-0.62, 0.62)
            ax.set_ylim(low, high)
            ax.set_title(v9.TEMP_LABELS[category], pad=12)
            ax.set_xticks([TEMP_FAMILY_OFFSETS[family] for family in TEMP_FAMILIES], ["Scratch\nMultiT", "FT\nMultiT"])
            if row_index == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("model", labelpad=16)
            if col_index == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
                ax.set_ylabel("")
            _square_axis(ax)
    return comparison_manifest


def _seed_aggregate(records: list[dict[str, object]], family: str, seed: str, metric: str) -> float:
    values = [record for record in records if str(record["family"]) == family and str(record["seed"]) == seed]
    if metric == "energy":
        errors = np.asarray([float(record["energy_error_meV"]) for record in values], dtype=float)
        return float(np.sqrt(np.mean(errors**2)))
    force = np.asarray([float(record["force_rmse_meV_A"]) for record in values], dtype=float)
    weights = np.asarray([float(record["n_atoms"]) for record in values], dtype=float)
    return float(np.sqrt(np.sum(weights * force**2) / np.sum(weights)))


def _overview_raw(records: list[dict[str, object]], family: str, metric: str) -> np.ndarray:
    values = [v9._metric_value(record, metric) for record in records if str(record["family"]) == family]
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def _draw_overview_axes(fig, axes, records: list[dict[str, object]], *, add_labels: bool = True) -> dict[str, object]:
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    comparisons = v9._comparison_pairs(v9.FAMILIES)
    comparison_manifest: dict[str, object] = {}
    for row_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[row_index]
        raw_by_family = {family: _overview_raw(records, family, metric) for family in v9.FAMILIES}
        seed_values = {
            family: [_seed_aggregate(records, family, seed, metric) for seed in v9.SEEDS]
            for family in v9.FAMILIES
        }
        combined = np.concatenate([np.log10(raw) for raw in raw_by_family.values()])
        combined = np.concatenate([combined] + [np.log10(np.asarray(values)) for values in seed_values.values()])
        low = float(np.floor(np.nanmin(combined) - 0.18))
        raw_high = float(np.nanmax(combined))
        high = raw_high + 0.45 + BRACKET_STEP * len(comparisons)
        for family in v9.FAMILIES:
            raw = raw_by_family[family]
            logged = np.log10(raw)
            xpos = OVERVIEW_FAMILY_X[family]
            rng = np.random.default_rng(v9.base._stable_seed("overview", metric, family))
            jitter = rng.uniform(-OVERVIEW_BOX_WIDTH * 0.42, OVERVIEW_BOX_WIDTH * 0.42, raw.size)
            ax.scatter(np.full(raw.size, xpos) + jitter, logged, s=5.0, color=v9.COLORS[family], alpha=0.22, edgecolors="none", zorder=1)
            box = ax.boxplot(
                [logged],
                positions=[xpos],
                widths=OVERVIEW_BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                whis=1.5,
                boxprops={"facecolor": v9.COLORS[family], "edgecolor": "#111111", "linewidth": 1.7, "alpha": 0.82},
                medianprops={"color": "#111111", "linewidth": 2.2},
                whiskerprops={"color": "#111111", "linewidth": 1.7},
                capprops={"color": "#111111", "linewidth": 1.7},
            )
            for patch in box["boxes"]:
                patch.set_zorder(4)
            center, lower, upper = v9._seed_summary(seed_values[family])
            ax.errorbar([xpos], [center], yerr=[[center - lower], [upper - center]], fmt="none", ecolor="#111111", elinewidth=1.5, capsize=4.0, capthick=1.5, zorder=6)
            ax.scatter([xpos], [center], s=34, color="#111111", edgecolors="white", linewidths=0.5, zorder=7)

        comparison_manifest[metric] = {}
        family_index = {family: i for i, family in enumerate(v9.FAMILIES)}
        for level, (family_a, family_b, description) in enumerate(comparisons):
            pvalue = v9._paired_p(seed_values[family_a], seed_values[family_b])
            comparison_manifest[metric][description] = {"p_value": pvalue, "marker": v9._sig_label(pvalue), "n_seed_pairs": 3}
            if np.isfinite(pvalue):
                v9._draw_bracket(ax, OVERVIEW_FAMILY_X[family_a], OVERVIEW_FAMILY_X[family_b], raw_high + 0.18 + level * BRACKET_STEP, v9._sig_label(pvalue))

        v9._set_log_exponent_axis(ax, (low, high))
        ax.set_xlim(-0.58, 1.98)
        ax.set_xticks([0.0, 1.4], ["Scratch", "FT"])
        if row_index == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("model lineage · 600K and MultiT shown by color", labelpad=20)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{chr(65 + row_index)}  {ylabel.split(' RMSE')[0]}", loc="left", pad=12)
        _square_axis(ax)
    return comparison_manifest


def _temperature_figure(records: list[dict[str, object]], out_dir: Path) -> tuple[dict[str, str], dict[str, object]]:
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 11.5), sharey="row", squeeze=False)
    comparison = _draw_temperature_grid(fig, axes, records)
    fig.legend(handles=v9._family_handles(TEMP_FAMILIES), frameon=False, loc="upper center", bbox_to_anchor=(0.34, 0.995), ncol=2, prop={"weight": "bold", "size": 20})
    fig.legend(handles=v9._seed_handles(), frameon=False, loc="upper center", bbox_to_anchor=(0.68, 0.995), ncol=3, prop={"weight": "bold", "size": 20})
    fig.suptitle("Figure 2 revision | temperature-resolved MultiT RMSE", y=1.055, fontsize=24, fontweight="bold")
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.13, top=0.82, wspace=0.12, hspace=0.28)
    return v9.base.save(fig, out_dir, "variant04_by_temperature_class_square_panels_seed_boxclusters"), comparison


def _overview_figure(records: list[dict[str, object]], out_dir: Path) -> tuple[dict[str, str], dict[str, object]]:
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 12.0), squeeze=False)
    comparison = _draw_overview_axes(fig, axes[:, 0], records)
    fig.legend(handles=v9._family_handles(v9.FAMILIES), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, prop={"weight": "bold", "size": 18})
    fig.suptitle("Figure 2 overview | Scratch vs FT across all physical trajectory classes", y=1.035, fontsize=24, fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.12, top=0.86, hspace=0.30)
    return v9.base.save(fig, out_dir, "variant06_scratch_vs_ft_overview_energy_force"), comparison


def _composite_figure(records: list[dict[str, object]], out_dir: Path) -> dict[str, str]:
    fig = plt.figure(figsize=(34.0, 15.0))
    grid = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.0, 1.0, 1.35, 1.35], wspace=0.24, hspace=0.36)
    temp_axes = np.empty((2, 3), dtype=object)
    shared = [None, None]
    for row in range(2):
        for col in range(3):
            temp_axes[row, col] = fig.add_subplot(grid[row, col], sharey=shared[row]) if shared[row] is not None else fig.add_subplot(grid[row, col])
            shared[row] = temp_axes[row, col]
    overview_axes = np.asarray([
        fig.add_subplot(grid[0, 3:]),
        fig.add_subplot(grid[1, 3:]),
    ], dtype=object)
    _draw_temperature_grid(fig, temp_axes, records, add_labels=False)
    _draw_overview_axes(fig, overview_axes, records, add_labels=False)
    fig.text(0.30, 0.985, "Temperature-resolved MultiT", ha="center", va="top", fontsize=24, fontweight="bold")
    fig.text(0.75, 0.985, "Overall Scratch vs FT", ha="center", va="top", fontsize=24, fontweight="bold")
    fig.legend(handles=v9._family_handles(TEMP_FAMILIES), frameon=False, loc="upper center", bbox_to_anchor=(0.26, 0.95), ncol=2, prop={"weight": "bold", "size": 18})
    fig.legend(handles=v9._seed_handles(), frameon=False, loc="upper center", bbox_to_anchor=(0.49, 0.95), ncol=3, prop={"weight": "bold", "size": 18})
    fig.legend(handles=v9._family_handles(v9.FAMILIES), frameon=False, loc="upper center", bbox_to_anchor=(0.76, 0.95), ncol=2, prop={"weight": "bold", "size": 18})
    fig.savefig(out_dir / "variant04_temperature_plus_scratch_ft_overview.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "variant04_temperature_plus_scratch_ft_overview.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "variant04_temperature_plus_scratch_ft_overview.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "png": str(out_dir / "variant04_temperature_plus_scratch_ft_overview.png"),
        "pdf": str(out_dir / "variant04_temperature_plus_scratch_ft_overview.pdf"),
        "svg": str(out_dir / "variant04_temperature_plus_scratch_ft_overview.svg"),
    }


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
        "checkpoint_before_revision": "56b54a0",
        "purpose": "square framed temperature panels with shared row y scales plus an overall Scratch-vs-FT energy/force overview",
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
    manifest_path = out_dir / "figure02_temperature_panels_and_overview_v11_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_temperature_panels_and_overview_v11_caption.md"
    caption_path.write_text(
        "# Figure 2 v11 temperature panels and Scratch-vs-FT overview\n\n"
        "The temperature view is redrawn as six individually framed square panels: energy and force share y scales across the three temperature columns, while each panel retains all seed-resolved frame-level points. The overview groups the four exact R04 model families under Scratch and FT x-axis labels, with 600K/MultiT retained in the legend; all physical trajectory classes are pooled after merging train and held-out test records. Significance tests remain paired at the seed level (n=3) and are exploratory.\n"
    )
    return {"manifest": str(manifest_path), "caption": str(caption_path), "outputs": manifest["outputs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
