#!/usr/bin/env python3
"""Build Figure 2 seed-level nested boxplots with complete frame points.

Each model cluster contains one box per seed. Train/test frame records are
merged for category views, and every frame point is plotted inside its exact
model/seed/category box position. The trajectory view uses two trajectory
identities shared by all four model families and all three seeds.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ttest_rel

import build_figure02_clustered_v4 as base
from build_figure02_merged_violin_v7 import merged_aggregate


WORKTREE = Path("/Users/alina/.codex/worktrees/canva-figure-provenance")
PACKAGED_DATA = WORKTREE / "docs/reproducibility/canva_figure_provenance/data/figure02"
FRAME_ROOT = PACKAGED_DATA / "raw_rockfish_parity"
SUMMARY_PATH = PACKAGED_DATA / "parity_summary.csv"
OUT_DEFAULT = WORKTREE / (
    "docs/reproducibility/canva_figure_provenance/"
    "R03_typography18pt/figures/figure02_seed_boxclusters_v8"
)

COLORS = base.COLORS
FAMILIES = base.FAMILIES
SEEDS = base.SEEDS
TEMP_LABELS = base.TEMP_LABELS
CR_LABELS = base.CR_LABELS

TEMP_CATEGORIES = ["300K", "600K", "1200K"]
CR_CATEGORIES = ["Cr0", "Cr1", "Cr2", "Cr3", "Cr4"]
MULTIT_FAMILIES = ["Scratch-MultiT", "FT-MultiT"]
ALL_TRAJECTORY_FAMILIES = ["Scratch-600K", "FT-600K", "Scratch-MultiT", "FT-MultiT"]
TRAJECTORIES = [
    "2L_o3_t1_600K_aimd_1",
    "2L_octo_Cr1_v2_600K_aimd_1",
]
TRAJECTORY_LABELS = {
    TRAJECTORIES[0]: "O3 t1\nNo Cr · 600 K",
    TRAJECTORIES[1]: "Octahedral Cr1\n1 Cr · 600 K",
}
MODEL_SHORT = {
    "Scratch-600K": "Scratch\n600K",
    "FT-600K": "FT\n600K",
    "Scratch-MultiT": "Scratch\nMultiT",
    "FT-MultiT": "FT\nMultiT",
}
SEED_HATCH = {"123": "", "234": "///", "345": "\\\\\\"}


def _metric_value(record: dict[str, object], metric: str) -> float:
    if metric == "energy":
        return abs(float(record["energy_error_meV"]))
    return float(record["force_rmse_meV_A"])


def _raw_for(
    records: list[dict[str, object]],
    *,
    family: str,
    seed: str,
    group_key: str,
    category: str,
    metric: str,
) -> np.ndarray:
    values = [
        _metric_value(record, metric)
        for record in records
        if str(record["family"]) == family
        and str(record["seed"]) == seed
        and str(record[group_key]) == category
    ]
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def _all_records_for(
    records: list[dict[str, object]], *, group_key: str, categories: list[str]
) -> list[dict[str, object]]:
    return [record for record in records if str(record[group_key]) in categories]


def _log_ticks(limits: tuple[float, float]) -> list[int]:
    low, high = limits
    return list(range(int(np.ceil(low)), int(np.floor(high)) + 1))


def _set_log_exponent_axis(ax, limits: tuple[float, float]) -> None:
    ax.set_ylim(*limits)
    ticks = _log_ticks(limits)
    if ticks:
        ax.set_yticks(ticks)
        ax.set_yticklabels([rf"$10^{{{tick}}}$" for tick in ticks])


def _seed_summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    lower = max(mean - sd, np.finfo(float).tiny)
    upper = max(mean + sd, lower * (1.0 + 1.0e-12))
    return np.log10(mean), np.log10(lower), np.log10(upper)


def _paired_p(values_a: list[float], values_b: list[float]) -> float:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size != 3 or b.size != 3 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a <= 0) or np.any(b <= 0):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = ttest_rel(np.log10(a), np.log10(b))
    return float(result.pvalue) if np.isfinite(result.pvalue) else float("nan")


def _sig_label(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return "n/a"
    if pvalue < 1.0e-4:
        return "****"
    if pvalue < 1.0e-3:
        return "***"
    if pvalue < 1.0e-2:
        return "**"
    if pvalue < 5.0e-2:
        return "*"
    return "ns"


def _comparison_pairs(families: list[str]) -> list[tuple[str, str, str]]:
    if families == MULTIT_FAMILIES:
        return [("Scratch-MultiT", "FT-MultiT", "Scratch vs FT")]
    return [
        ("Scratch-600K", "FT-600K", "Scratch vs FT · 600K"),
        ("Scratch-MultiT", "FT-MultiT", "Scratch vs FT · MultiT"),
        ("Scratch-600K", "Scratch-MultiT", "600K vs MultiT · Scratch"),
        ("FT-600K", "FT-MultiT", "600K vs MultiT · FT"),
    ]


def _draw_bracket(ax, x1: float, x2: float, y: float, label: str) -> None:
    h = 0.045
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#222222", linewidth=1.0, zorder=7, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h + 0.015, label, ha="center", va="bottom", fontsize=18, fontweight="bold", zorder=8)


def _box_and_points(
    ax,
    raw: np.ndarray,
    xpos: float,
    *,
    color: str,
    seed: str,
    family: str,
    category: str,
    metric: str,
    box_width: float,
) -> None:
    if raw.size == 0:
        return
    logged = np.log10(raw)
    rng = np.random.default_rng(base._stable_seed("seed-box", metric, family, category, seed))
    jitter = rng.uniform(-box_width * 0.43, box_width * 0.43, raw.size)
    ax.scatter(
        np.full(raw.size, xpos) + jitter,
        logged,
        s=5.0,
        color=color,
        alpha=0.22,
        edgecolors="none",
        zorder=1,
    )
    result = ax.boxplot(
        [logged],
        positions=[xpos],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        whis=1.5,
        boxprops={
            "facecolor": color,
            "edgecolor": "#111111",
            "linewidth": 0.85,
            "alpha": 0.72,
        },
        medianprops={"color": "#111111", "linewidth": 1.35},
        whiskerprops={"color": "#111111", "linewidth": 0.85},
        capprops={"color": "#111111", "linewidth": 0.85},
    )
    for patch in result["boxes"]:
        patch.set_hatch(SEED_HATCH[seed])
        patch.set_zorder(4)


def _family_handles(families: list[str]) -> list[Patch]:
    return [
        Patch(facecolor=COLORS[family], edgecolor="#111111", linewidth=0.7, alpha=0.72, label=family)
        for family in families
    ]


def _seed_handles() -> list[Patch]:
    return [
        Patch(facecolor="#777777", edgecolor="#111111", linewidth=0.7, alpha=0.72, hatch=SEED_HATCH[seed], label=f"seed {seed}")
        for seed in SEEDS
    ]


def _group_layout(category_index: int, family_index: int, seed_index: int, n_families: int) -> tuple[float, float, float]:
    spacing = 1.85
    if n_families == 2:
        family_offsets = np.asarray([-0.30, 0.30])
        seed_offsets = np.asarray([-0.095, 0.0, 0.095])
        box_width = 0.12
    else:
        family_offsets = np.asarray([-0.56, -0.19, 0.19, 0.56])
        seed_offsets = np.asarray([-0.095, 0.0, 0.095])
        box_width = 0.105
    center = category_index * spacing
    return float(center + family_offsets[family_index] + seed_offsets[seed_index]), float(center + family_offsets[family_index]), box_width


def _short_category_label(category: str, group_key: str) -> str:
    if group_key == "temperature":
        return TEMP_LABELS[category]
    if group_key == "cr":
        return CR_LABELS[category]
    return TRAJECTORY_LABELS[category]


def _collect_limits(
    raw_by_key: dict[tuple[str, str, str], np.ndarray],
    aggregate: dict[str, dict[tuple[str, str], float]],
    categories: list[str],
    families: list[str],
) -> tuple[float, float, float]:
    log_arrays = [np.log10(raw) for raw in raw_by_key.values() if raw.size]
    summary = [
        aggregate.get(category, {}).get((family, seed), np.nan)
        for category in categories
        for family in families
        for seed in SEEDS
    ]
    summary = np.asarray([value for value in summary if np.isfinite(value) and value > 0], dtype=float)
    if summary.size:
        log_arrays.append(np.log10(summary))
    combined = np.concatenate(log_arrays)
    low = float(np.floor(np.nanmin(combined) - 0.18))
    raw_high = float(np.nanmax(combined))
    return low, raw_high, raw_high + 0.35 + 0.23 * len(_comparison_pairs(families))


def clustered_box_figure(
    records: list[dict[str, object]],
    *,
    families: list[str],
    group_key: str,
    categories: list[str],
    labels: dict[str, str],
    xlabel: str,
    title: str,
    stem: str,
    out_dir: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    records = _all_records_for(records, group_key=group_key, categories=categories)
    aggregate = merged_aggregate(records, group_key, categories)
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    fig, axes = plt.subplots(2, 1, figsize=(24.0 if len(families) == 4 else 18.5, 13.2), squeeze=False)
    comparison_manifest: dict[str, object] = {}
    category_spacing = 1.85
    model_centers = []
    for category_index in range(len(categories)):
        for family_index, family in enumerate(families):
            _, model_center, _ = _group_layout(category_index, family_index, 0, len(families))
            model_centers.append((category_index, family, model_center))

    for row_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[row_index, 0]
        raw_by_key: dict[tuple[str, str, str], np.ndarray] = {}
        for category in categories:
            for family in families:
                for seed in SEEDS:
                    raw_by_key[(category, family, seed)] = _raw_for(
                        records,
                        family=family,
                        seed=seed,
                        group_key=group_key,
                        category=category,
                        metric=metric,
                    )
        aggregate_metric = aggregate[metric]
        low, raw_high, bracket_high = _collect_limits(raw_by_key, aggregate_metric, categories, families)
        comparison_manifest[metric] = {}

        for category_index, category in enumerate(categories):
            for family_index, family in enumerate(families):
                for seed_index, seed in enumerate(SEEDS):
                    xpos, _, box_width = _group_layout(category_index, family_index, seed_index, len(families))
                    _box_and_points(
                        ax,
                        raw_by_key[(category, family, seed)],
                        xpos,
                        color=COLORS[family],
                        seed=seed,
                        family=family,
                        category=category,
                        metric=metric,
                        box_width=box_width,
                    )

                values = [aggregate_metric.get(category, {}).get((family, seed), np.nan) for seed in SEEDS]
                if np.isfinite(values).all() and np.all(np.asarray(values) > 0):
                    center, lower, upper = _seed_summary(values)
                    # A compact mean +/- SD summary is placed above the three
                    # seed boxes, preserving the prior Figure 2 statistic.
                    _, model_center, _ = _group_layout(category_index, family_index, 0, len(families))
                    ax.errorbar(
                        [model_center],
                        [center],
                        yerr=[[center - lower], [upper - center]],
                        fmt="none",
                        ecolor="#111111",
                        elinewidth=1.0,
                        capsize=3.0,
                        capthick=1.0,
                        zorder=6,
                    )
                    ax.scatter([model_center], [center], s=28, color="#111111", edgecolors="white", linewidths=0.4, zorder=7)

            pairs = _comparison_pairs(families)
            pair_pvalues = {}
            family_index_lookup = {family: index for index, family in enumerate(families)}
            for level, (family_a, family_b, description) in enumerate(pairs):
                values_a = [aggregate_metric.get(category, {}).get((family_a, seed), np.nan) for seed in SEEDS]
                values_b = [aggregate_metric.get(category, {}).get((family_b, seed), np.nan) for seed in SEEDS]
                pvalue = _paired_p(values_a, values_b)
                pair_pvalues[description] = {"p_value": pvalue, "marker": _sig_label(pvalue), "n_seed_pairs": 3}
                if not np.isfinite(pvalue):
                    continue
                x1 = category_index * category_spacing + np.asarray([-0.56, -0.19, 0.19, 0.56])[family_index_lookup[family_a]] if len(families) == 4 else category_index * category_spacing + np.asarray([-0.30, 0.30])[family_index_lookup[family_a]]
                x2 = category_index * category_spacing + np.asarray([-0.56, -0.19, 0.19, 0.56])[family_index_lookup[family_b]] if len(families) == 4 else category_index * category_spacing + np.asarray([-0.30, 0.30])[family_index_lookup[family_b]]
                _draw_bracket(ax, float(x1), float(x2), raw_high + 0.13 + level * 0.23, _sig_label(pvalue))
            comparison_manifest[metric][category] = pair_pvalues

        _set_log_exponent_axis(ax, (low, bracket_high))
        tick_positions = [center for _, _, center in model_centers]
        tick_labels = [MODEL_SHORT[family] for _, family, _ in model_centers]
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_xlim(-0.95, (len(categories) - 1) * category_spacing + 0.95)
        if row_index == 0:
            ax.tick_params(labelbottom=False)
        else:
            for category_index, category in enumerate(categories):
                ax.text(
                    category_index * category_spacing,
                    -0.205,
                    labels[category],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=18,
                    fontweight="bold",
                )
            ax.set_xlabel(xlabel, labelpad=58)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{chr(65 + row_index)}  {ylabel.split(' RMSE')[0]} distribution", loc="left", pad=10)
        base.style_axis(ax)

    fig.legend(handles=_family_handles(families), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=min(4, len(families)), fontsize=18)
    fig.legend(handles=_seed_handles(), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=3, fontsize=18)
    fig.suptitle(title, y=1.045, fontsize=22, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.83, hspace=0.42)
    return base.save(fig, out_dir, stem), comparison_manifest


def build(out_dir: Path) -> dict[str, object]:
    base.house_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    base.FRAME_ROOT = FRAME_ROOT
    records = base.read_frame_records()
    sources = [SUMMARY_PATH, *sorted(FRAME_ROOT.glob("*__per_frame.csv"))]

    outputs = {}
    significance = {}
    outputs["temperature"], significance["temperature"] = clustered_box_figure(
        records,
        families=MULTIT_FAMILIES,
        group_key="temperature",
        categories=TEMP_CATEGORIES,
        labels=TEMP_LABELS,
        xlabel="trajectory temperature class · train + test merged",
        title="Figure 2 revision | seed-resolved RMSE distributions by MultiT temperature",
        stem="variant04_by_temperature_class_merged_multit_seed_boxclusters",
        out_dir=out_dir,
    )
    outputs["cr_count"], significance["cr_count"] = clustered_box_figure(
        records,
        families=FAMILIES,
        group_key="cr",
        categories=CR_CATEGORIES,
        labels=CR_LABELS,
        xlabel="trajectory Cr-count class · train + test merged",
        title="Figure 2 revision | seed-resolved RMSE distributions by Cr count",
        stem="variant05_by_cr_count_class_merged_seed_boxclusters",
        out_dir=out_dir,
    )
    outputs["trajectory"], significance["trajectory"] = clustered_box_figure(
        records,
        families=ALL_TRAJECTORY_FAMILIES,
        group_key="trajectory",
        categories=TRAJECTORIES,
        labels=TRAJECTORY_LABELS,
        xlabel="common trajectory identity · train + test merged",
        title="Figure 2 revision | fair common-trajectory comparison",
        stem="variant03_by_common_trajectory_seed_boxclusters",
        out_dir=out_dir,
    )

    frame_counts = {}
    for view, group_key, categories, families in [
        ("temperature", "temperature", TEMP_CATEGORIES, MULTIT_FAMILIES),
        ("cr_count", "cr", CR_CATEGORIES, FAMILIES),
        ("trajectory", "trajectory", TRAJECTORIES, ALL_TRAJECTORY_FAMILIES),
    ]:
        frame_counts[view] = {
            f"{family}/{seed}/{category}": int(
                sum(
                    1
                    for record in records
                    if str(record["family"]) == family
                    and str(record["seed"]) == seed
                    and str(record[group_key]) == category
                )
            )
            for family in families
            for seed in SEEDS
            for category in categories
        }

    manifest = {
        "status": "RECONSTRUCTION",
        "purpose": "seed-resolved nested boxplots with complete frame-level point overlays",
        "commits": {"previous_violin_baseline": "a779398", "current_boxcluster_revision": "pending"},
        "grammar": {
            "cluster": "one model cluster per category",
            "sub_box": "one box per seed (123, 234, 345) within each model cluster",
            "raw_points": "all finite frame-level values for that family/seed/category, jittered only horizontally",
            "box_statistics": "median, IQR, and standard 1.5-IQR whiskers in log10-RMSE coordinates; fliers are not hidden because all raw points remain visible",
            "seed_summary": "black mean point and sample SD whisker computed from the three seed-level aggregate RMSE values",
            "significance": "paired two-sided t-test on log10 seed-level aggregates, n=3 paired seeds; markers are ns, *, **, ***, ****",
            "train_test": "train and held-out test frame records merged for all category and selected-trajectory views",
            "model_colors": "R04 / Canva slide-5 palette",
        },
        "data": {
            "source_mode": "readable_packaged_per_frame_csv",
            "summary_path": str(SUMMARY_PATH),
            "frame_root": str(FRAME_ROOT),
            "families": FAMILIES,
            "seeds": SEEDS,
            "splits_merged": base.SPLITS,
            "frame_point_counts": frame_counts,
            "trajectory_selection": {
                "no_cr": TRAJECTORIES[0],
                "with_cr": TRAJECTORIES[1],
                "selection_rule": "common trajectory identity present for all four families and all three seeds; both selected identities are 600 K train trajectories",
            },
            "significance_tests": significance,
            "multiple_testing": "no correction applied; markers are exploratory seed-level diagnostics with n=3",
        },
        "sources": [
            {"path": str(path), "sha256": base.sha256(path), "size_bytes": path.stat().st_size}
            for path in sources
        ],
        "outputs": {
            name: {
                ext: {"path": path, "sha256": base.sha256(Path(path)), "size_bytes": Path(path).stat().st_size}
                for ext, path in files.items()
            }
            for name, files in outputs.items()
        },
    }
    manifest_path = out_dir / "figure02_seed_boxclusters_v8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_seed_boxclusters_v8_caption.md"
    caption_path.write_text(
        "# Figure 2 seed-resolved nested boxplots\n\n"
        "Train and held-out-test frame records are merged. Each model cluster contains three sub-boxplots, one for each seed; every sub-boxplot retains all finite frame-level scatter points for that seed. Boxes are computed in log10-RMSE coordinates and show the median, IQR, and 1.5-IQR whiskers. Black points and whiskers show the mean and sample SD of the three seed-level aggregate RMSE values. Significance brackets use paired two-sided t-tests on the three log10 seed-level aggregates and are exploratory because n=3. The trajectory comparison uses the common 600 K training identities O3 t1 (no Cr) and octahedral Cr1 (1 Cr), which are present for all four model families and all three seeds.\n"
    )
    return {"outputs": outputs, "manifest": str(manifest_path), "caption": str(caption_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
