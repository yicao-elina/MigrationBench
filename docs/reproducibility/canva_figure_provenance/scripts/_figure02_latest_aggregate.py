#!/usr/bin/env python3
"""Build the merged-train/test, violin-only Figure 2 diagnostics.

Train and held-out-test frame records are merged whenever the category label
is available. The temperature view is restricted to the two MultiT families,
because the 600K-only families cannot support a temperature-class comparison.
The held-out-trajectory view remains test-identity-specific because the
trajectory identity is not recoverable for training records; those unknown
training trajectories are not silently relabelled.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _figure02_latest_base import (
    COLORS,
    CR_LABELS,
    FAMILIES,
    SEEDS,
    SPLITS,
    TEMP_LABELS,
    TEMP_ORDER,
    WORKTREE,
    _stable_seed,
    friendly_trajectory,
    house_style,
    read_frame_records,
    read_frame_records_from_svg,
    save,
    sha256,
    style_axis,
)


OUT_DEFAULT = WORKTREE / (
    "docs/reproducibility/canva_figure_provenance/"
    "R03_typography18pt/figures/figure02_merged_violin_v7"
)

MERGED_TEMP_ORDER = ["300K", "600K", "1200K"]
MERGED_CR_ORDER = ["Cr0", "Cr1", "Cr2", "Cr3", "Cr4"]
MULTIT_FAMILIES = ["Scratch-MultiT", "FT-MultiT"]
TRAJECTORIES = ["2L_o3_t3_600K_aimd_1", "2L_tetra_Cr4_600K_aimd_1"]


def _metric_value(record: dict[str, object], metric: str) -> float:
    if metric == "energy":
        return abs(float(record["energy_error_meV"]))
    return float(record["force_rmse_meV_A"])


def merged_raw_points(
    records: list[dict[str, object]],
    *,
    family: str,
    group_key: str,
    category: str,
    metric: str,
    split: str | None = None,
) -> np.ndarray:
    values = [
        _metric_value(record, metric)
        for record in records
        if str(record["family"]) == family
        and str(record[group_key]) == category
        and (split is None or str(record["split"]) == split)
    ]
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def merged_aggregate(
    records: list[dict[str, object]], group_key: str, categories: list[str]
) -> dict[str, dict[str, dict[tuple[str, str], float]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        category = str(record[group_key])
        if category in categories:
            grouped[(category, str(record["family"]), str(record["seed"]))].append(record)
    output: dict[str, dict[str, dict[tuple[str, str], float]]] = {"energy": {}, "force": {}}
    for (category, family, seed), values in grouped.items():
        energy = np.asarray([float(value["energy_error_meV"]) for value in values], dtype=float)
        force = np.asarray([float(value["force_rmse_meV_A"]) for value in values], dtype=float)
        weights = np.asarray([float(value["n_atoms"]) for value in values], dtype=float)
        output["energy"].setdefault(category, {})[(family, seed)] = float(np.sqrt(np.mean(energy * energy)))
        output["force"].setdefault(category, {})[(family, seed)] = float(
            np.sqrt(np.sum(weights * force * force) / np.sum(weights))
        )
    return output


def _log_values(values: np.ndarray) -> np.ndarray:
    return np.log10(values[np.isfinite(values) & (values > 0.0)])


def _axis_limits(raw_arrays: list[np.ndarray], summary_values: list[float]) -> tuple[float, float]:
    logged = [_log_values(array) for array in raw_arrays if array.size]
    if summary_values:
        logged.append(_log_values(np.asarray(summary_values, dtype=float)))
    combined = np.concatenate(logged) if logged else np.asarray([0.0])
    low = float(np.nanmin(combined))
    high = float(np.nanmax(combined))
    # A small exponent-space margin keeps the violin outlines and whiskers off
    # the frame while preserving every finite point, including long-tail values.
    return np.floor(low - 0.15), np.ceil(high + 0.15)


def _set_log_exponent_axis(ax, limits: tuple[float, float]) -> None:
    low, high = limits
    ax.set_ylim(low, high)
    ticks = np.arange(int(np.ceil(low)), int(np.floor(high)) + 1, dtype=int)
    if ticks.size:
        ax.set_yticks(ticks)
        ax.set_yticklabels([rf"$10^{{{tick}}}$" for tick in ticks])


def _seed_summary(ax, values: list[float], xpos: float) -> None:
    array = np.asarray(values, dtype=float)
    if array.size != 3 or not np.isfinite(array).all() or np.any(array <= 0.0):
        return
    logged = np.log10(array)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    lower = max(mean - sd, np.finfo(float).tiny)
    upper = max(mean + sd, lower * (1.0 + 1.0e-9))
    center_log = np.log10(mean)
    yerr = np.asarray([[center_log - np.log10(lower)], [np.log10(upper) - center_log]])
    ax.errorbar(
        [xpos],
        [center_log],
        yerr=yerr,
        fmt="none",
        ecolor="#222222",
        elinewidth=1.15,
        capsize=3.0,
        capthick=1.15,
        zorder=5,
    )
    jitter = np.linspace(-0.045, 0.045, array.size)
    ax.scatter(
        np.full(array.size, xpos) + jitter,
        logged,
        s=34,
        color="#222222",
        edgecolors="white",
        linewidths=0.4,
        zorder=6,
    )


def _violin(
    ax,
    raw: np.ndarray,
    xpos: float,
    *,
    color: str,
    metric: str,
    family: str,
    category: str,
    max_count: int,
    base_width: float,
) -> None:
    if raw.size == 0:
        return
    logged = np.log10(raw)
    count_scale = float(np.sqrt(raw.size / max(max_count, 1)))
    width = base_width * (0.68 + 0.32 * count_scale)
    alpha = 0.24 + 0.16 * count_scale
    violin = ax.violinplot(
        [logged],
        positions=[xpos],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=100,
        bw_method=0.22,
    )
    body = violin["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("#222222")
    body.set_linewidth(0.65)
    body.set_alpha(alpha)
    body.set_zorder(2)


def _family_legend(families: list[str]) -> list[Line2D]:
    return [
        Line2D([], [], marker="s", linestyle="", color=COLORS[family], markersize=9, label=family)
        for family in families
    ]


def clustered_violin_figure(
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
    split: str | None = None,
) -> dict[str, str]:
    filtered_records = records if split is None else [record for record in records if str(record["split"]) == split]
    aggregates = merged_aggregate(filtered_records, group_key, categories)
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    fig, axes = plt.subplots(2, 1, figsize=(17.5, 12.5), squeeze=False)
    offsets = np.linspace(-0.27, 0.27, len(families)) if len(families) > 1 else np.asarray([0.0])
    for row_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[row_index, 0]
        raw_by_position: dict[tuple[int, str], np.ndarray] = {}
        raw_arrays = []
        seed_values = []
        for category_index, category in enumerate(categories):
            for family in families:
                raw = merged_raw_points(
                    filtered_records,
                    family=family,
                    group_key=group_key,
                    category=category,
                    metric=metric,
                    split=None,
                )
                raw_by_position[(category_index, family)] = raw
                raw_arrays.append(raw)
                values = [aggregates[metric].get(category, {}).get((family, seed), np.nan) for seed in SEEDS]
                seed_values.extend([value for value in values if np.isfinite(value) and value > 0.0])
        limits = _axis_limits(raw_arrays, seed_values)
        max_count = max((raw.size for raw in raw_arrays), default=1)
        for category_index, category in enumerate(categories):
            for family_index, family in enumerate(families):
                xpos = float(category_index) + float(offsets[family_index])
                _violin(
                    ax,
                    raw_by_position[(category_index, family)],
                    xpos,
                    color=COLORS[family],
                    metric=metric,
                    family=family,
                    category=category,
                    max_count=max_count,
                    base_width=0.30 if len(families) <= 2 else 0.18,
                )
                values = [aggregates[metric].get(category, {}).get((family, seed), np.nan) for seed in SEEDS]
                if np.isfinite(values).all() and np.all(np.asarray(values) > 0.0):
                    _seed_summary(ax, values, xpos)
        _set_log_exponent_axis(ax, limits)
        ax.set_xticks(range(len(categories)), [labels[key] for key in categories])
        ax.set_xlim(-0.55, len(categories) - 0.45)
        ax.set_ylabel(ylabel)
        if row_index == 1:
            ax.set_xlabel(xlabel)
        else:
            ax.tick_params(labelbottom=False)
        ax.set_title(f"{chr(65 + row_index)}  {ylabel.split(' RMSE')[0]} distribution", loc="left", pad=10)
        style_axis(ax)
    ncol = min(4, len(families))
    fig.legend(handles=_family_legend(families), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=ncol, fontsize=18)
    fig.suptitle(title, y=1.015, fontsize=22, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.86, hspace=0.36)
    return save(fig, out_dir, stem)


def build(out_dir: Path, vector_dir: Path | None = None) -> dict[str, object]:
    house_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    if vector_dir is None:
        records = read_frame_records()
        source_root = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/raw/rockfish_parity")
        sources = [
            Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/parity_summary.csv"),
            *sorted(source_root.glob("*__per_frame.csv")),
        ]
        source_mode = "per_frame_csv"
        force_aggregation = "n_atoms-weighted RMSE of merged train+test per-frame force RMSE in meV A^-1"
        energy_aggregation = "RMSE of signed merged train+test per-frame energy error in meV per atom"
    else:
        temperature_svg = vector_dir / "variant04_by_temperature_class.svg"
        cr_svg = vector_dir / "variant05_by_cr_count_class.svg"
        vector_manifest = vector_dir / "trajectory_variants_v2_manifest.json"
        temperature_records = read_frame_records_from_svg(temperature_svg, "temperature")
        cr_records = read_frame_records_from_svg(cr_svg, "cr")
        records = temperature_records
        sources = [temperature_svg, cr_svg, vector_manifest]
        source_mode = "recovered_from_prior_vector_point_sources"
        force_aggregation = "unweighted RMS of merged plotted per-frame force RMSE; n_atoms are not encoded in the source SVG"
        energy_aggregation = "RMSE of merged plotted per-frame absolute energy error; this equals signed-error RMSE after squaring"
    if vector_dir is None:
        temperature_records = records
        cr_records = records

    # Temperature is a MultiT question: the 600K-only families cannot define
    # 300K/600K/1200K categories and are therefore excluded from this view.
    outputs = {
        "temperature": clustered_violin_figure(
            temperature_records,
            families=MULTIT_FAMILIES,
            group_key="temperature",
            categories=MERGED_TEMP_ORDER,
            labels=TEMP_LABELS,
            xlabel="trajectory temperature class; train + test merged",
            title="Figure 2 revision | merged MultiT frame distributions by temperature class",
            stem="variant04_by_temperature_class_merged_multit_violin",
            out_dir=out_dir,
        ),
        "cr_count": clustered_violin_figure(
            cr_records,
            families=FAMILIES,
            group_key="cr",
            categories=MERGED_CR_ORDER,
            labels=CR_LABELS,
            xlabel="trajectory Cr-count class; train + test merged",
            title="Figure 2 revision | merged frame distributions by Cr-count class",
            stem="variant05_by_cr_count_class_merged_violin",
            out_dir=out_dir,
        ),
        "trajectory": clustered_violin_figure(
            cr_records,
            families=FAMILIES,
            group_key="trajectory",
            categories=TRAJECTORIES,
            labels={trajectory: friendly_trajectory(trajectory) for trajectory in TRAJECTORIES},
            xlabel="available held-out trajectory identity",
            title="Figure 2 revision | held-out trajectory distributions",
            stem="variant03_by_heldout_trajectory_violin_rescaled",
            out_dir=out_dir,
            split="test",
        ),
    }
    frame_counts = {
        "temperature_merged_train_test": {
            f"{family}/{category}": sum(
                1
                for record in temperature_records
                if str(record["family"]) == family
                and str(record["temperature"]) == category
                and str(record["split"]) in SPLITS
            )
            for family in MULTIT_FAMILIES
            for category in MERGED_TEMP_ORDER
            if any(
                str(record["family"]) == family and str(record["temperature"]) == category
                for record in temperature_records
            )
        },
        "cr_count_merged_train_test": {
            f"{family}/{category}": sum(
                1
                for record in cr_records
                if str(record["family"]) == family
                and str(record["cr"]) == category
                and str(record["split"]) in SPLITS
            )
            for family in FAMILIES
            for category in MERGED_CR_ORDER
            if any(str(record["family"]) == family and str(record["cr"]) == category for record in cr_records)
        },
        "held_out_trajectory_test_only": {
            f"{family}/{trajectory}": sum(
                1
                for record in cr_records
                if str(record["family"]) == family
                and str(record["trajectory"]) == trajectory
                and str(record["split"]) == "test"
            )
            for family in FAMILIES
            for trajectory in TRAJECTORIES
            if any(
                str(record["family"]) == family
                and str(record["trajectory"]) == trajectory
                and str(record["split"]) == "test"
                for record in cr_records
            )
        },
    }
    manifest = {
        "status": "RECONSTRUCTION",
        "purpose": "merged-train/test violin-only Figure 2 diagnostics",
        "recommendation": "violin-only is used for the primary distribution view; width is density and is mildly scaled by frame count, while three black seed points and their SD whisker retain seed-level summaries",
        "grammar": {
            "distribution": "violin in log10-RMSE coordinates",
            "frame_points": "all finite frame-level values are included in the violin input; no box or bar overlays",
            "count_encoding": "violin width and a subtle alpha modulation scale with frame count within each panel",
            "seed_summary": "three black seed-level RMSE points with sample SD whisker",
            "split_handling": "train and test are merged where a category label is available",
            "model_colors": "R04 palette",
            "energy_aggregation": energy_aggregation,
            "force_aggregation": force_aggregation,
        },
        "data": {
            "source_mode": source_mode,
            "source_summary_path": "/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/parity_summary.csv",
            "source_frame_root": "/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/raw/rockfish_parity",
            "families": FAMILIES,
            "temperature_families": MULTIT_FAMILIES,
            "seeds": SEEDS,
            "splits_merged": SPLITS,
            "frame_point_counts": frame_counts,
            "test_trajectory_identities": TRAJECTORIES,
            "trajectory_split_note": "The trajectory identity is only available for the fixed held-out test records in the recovered source; unknown training trajectories are not relabelled into O3/Tetra.",
            "loto_status": "NOT_PERFORMED",
            "vector_recovery_note": (
                "Prior SVG point vectors were used because some canonical CSVs are iCloud dataless placeholders; "
                "the SVG stores all plotted frame points but not n_atoms or original seed IDs."
                if vector_dir is not None
                else None
            ),
        },
        "sources": [
            {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in sources
        ],
        "outputs": {
            name: {
                ext: {"path": path, "sha256": sha256(Path(path)), "size_bytes": Path(path).stat().st_size}
                for ext, path in files.items()
            }
            for name, files in outputs.items()
        },
    }
    manifest_path = out_dir / "figure02_merged_violin_v7_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_merged_violin_v7_caption.md"
    caption_path.write_text(
        "# Figure 2 merged-train/test violin diagnostics\n\n"
        "Train and held-out-test frame-level values are merged whenever their category labels are available. The temperature panel is restricted to Scratch-MultiT and FT-MultiT, because the 600K-only families do not define a 300K/600K/1200K temperature comparison. Each violin is drawn in log10-RMSE coordinates from all finite frame-level values; its width represents density and is mildly scaled by frame count. Three black points and the sample SD whisker retain the three seed-level category summaries. The held-out-trajectory panel remains test-identity-specific because training records do not carry recoverable O3/Tetra labels.\n\n"
        f"Energy aggregation: {energy_aggregation}. Force aggregation: {force_aggregation}.\n"
    )
    return {"outputs": outputs, "manifest": str(manifest_path), "caption": str(caption_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--frame-root", type=Path, default=None)
    parser.add_argument("--vector-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.frame_root is not None:
        import _figure02_latest_base as base

        base.FRAME_ROOT = args.frame_root
    print(json.dumps(build(args.out_dir, args.vector_dir), indent=2))


if __name__ == "__main__":
    main()

