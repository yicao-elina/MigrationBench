#!/usr/bin/env python3
"""Build the distribution-first Figure 2 revision.

The primary grammar is clustered boxplot + all frame-level points.  The box
shows the empirical median and IQR, the semi-transparent points retain the
complete frame-level coverage, and black points/error bars retain the three
seed-level category RMSE summaries.  This is preferable to a bar-only view
for the skewed, long-tailed frame distributions in these diagnostics.
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

from build_figure02_clustered_v4 import (
    COLORS,
    CR_LABELS,
    CR_ORDER,
    FAMILIES,
    FONT,
    SEEDS,
    SPLITS,
    TEMP_LABELS,
    TEMP_ORDER,
    VECTOR_FIGURE_DIR,
    WORKTREE,
    _asymmetric_sd,
    _raw_points_for,
    _stable_seed,
    _y_floor,
    aggregate_by_group,
    frame_point_counts,
    friendly_trajectory,
    house_style,
    mean_sd,
    read_frame_records,
    read_frame_records_from_svg,
    save,
    sha256,
    split_handles,
    style_axis,
)


OUT_DEFAULT = WORKTREE / (
    "docs/reproducibility/canva_figure_provenance/"
    "R03_typography18pt/figures/figure02_clustered_v6_distribution"
)


def _distribution_artist(
    ax,
    raw: np.ndarray,
    xpos: float,
    *,
    color: str,
    split: str,
    metric: str,
    family: str,
    category: str,
) -> None:
    """Draw all frame points, then a compact empirical violin and box."""
    if raw.size == 0:
        return
    alpha = 0.12 if split == "train" else 0.18
    rng = np.random.default_rng(_stable_seed("raw", metric, family, category, split))
    jitter = rng.uniform(-0.125, 0.125, raw.size)
    ax.scatter(
        np.full(raw.size, xpos) + jitter,
        raw,
        s=5.0,
        color=color,
        alpha=alpha,
        edgecolors="none",
        zorder=1,
    )

    # The violin is deliberately narrow: it summarizes the full empirical
    # distribution without hiding the underlying points.
    violin = ax.violinplot(
        [raw],
        positions=[xpos],
        widths=0.27,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=80,
    )
    body = violin["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("#222222")
    body.set_linewidth(0.55)
    body.set_alpha(0.18 if split == "train" else 0.30)
    if split == "train":
        body.set_hatch("///")

    box = ax.boxplot(
        [raw],
        positions=[xpos],
        widths=0.095,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        boxprops={"facecolor": "white", "edgecolor": "#222222", "linewidth": 0.8, "alpha": 0.80},
        medianprops={"color": "#111111", "linewidth": 1.25},
        whiskerprops={"color": "#222222", "linewidth": 0.8},
        capprops={"color": "#222222", "linewidth": 0.8},
    )
    for patch in box["boxes"]:
        patch.set_zorder(4)


def _draw_seed_summary(
    ax,
    values: list[float],
    xpos: float,
    *,
    floor: float,
) -> None:
    mean, sd = mean_sd(values)
    ax.errorbar(
        [xpos],
        [mean],
        yerr=_asymmetric_sd(mean, sd, floor),
        fmt="none",
        ecolor="#222222",
        elinewidth=1.15,
        capsize=3.0,
        capthick=1.15,
        zorder=5,
    )
    jitter = np.linspace(-0.045, 0.045, len(values))
    ax.scatter(
        np.full(len(values), xpos) + jitter,
        values,
        s=34,
        color="#222222",
        edgecolors="white",
        linewidths=0.4,
        zorder=6,
    )


def category_distribution_figure(
    aggregates: dict[str, dict[str, dict[tuple[str, str, str], float]]],
    frame_records: list[dict[str, object]],
    *,
    group_key: str,
    order: list[str],
    labels: dict[str, str],
    xlabel: str,
    title: str,
    stem: str,
    out_dir: Path,
) -> dict[str, str]:
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    fig, axes = plt.subplots(2, 4, figsize=(24.5, 13.0), squeeze=False)
    for row_index, (metric, ylabel) in enumerate(metrics):
        for col, family in enumerate(FAMILIES):
            ax = axes[row_index, col]
            raw_by_position: dict[tuple[int, str], np.ndarray] = {}
            raw_values = []
            for category_index, category in enumerate(order):
                for split in SPLITS:
                    raw = _raw_points_for(
                        frame_records,
                        family=family,
                        group_key=group_key,
                        category=category,
                        split=split,
                        metric=metric,
                    )
                    raw_by_position[(category_index, split)] = raw
                    raw_values.append(raw)
            floor = _y_floor(raw_values)
            for category_index, category in enumerate(order):
                for split in SPLITS:
                    xpos = float(category_index) + (-0.18 if split == "train" else 0.18)
                    _distribution_artist(
                        ax,
                        raw_by_position[(category_index, split)],
                        xpos,
                        color=COLORS[family],
                        split=split,
                        metric=metric,
                        family=family,
                        category=category,
                    )
                    values_by_seed = aggregates[metric].get(category, {})
                    values = [values_by_seed.get((family, seed, split), np.nan) for seed in SEEDS]
                    if np.isfinite(values).all():
                        _draw_seed_summary(ax, values, xpos, floor=floor)

            ax.set_yscale("log")
            ax.set_ylim(bottom=floor * 0.9)
            ax.set_xticks(range(len(order)), [labels[key] for key in order])
            ax.set_xlim(-0.55, len(order) - 0.35)
            if row_index == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
            ax.set_title(f"{chr(65 + col + row_index * 4)}  {family}", loc="left", color=COLORS[family], pad=10)
            style_axis(ax)
    fig.legend(handles=split_handles(), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, fontsize=18)
    fig.suptitle(title, y=1.015, fontsize=22, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.865, hspace=0.36, wspace=0.12)
    return save(fig, out_dir, stem)


def trajectory_distribution_figure(
    aggregates: dict[str, dict[str, dict[tuple[str, str, str], float]]],
    frame_records: list[dict[str, object]],
    out_dir: Path,
) -> dict[str, str]:
    test_trajectories = ["2L_o3_t3_600K_aimd_1", "2L_tetra_Cr4_600K_aimd_1"]
    labels = {trajectory: friendly_trajectory(trajectory) for trajectory in test_trajectories}
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    fig, axes = plt.subplots(2, 1, figsize=(17.5, 12.5), squeeze=False)
    for row_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[row_index, 0]
        raw_by_position: dict[tuple[int, str], np.ndarray] = {}
        raw_values = []
        for trajectory_index, trajectory in enumerate(test_trajectories):
            for family in FAMILIES:
                raw = _raw_points_for(
                    frame_records,
                    family=family,
                    group_key="trajectory",
                    category=trajectory,
                    split="test",
                    metric=metric,
                )
                raw_by_position[(trajectory_index, family)] = raw
                raw_values.append(raw)
        floor = _y_floor(raw_values)
        for trajectory_index, trajectory in enumerate(test_trajectories):
            values_by_seed = aggregates[metric].get(trajectory, {})
            for family_index, family in enumerate(FAMILIES):
                xpos = float(trajectory_index) + (family_index - 1.5) * 0.18
                _distribution_artist(
                    ax,
                    raw_by_position[(trajectory_index, family)],
                    xpos,
                    color=COLORS[family],
                    split="test",
                    metric=metric,
                    family=family,
                    category=trajectory,
                )
                values = [values_by_seed.get((family, seed, "test"), np.nan) for seed in SEEDS]
                if np.isfinite(values).all():
                    _draw_seed_summary(ax, values, xpos, floor=floor)
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor * 0.9)
        ax.set_xticks(range(len(test_trajectories)), [labels[trajectory] for trajectory in test_trajectories])
        ax.set_ylabel(ylabel)
        if row_index == 1:
            ax.set_xlabel("held-out trajectory")
        else:
            ax.tick_params(labelbottom=False)
        ax.set_title(f"{chr(65 + row_index)}  held-out trajectory distribution", loc="left", pad=10)
        style_axis(ax)
    family_handles = [
        Line2D([], [], marker="s", linestyle="", color=COLORS[family], markersize=9, label=family)
        for family in FAMILIES
    ]
    fig.legend(handles=family_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, fontsize=18)
    fig.suptitle("Figure 2 revision | held-out trajectory distributions", y=1.015, fontsize=22, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.86, hspace=0.36)
    return save(fig, out_dir, "variant03_by_heldout_trajectory_clustered_box_full_points")


def build(out_dir: Path, vector_dir: Path | None = None) -> dict[str, object]:
    house_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    if vector_dir is None:
        records = read_frame_records()
        temperature_records = records
        cr_records = records
        sources = [
            Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/parity_summary.csv"),
            *sorted(Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/raw/rockfish_parity").glob("*__per_frame.csv")),
        ]
        source_mode = "per_frame_csv"
        force_aggregation = "n_atoms-weighted RMSE of per-frame force RMSE in meV A^-1"
        energy_aggregation = "RMSE of signed per-frame energy error in meV per atom"
    else:
        temperature_svg = vector_dir / "variant04_by_temperature_class.svg"
        cr_svg = vector_dir / "variant05_by_cr_count_class.svg"
        vector_manifest = vector_dir / "trajectory_variants_v2_manifest.json"
        temperature_records = read_frame_records_from_svg(temperature_svg, "temperature")
        cr_records = read_frame_records_from_svg(cr_svg, "cr")
        records = temperature_records
        sources = [temperature_svg, cr_svg, vector_manifest]
        source_mode = "recovered_from_prior_vector_point_sources"
        force_aggregation = "unweighted RMS of plotted per-frame force RMSE; n_atoms are not encoded in the source SVG"
        energy_aggregation = "RMSE of plotted per-frame absolute energy error; this equals signed-error RMSE after squaring"

    temperature_aggregates = aggregate_by_group(temperature_records, "temperature")
    cr_aggregates = aggregate_by_group(cr_records, "cr")
    trajectory_aggregates = aggregate_by_group(cr_records, "trajectory")
    outputs = {
        "temperature": category_distribution_figure(
            temperature_aggregates,
            temperature_records,
            group_key="temperature",
            order=TEMP_ORDER,
            labels=TEMP_LABELS,
            xlabel="trajectory temperature class",
            title="Figure 2 revision | clustered distributions by temperature class",
            stem="variant04_by_temperature_class_clustered_box_full_points",
            out_dir=out_dir,
        ),
        "cr_count": category_distribution_figure(
            cr_aggregates,
            cr_records,
            group_key="cr",
            order=CR_ORDER,
            labels=CR_LABELS,
            xlabel="trajectory Cr-count class",
            title="Figure 2 revision | clustered distributions by Cr-count class",
            stem="variant05_by_cr_count_class_clustered_box_full_points",
            out_dir=out_dir,
        ),
        "trajectory": trajectory_distribution_figure(trajectory_aggregates, cr_records, out_dir),
    }
    coverage = defaultdict(int)
    for record in temperature_records:
        coverage[f"{record['family']}/{record['split']}/{record['temperature']}"] += 1
    frame_counts = {
        "temperature": frame_point_counts(temperature_records, "temperature", TEMP_ORDER),
        "cr_count": frame_point_counts(cr_records, "cr", CR_ORDER),
        "held_out_trajectory": frame_point_counts(
            [record for record in cr_records if str(record["split"]) == "test"],
            "trajectory",
            ["2L_o3_t3_600K_aimd_1", "2L_tetra_Cr4_600K_aimd_1"],
        ),
    }
    manifest = {
        "status": "RECONSTRUCTION",
        "purpose": "distribution-first clustered box/violin replacements for temperature, Cr-count, and held-out trajectory diagnostics",
        "recommendation": "clustered boxplot plus complete raw points is the primary paper-facing grammar; violin is retained as a light density layer rather than the only distribution encoding",
        "grammar": {
            "raw_points": "all finite frame-level points, model-colored and semi-transparent",
            "violin": "narrow empirical density layer around each train/test/family position",
            "box": "median and IQR, with whiskers from the empirical boxplot convention and fliers suppressed because all raw points are shown",
            "seed_summary": "three black seed-level category aggregates with sample SD whisker",
            "train_style": "transparent hatched distribution layer",
            "held_out_test_style": "opaque distribution layer",
            "model_colors": "R04 palette",
            "energy_aggregation": energy_aggregation,
            "force_aggregation": force_aggregation,
        },
        "category_definitions": {"temperature": TEMP_LABELS, "cr_count": CR_LABELS},
        "data": {
            "source_mode": source_mode,
            "source_summary_path": "/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/parity_summary.csv",
            "source_frame_root": "/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/rmse_train_test_20261026/raw/rockfish_parity",
            "families": FAMILIES,
            "seeds": SEEDS,
            "splits": SPLITS,
            "frame_point_counts": frame_counts,
            "test_trajectory_identities": ["2L_o3_t3_600K_aimd_1", "2L_tetra_Cr4_600K_aimd_1"],
            "loto_status": "NOT_PERFORMED",
            "loto_requirement": "requires retraining/evaluation with trajectory-group holdout; fixed-split predictions are not LOTO",
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
    manifest_path = out_dir / "figure02_clustered_v6_distribution_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_clustered_v6_distribution_caption.md"
    caption_path.write_text(
        "# Figure 2 distribution-first clustered diagnostics\n\n"
        "Each train/test and model-family position shows every available finite frame-level point, a narrow violin density layer, and an empirical boxplot for the median/IQR. The three black points are the seed-level category RMSE aggregates and their whisker is the sample SD across seeds. Transparent hatched layers are train and opaque layers are held-out test. The two held-out trajectories are the fixed test identities present in the package; this is not a leave-one-trajectory-out retraining result.\n\n"
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
        import build_figure02_clustered_v4 as base

        base.FRAME_ROOT = args.frame_root
    print(json.dumps(build(args.out_dir, args.vector_dir), indent=2))


if __name__ == "__main__":
    main()
