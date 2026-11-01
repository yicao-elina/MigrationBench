#!/usr/bin/env python3
"""Build the compact Figure 2 revision and an exploratory 3-D surface view.

The compact temperature panel reuses the seed-resolved box grammar with a
shorter category spacing.  The Cr-count deliverables include both an
exploratory 3-D surface-boxplot and a paper-facing 2-D view with the three
seeds pooled inside one distribution per model and Cr-count class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import build_figure02_seed_boxclusters_v9 as v9


WORKTREE = Path("/Users/alina/.codex/worktrees/canva-figure-provenance")
OUT_DEFAULT = WORKTREE / (
    "docs/reproducibility/canva_figure_provenance/"
    "R03_typography18pt/figures/figure02_surface_pooled_v10"
)


def _records() -> list[dict[str, object]]:
    v9.base.FRAME_ROOT = v9.FRAME_ROOT
    return v9._apply_atom_based_cr_classes(v9.base.read_frame_records())


def _pooled_layout(category_index: int, family_index: int, n_families: int) -> tuple[float, float, float]:
    spacing = 2.20
    if n_families != 4:
        offsets = np.asarray([-0.36, 0.36])
        width = 0.24
    else:
        offsets = np.asarray([-0.66, -0.22, 0.22, 0.66])
        width = 0.22
    center = category_index * spacing
    x = float(center + offsets[family_index])
    return x, x, width


def _pooled_box_and_points(ax, raw: np.ndarray, xpos: float, *, color: str, category: str, family: str, metric: str, width: float) -> None:
    if raw.size == 0:
        return
    logged = np.log10(raw)
    rng = np.random.default_rng(v9.base._stable_seed("pooled-box", metric, family, category))
    jitter = rng.uniform(-width * 0.42, width * 0.42, raw.size)
    ax.scatter(
        np.full(raw.size, xpos) + jitter,
        logged,
        s=5.0,
        color=color,
        alpha=0.24,
        edgecolors="none",
        zorder=1,
    )
    result = ax.boxplot(
        [logged],
        positions=[xpos],
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        whis=1.5,
        boxprops={"facecolor": color, "edgecolor": "#111111", "linewidth": 1.55, "alpha": 0.78},
        medianprops={"color": "#111111", "linewidth": 2.1},
        whiskerprops={"color": "#111111", "linewidth": 1.55},
        capprops={"color": "#111111", "linewidth": 1.55},
    )
    for patch in result["boxes"]:
        patch.set_zorder(4)


def _pooled_cr_count_figure(records: list[dict[str, object]], out_dir: Path) -> dict[str, object]:
    categories = v9.CR_CATEGORIES
    families = v9.FAMILIES
    records = v9._all_records_for(records, group_key="cr", categories=categories)
    aggregate = v9.merged_aggregate(records, "cr", categories)
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    fig, axes = plt.subplots(2, 1, figsize=(29.0, 13.8), squeeze=False)
    comparison_manifest: dict[str, object] = {}
    category_spacing = 2.20

    for row_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[row_index, 0]
        raw_by_key: dict[tuple[str, str], np.ndarray] = {}
        for category in categories:
            for family in families:
                seed_arrays = [
                    v9._raw_for(records, family=family, seed=seed, group_key="cr", category=category, metric=metric)
                    for seed in v9.SEEDS
                ]
                raw_by_key[(category, family)] = np.concatenate([array for array in seed_arrays if array.size])
        low, raw_high, _ = v9._collect_limits(
            {(category, family, "pooled"): raw for (category, family), raw in raw_by_key.items()},
            aggregate[metric],
            categories,
            families,
        )
        bracket_step = 0.60
        bracket_high = raw_high + 0.45 + bracket_step * len(v9._comparison_pairs(families))

        for category_index, category in enumerate(categories):
            for family_index, family in enumerate(families):
                xpos, model_center, width = _pooled_layout(category_index, family_index, len(families))
                _pooled_box_and_points(
                    ax,
                    raw_by_key[(category, family)],
                    xpos,
                    color=v9.COLORS[family],
                    category=category,
                    family=family,
                    metric=metric,
                    width=width,
                )
                values = [aggregate[metric].get(category, {}).get((family, seed), np.nan) for seed in v9.SEEDS]
                if np.isfinite(values).all() and np.all(np.asarray(values) > 0):
                    center, lower, upper = v9._seed_summary(values)
                    ax.errorbar(
                        [model_center],
                        [center],
                        yerr=[[center - lower], [upper - center]],
                        fmt="none",
                        ecolor="#111111",
                        elinewidth=1.5,
                        capsize=4.0,
                        capthick=1.5,
                        zorder=6,
                    )
                    ax.scatter([model_center], [center], s=32, color="#111111", edgecolors="white", linewidths=0.5, zorder=7)

            pair_pvalues = {}
            family_index_lookup = {family: index for index, family in enumerate(families)}
            for level, (family_a, family_b, description) in enumerate(v9._comparison_pairs(families)):
                values_a = [aggregate[metric].get(category, {}).get((family_a, seed), np.nan) for seed in v9.SEEDS]
                values_b = [aggregate[metric].get(category, {}).get((family_b, seed), np.nan) for seed in v9.SEEDS]
                pvalue = v9._paired_p(values_a, values_b)
                pair_pvalues[description] = {"p_value": pvalue, "marker": v9._sig_label(pvalue), "n_seed_pairs": 3}
                if not np.isfinite(pvalue):
                    continue
                _, x1, _ = _pooled_layout(category_index, family_index_lookup[family_a], len(families))
                _, x2, _ = _pooled_layout(category_index, family_index_lookup[family_b], len(families))
                v9._draw_bracket(ax, x1, x2, raw_high + 0.18 + level * bracket_step, v9._sig_label(pvalue))
            comparison_manifest.setdefault(metric, {})[category] = pair_pvalues

        v9._set_log_exponent_axis(ax, (low, bracket_high))
        tick_positions = [
            _pooled_layout(category_index, family_index, len(families))[1]
            for category_index in range(len(categories))
            for family_index in range(len(families))
        ]
        tick_labels = [
            v9.MODEL_SHORT[family]
            for _ in categories
            for family in families
        ]
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_xlim(-1.10, (len(categories) - 1) * category_spacing + 1.10)
        if row_index == 0:
            ax.tick_params(labelbottom=False)
        else:
            for category_index, category in enumerate(categories):
                ax.text(
                    category_index * category_spacing,
                    -0.205,
                    v9.CR_LABELS[category],
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=20,
                    fontweight="bold",
                )
            ax.set_xlabel("trajectory Cr-count class · train + test merged · seeds pooled", labelpad=58)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{chr(65 + row_index)}  {ylabel.split(' RMSE')[0]} distribution", loc="left", pad=10)
        v9._style_axis(ax)

    fig.legend(
        handles=v9._family_handles(families),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=4,
        prop={"weight": "bold", "size": 20},
    )
    fig.suptitle("Figure 2 revision | seed-pooled RMSE distributions by Cr count", y=1.045, fontsize=24, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.23, top=0.83, hspace=0.42)
    paths = v9.base.save(fig, out_dir, "variant05_by_cr_count_class_seed_pooled_boxplot")
    return {"outputs": paths, "significance": comparison_manifest}


def _surface_statistics(records: list[dict[str, object]], metric: str) -> dict[str, np.ndarray]:
    records = v9._all_records_for(records, group_key="cr", categories=v9.CR_CATEGORIES)
    shape = (len(v9.FAMILIES), len(v9.CR_CATEGORIES))
    q1 = np.full(shape, np.nan)
    median = np.full(shape, np.nan)
    q3 = np.full(shape, np.nan)
    lower = np.full(shape, np.nan)
    upper = np.full(shape, np.nan)
    for family_index, family in enumerate(v9.FAMILIES):
        for category_index, category in enumerate(v9.CR_CATEGORIES):
            arrays = [
                v9._raw_for(records, family=family, seed=seed, group_key="cr", category=category, metric=metric)
                for seed in v9.SEEDS
            ]
            raw = np.concatenate([array for array in arrays if array.size])
            logged = np.log10(raw)
            q1_value, med_value, q3_value = np.quantile(logged, [0.25, 0.50, 0.75])
            iqr = q3_value - q1_value
            q1[family_index, category_index] = q1_value
            median[family_index, category_index] = med_value
            q3[family_index, category_index] = q3_value
            lower[family_index, category_index] = max(float(np.min(logged)), q1_value - 1.5 * iqr)
            upper[family_index, category_index] = min(float(np.max(logged)), q3_value + 1.5 * iqr)
    return {"q1": q1, "median": median, "q3": q3, "lower": lower, "upper": upper}


def _surface_panel(ax, stats: dict[str, np.ndarray], metric_label: str) -> None:
    x_values = np.arange(1, len(v9.CR_CATEGORIES) + 1, dtype=float)
    y_values = np.arange(len(v9.FAMILIES), dtype=float)
    for family_index, family in enumerate(v9.FAMILIES):
        color = v9.COLORS[family]
        x_surface = np.vstack([x_values, x_values])
        y_surface = np.vstack([
            np.full_like(x_values, family_index - 0.25),
            np.full_like(x_values, family_index + 0.25),
        ])
        z_median = np.vstack([stats["median"][family_index], stats["median"][family_index]])
        z_q1 = np.vstack([stats["q1"][family_index], stats["q1"][family_index]])
        z_q3 = np.vstack([stats["q3"][family_index], stats["q3"][family_index]])
        ax.plot_surface(x_surface, y_surface, z_median, color=color, alpha=0.88, linewidth=0.35, edgecolor="#222222", shade=False)
        ax.plot_surface(x_surface, y_surface, z_q1, color=color, alpha=0.22, linewidth=0.25, edgecolor="#333333", shade=False)
        ax.plot_surface(x_surface, y_surface, z_q3, color=color, alpha=0.22, linewidth=0.25, edgecolor="#333333", shade=False)
        ax.plot(x_values, np.full_like(x_values, family_index), stats["lower"][family_index], color=color, linewidth=1.5, linestyle="--")
        ax.plot(x_values, np.full_like(x_values, family_index), stats["upper"][family_index], color=color, linewidth=1.5, linestyle="--")
        for x, lo, hi in zip(x_values, stats["lower"][family_index], stats["upper"][family_index]):
            ax.plot([x, x], [family_index, family_index], [lo, hi], color=color, linewidth=1.0, alpha=0.75)

    ax.set_xlabel("Cr count", labelpad=12, fontweight="bold", fontsize=20)
    ax.set_ylabel("Model family", labelpad=14, fontweight="bold", fontsize=20)
    ax.set_zlabel(f"log$_{{10}}$ {metric_label}", labelpad=12, fontweight="bold", fontsize=20)
    ax.set_xticks(np.arange(1, 7), [str(i) for i in range(1, 7)])
    ax.set_yticks(y_values, [family.replace("-", "\n") for family in v9.FAMILIES])
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels(), *ax.get_zticklabels()]:
        label.set_fontsize(16)
        label.set_fontweight("bold")
    ax.set_title(metric_label, fontsize=22, fontweight="bold", pad=16)
    ax.view_init(elev=25, azim=-58)
    ax.set_box_aspect((1.8, 1.25, 1.25))
    ax.grid(False)


def _surface_figure(records: list[dict[str, object]], out_dir: Path) -> dict[str, str]:
    fig = plt.figure(figsize=(25.0, 12.8))
    axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    _surface_panel(axes[0], _surface_statistics(records, "energy"), "Energy RMSE")
    _surface_panel(axes[1], _surface_statistics(records, "force"), "Force RMSE")
    legend = [
        Patch(facecolor=v9.COLORS[family], edgecolor="#111111", label=family, alpha=0.78)
        for family in v9.FAMILIES
    ]
    legend.extend([
        Line2D([0], [0], color="#222222", linewidth=2.0, label="median surface"),
        Line2D([0], [0], color="#222222", linewidth=1.4, linestyle="--", label="1.5-IQR envelope"),
    ])
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, prop={"weight": "bold", "size": 18})
    fig.suptitle("Figure 2 exploratory 3-D surface boxplot | Cr count × model × RMSE", fontsize=24, fontweight="bold", y=1.02)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.84, wspace=0.02)
    paths = {}
    for extension in ("png", "pdf", "svg"):
        path = out_dir / f"variant05_cr_count_surface_boxplot.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths[extension] = str(path)
    plt.close(fig)
    return paths


def build(out_dir: Path) -> dict[str, object]:
    v9._house_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _records()

    # Compact two-family temperature view, with extra vertical separation for
    # the three-comparison brackets used by the four-family view.
    v9.LAYOUT_SPACING = 1.55
    v9.BRACKET_STEP = 0.60
    temperature_outputs, temperature_significance = v9.clustered_box_figure(
        records,
        families=v9.MULTIT_FAMILIES,
        group_key="temperature",
        categories=v9.TEMP_CATEGORIES,
        labels=v9.TEMP_LABELS,
        xlabel="trajectory temperature class · train + test merged",
        title="Figure 2 revision | compact seed-resolved RMSE distributions by MultiT temperature",
        stem="variant04_by_temperature_class_compact_seed_boxclusters",
        out_dir=out_dir,
    )

    pooled = _pooled_cr_count_figure(records, out_dir)
    surface = _surface_figure(records, out_dir)
    trajectory_audit = v9._trajectory_atom_audit(records)
    manifest = {
        "status": "RECONSTRUCTION",
        "purpose": "compact temperature seed-resolved view, exploratory Cr-count surface boxplot, and seed-pooled Cr-count paper-facing view",
        "checkpoint_before_revision": "7303075",
        "data": {
            "source_mode": "readable_packaged_per_frame_csv",
            "frame_root": str(v9.FRAME_ROOT),
            "splits_merged": v9.base.SPLITS,
            "families": v9.FAMILIES,
            "seeds": v9.SEEDS,
            "cr_categories": v9.CR_CATEGORIES,
            "bulk_reference_atoms": v9.BULK_ATOMS,
            "trajectory_audit": trajectory_audit,
            "significance": {
                "comparisons": [
                    "Scratch vs FT · 600K",
                    "Scratch vs FT · MultiT",
                    "600K vs MultiT · FT",
                ],
                "test": "paired two-sided t-test on three seed-level log10 aggregates; exploratory n=3; no multiplicity correction",
            },
        },
        "outputs": {
            "temperature_compact": temperature_outputs,
            "cr_count_seed_pooled": pooled,
            "cr_count_surface_exploratory": surface,
        },
        "interpretation": {
            "surface_boxplot": "EXPLORATORY_NOT_SELECTED_AS_PRIMARY",
            "primary_cr_count_view": "seed-pooled 2-D boxplot with complete frame-level points",
            "reason": "The 3-D surface view is useful as a diagnostic overview but less precise for reading individual distributions, significance brackets, and exact model comparisons than the 2-D pooled view.",
        },
    }
    manifest_path = out_dir / "figure02_surface_and_pooled_v10_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_surface_and_pooled_v10_caption.md"
    caption_path.write_text(
        "# Figure 2 v10 compact and seed-pooled revisions\n\n"
        "The temperature panel uses compact, non-overlapping seed-resolved boxes with larger typography. The exploratory surface-boxplot maps Cr count, model family, and log10 RMSE into three dimensions; its median, IQR surfaces, and 1.5-IQR envelopes are retained as a diagnostic but are not selected as the primary paper-facing view. The primary Cr-count view pools all three seeds inside one box per model and Cr1–Cr6 class while retaining all frame-level points. Train and held-out-test records are merged. O3 t1 has 84 atoms and is therefore Cr4 under the 80-atom bulk reference; no common 80-atom no-Cr trajectory is available.\n"
    )
    return {"manifest": str(manifest_path), "caption": str(caption_path), "outputs": manifest["outputs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    print(json.dumps(build(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
