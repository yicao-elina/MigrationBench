"""Seed-resolved, fixed-path NEB error heatmaps with provenance checks.

This module deliberately keeps a fixed DFT image path separate from
self-relaxed MLFF-NEB.  It accepts one row per
model/checkpoint/route/image on an identical fixed geometry and produces:

* a seed-preserving error map plus a model-by-task performance bubble heatmap;
* a ranked signed-error map with overall image-wise and row-wise summaries;
* a normalized bubble similarity matrix using the reference figure's visual grammar.

The dendrogram is a similarity summary of error signatures.  It is not a
training-lineage or a statistical significance claim.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist

REQUIRED_COLUMNS = {
    "record_id",
    "model_family",
    "model_id",
    "model_label",
    "training_seed",
    "path_id",
    "path_label",
    "image_index",
    "reaction_coordinate",
    "model_relative_energy_eV",
    "dft_relative_energy_eV",
    "model_sha256",
    "path_geometry_sha256",
    "dft_path_geometry_sha256",
    "dft_reference_id",
    "dft_reference_status",
    "observable",
    "scientific_status",
    "publication_status",
    "source_profile_artifact",
    "dataset_status",
}
FIXED_PATH_OBSERVABLE = "FIXED_PATH_SINGLE_POINT"
FAMILY_COLORS = ("#1B5E8A", "#E17C05", "#2A9D8F", "#8C4FA4")
TASK_COLORS = ("#4C78A8", "#F58518", "#54A24B")
TOTAL_COLUMN_ORDER = [
    "migration_in_gap_fixed",
    "migration_gap_to_ql_fixed",
    "migration_within_ql_fixed",
    "migration_in_gap_self",
    "migration_gap_to_ql_self",
    "migration_within_ql_self",
    "sliding_pristine_fixed",
    "sliding_cr_in_gap_fixed",
]


@dataclass(frozen=True)
class InputAudit:
    rows: int
    model_families: int
    model_seed_units: int
    paths: int
    images: int
    dataset_statuses: list[str]
    fixed_path_only: bool


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=360, bbox_inches="tight", facecolor="white")
        if suffix == ".svg":
            # Matplotlib wraps SVG path coordinates on lines ending in spaces.
            # Normalize this generated text so a reproducibility commit passes
            # the repository's whitespace check without changing SVG geometry.
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _publication_style() -> None:
    """Apply the one-off figure style derived from the Plot Atlas contract."""
    mpl.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.linewidth": 1.05,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _ordered(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values}, key=lambda value: (not value.isdigit(), value))


def _family_order(frame: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(frame["model_family"].astype(str)))


def _path_order(frame: pd.DataFrame) -> list[str]:
    labels = frame[["path_id", "path_label"]].drop_duplicates("path_id")
    return list(labels["path_id"].astype(str))


def _family_palette(families: list[str]) -> dict[str, str]:
    if len(families) > len(FAMILY_COLORS):
        colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(families)))
        return {family: mpl.colors.to_hex(color) for family, color in zip(families, colors)}
    return {family: FAMILY_COLORS[index] for index, family in enumerate(families)}


def _task_palette(paths: list[str]) -> dict[str, str]:
    if len(paths) > len(TASK_COLORS):
        colors = plt.get_cmap("tab20c")(np.linspace(0.0, 1.0, len(paths)))
        return {path: mpl.colors.to_hex(color) for path, color in zip(paths, colors)}
    return {path: TASK_COLORS[index] for index, path in enumerate(paths)}


def read_and_validate(
    path: Path,
    *,
    expected_families: int | None = None,
    expected_seeds_per_family: int | None = None,
    expected_paths: int | None = None,
    allow_synthetic_demo: bool = False,
) -> tuple[pd.DataFrame, InputAudit]:
    """Read a tidy fixed-path profile table and reject non-comparable input."""
    frame = pd.read_csv(path, sep="\t")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"input lacks required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("input contains no profile rows")
    statuses = sorted(frame["dataset_status"].astype(str).unique())
    if any("SYNTHETIC" in status.upper() for status in statuses) and not allow_synthetic_demo:
        raise ValueError("synthetic input requires --allow-synthetic-demo")
    if set(frame["observable"].astype(str)) != {FIXED_PATH_OBSERVABLE}:
        observed = sorted(frame["observable"].astype(str).unique())
        raise ValueError(
            "only fixed-path DFT-retained scores are comparable here; observed "
            + ", ".join(observed)
        )
    if not (frame["path_geometry_sha256"].astype(str) == frame["dft_path_geometry_sha256"].astype(str)).all():
        raise ValueError("model and DFT path geometry hashes must match on every row")
    numerical = (
        "image_index",
        "reaction_coordinate",
        "model_relative_energy_eV",
        "dft_relative_energy_eV",
    )
    for column in numerical:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[list(numerical)].to_numpy(float)).all():
        raise ValueError("non-finite image coordinate or energy value")
    identity = ["model_id", "path_id", "image_index"]
    if frame.duplicated(identity).any():
        duplicates = frame.loc[frame.duplicated(identity, keep=False), identity]
        raise ValueError(f"duplicate model/path/image rows: {duplicates.head().to_dict('records')}")
    dft_keys = ["path_id", "image_index"]
    for column in ("dft_relative_energy_eV", "dft_path_geometry_sha256", "dft_reference_id", "reaction_coordinate"):
        if (frame.groupby(dft_keys)[column].nunique(dropna=False) != 1).any():
            raise ValueError(f"{column} must be identical across models for each fixed DFT image")
    family_by_model = frame.groupby("model_id")["model_family"].nunique()
    if (family_by_model != 1).any():
        raise ValueError("a model_id must belong to exactly one model_family")
    seed_by_model = frame.groupby("model_id")["training_seed"].nunique()
    if (seed_by_model != 1).any():
        raise ValueError("a model_id must have exactly one training_seed")
    route_grids = frame.groupby("path_id")["image_index"].apply(lambda value: tuple(sorted(value.unique())))
    for model_id, subframe in frame.groupby("model_id"):
        observed = subframe.groupby("path_id")["image_index"].apply(lambda value: tuple(sorted(value.unique())))
        if observed.to_dict() != route_grids.to_dict():
            raise ValueError(f"{model_id} does not cover the same image grid as all paths")
    units = frame[["model_family", "model_id", "training_seed"]].drop_duplicates()
    family_seed_counts = units.groupby(["model_family", "training_seed"]).size()
    if (family_seed_counts != 1).any():
        raise ValueError("each model_family/training_seed combination must map to one checkpoint")
    if expected_families is not None and units["model_family"].nunique() != expected_families:
        raise ValueError(f"expected {expected_families} model families, found {units['model_family'].nunique()}")
    if expected_seeds_per_family is not None:
        counts = units.groupby("model_family")["training_seed"].nunique()
        if not (counts == expected_seeds_per_family).all():
            raise ValueError(
                f"expected {expected_seeds_per_family} distinct training seeds per family; "
                f"observed {counts.to_dict()}"
            )
    if expected_paths is not None and frame["path_id"].nunique() != expected_paths:
        raise ValueError(f"expected {expected_paths} paths, found {frame['path_id'].nunique()}")
    audit = InputAudit(
        rows=len(frame),
        model_families=units["model_family"].nunique(),
        model_seed_units=len(units),
        paths=frame["path_id"].nunique(),
        images=int(frame.groupby("path_id")["image_index"].nunique().sum()),
        dataset_statuses=statuses,
        fixed_path_only=True,
    )
    family_order = list(dict.fromkeys(frame["model_family"].astype(str)))
    path_order = list(dict.fromkeys(frame["path_id"].astype(str)))
    frame = frame.assign(
        _family_order=pd.Categorical(frame["model_family"].astype(str), categories=family_order, ordered=True),
        _path_order=pd.Categorical(frame["path_id"].astype(str), categories=path_order, ordered=True),
    )
    frame = frame.sort_values(["_family_order", "training_seed", "_path_order", "image_index"])
    return frame.drop(columns=["_family_order", "_path_order"]), audit


def derive_tables(frame: pd.DataFrame, baseline_family: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive route metrics, image-wise errors, and seed-matched effects."""
    error_rows = frame.copy()
    error_rows["signed_error_eV"] = (
        error_rows["model_relative_energy_eV"] - error_rows["dft_relative_energy_eV"]
    )
    error_rows["absolute_error_eV"] = error_rows["signed_error_eV"].abs()
    rows: list[dict[str, object]] = []
    group_columns = ["model_family", "model_id", "model_label", "training_seed", "path_id", "path_label"]
    for keys, group in error_rows.groupby(group_columns, sort=False):
        model_profile = group["model_relative_energy_eV"].to_numpy(float)
        dft_profile = group["dft_relative_energy_eV"].to_numpy(float)
        error = group["signed_error_eV"].to_numpy(float)
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_images": len(group),
                "mae_eV": float(np.mean(np.abs(error))),
                "rmse_eV": float(np.sqrt(np.mean(error**2))),
                "signed_bias_eV": float(np.mean(error)),
                "forward_barrier_error_eV": float(np.max(model_profile) - np.max(dft_profile)),
            }
        )
    metrics = pd.DataFrame(rows)
    family_order = _family_order(frame)
    baseline = baseline_family or family_order[0]
    if baseline not in family_order:
        raise ValueError(f"baseline_family {baseline!r} is absent from input")
    baseline_rows = metrics.loc[metrics["model_family"] == baseline].set_index(["training_seed", "path_id"])
    effects: list[dict[str, object]] = []
    for _, row in metrics.iterrows():
        if row["model_family"] == baseline:
            continue
        key = (row["training_seed"], row["path_id"])
        if key not in baseline_rows.index:
            continue
        reference = baseline_rows.loc[key]
        effects.append(
            {
                "baseline_family": baseline,
                "comparison_family": row["model_family"],
                "training_seed": row["training_seed"],
                "path_id": row["path_id"],
                "path_label": row["path_label"],
                "mae_difference_eV": float(row["mae_eV"] - reference["mae_eV"]),
                "rmse_difference_eV": float(row["rmse_eV"] - reference["rmse_eV"]),
                "barrier_error_difference_eV": float(
                    row["forward_barrier_error_eV"] - reference["forward_barrier_error_eV"]
                ),
                "comparison_model_id": row["model_id"],
                "baseline_model_id": reference["model_id"],
            }
        )
    return error_rows, metrics, pd.DataFrame(effects)


def _draw_family_strip(axis: plt.Axes, families: list[str], colors: dict[str, str]) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(len(families), 0)
    for index, family in enumerate(families):
        axis.add_patch(Rectangle((0, index), 1, 1, facecolor=colors[family], edgecolor="white", linewidth=0.6))
    axis.set_axis_off()


def render_performance_heatmap(metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Render a fixed-order family-seed by route MAE heatmap."""
    _publication_style()
    families = _family_order(metrics)
    paths = _path_order(metrics)
    family_colors = _family_palette(families)
    seed_order = _ordered(metrics["training_seed"])
    unit_order = [
        unit
        for family in families
        for seed in seed_order
        for unit in [f"{family} · s{seed}"]
        if ((metrics["model_family"].astype(str) == family) & (metrics["training_seed"].astype(str) == seed)).any()
    ]
    indexed = metrics.assign(unit=lambda data: data["model_family"].astype(str) + " · s" + data["training_seed"].astype(str))
    values = indexed.pivot(index="unit", columns="path_id", values="mae_eV").reindex(index=unit_order, columns=paths)
    z_values = values.copy()
    for column in z_values:
        deviation = float(z_values[column].std(ddof=0))
        z_values[column] = 0.0 if deviation == 0 else (z_values[column] - z_values[column].mean()) / deviation
    limit = max(1.0, float(np.nanmax(np.abs(z_values.to_numpy(float)))))
    fig = plt.figure(figsize=(10.3, 6.25), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=(0.13, 0.24, 1.0, 0.055), wspace=0.04)
    ax_strip = fig.add_subplot(grid[0, 0])
    ax_labels = fig.add_subplot(grid[0, 1])
    ax = fig.add_subplot(grid[0, 2])
    cax = fig.add_subplot(grid[0, 3])
    row_families = [unit.split(" · s", maxsplit=1)[0] for unit in unit_order]
    _draw_family_strip(ax_strip, row_families, family_colors)
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(len(unit_order), 0)
    for index, label in enumerate(unit_order):
        ax_labels.text(1.0, index + 0.5, label, ha="right", va="center", fontsize=10.5, fontweight="bold")
    ax_labels.set_axis_off()
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    image = ax.imshow(z_values.to_numpy(float), cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
    path_labels = (
        metrics[["path_id", "path_label"]].drop_duplicates("path_id").set_index("path_id")["path_label"].reindex(paths)
    )
    ax.set_xticks(np.arange(len(paths)), list(path_labels))
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=9, length=0)
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, len(paths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(unit_order), 1), minor=True)
    ax.grid(which="minor", color="#E3E3E3", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for index, family in enumerate(row_families[:-1]):
        if family != row_families[index + 1]:
            ax.axhline(index + 0.5, color="#222222", linewidth=1.25)
    for row_index, unit in enumerate(unit_order):
        for column_index, path in enumerate(paths):
            value = values.loc[unit, path]
            contrast = "white" if abs(z_values.loc[unit, path]) > limit * 0.62 else "#151515"
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color=contrast, fontsize=9.4, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#222222")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Route-standardized MAE\n(lower is bluer)", fontsize=10.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=9.5, length=3)
    return _save_figure(fig, out_dir / "fig_neb_seed_performance_heatmap")


def _signature_matrix(errors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    label_lookup = errors[["model_id", "model_family", "training_seed"]].drop_duplicates().copy()
    label_lookup["unit"] = label_lookup["model_family"].astype(str) + " · s" + label_lookup["training_seed"].astype(str)
    pivot = errors.pivot(index="model_id", columns=["path_id", "image_index"], values="signed_error_eV")
    families = _family_order(errors)
    paths = _path_order(errors)
    columns = [column for path in paths for column in pivot.columns if column[0] == path]
    columns = sorted(columns, key=lambda value: (paths.index(value[0]), value[1]))
    pivot = pivot.reindex(columns=columns)
    unit_order = [
        row.model_id
        for family in families
        for seed in _ordered(errors["training_seed"])
        for row in label_lookup.itertuples()
        if row.model_family == family and str(row.training_seed) == seed
    ]
    return pivot.reindex(index=unit_order), label_lookup.set_index("model_id"), paths


def render_total_performance_heatmap(total_metrics: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Render the compact model-by-task score/coverage bubble heatmap."""
    _publication_style()
    required = {
        "row_id",
        "row_label",
        "model_kind",
        "column_id",
        "column_label",
        "task_block",
        "raw_value_eV",
        "score_01",
        "coverage_fraction",
    }
    missing = sorted(required - set(total_metrics.columns))
    if missing:
        raise ValueError(f"dot performance table lacks required columns: {', '.join(missing)}")
    column_order = [column for column in TOTAL_COLUMN_ORDER if column in set(total_metrics["column_id"])]
    row_order = list(dict.fromkeys(total_metrics["row_id"].astype(str)))
    table = total_metrics.copy()
    table["row_id"] = table["row_id"].astype(str)
    table["column_id"] = table["column_id"].astype(str)
    scores = table.pivot(index="row_id", columns="column_id", values="score_01").reindex(index=row_order, columns=column_order)
    coverage = table.pivot(index="row_id", columns="column_id", values="coverage_fraction").reindex(index=row_order, columns=column_order)
    row_labels = table.drop_duplicates("row_id").set_index("row_id")["row_label"].reindex(row_order)
    row_kinds = table.drop_duplicates("row_id").set_index("row_id")["model_kind"].reindex(row_order)
    labels = table.drop_duplicates("column_id").set_index("column_id")["column_label"].reindex(column_order).tolist()

    fig = plt.figure(figsize=(13.2, 6.9), layout="constrained")
    grid = fig.add_gridspec(
        3,
        5,
        height_ratios=(0.26, 0.22, 1.0),
        width_ratios=(1.85, 0.22, 6.0, 0.84, 0.16),
        hspace=0.02,
        wspace=0.035,
    )
    ax_size = fig.add_subplot(grid[0, 0:2])
    ax_color = fig.add_subplot(grid[0, 2:4])
    ax_group = fig.add_subplot(grid[1, 2:4])
    ax_row_header = fig.add_subplot(grid[1, 0])
    ax_labels = fig.add_subplot(grid[2, 0])
    ax = fig.add_subplot(grid[2, 2])
    ax_cbar = fig.add_subplot(grid[2, 3])

    # Reference-style top legend: circle area is coverage and color is a [0, 1] score.
    ax_size.set_xlim(0, 5.5)
    ax_size.set_ylim(0, 1)
    ax_size.axis("off")
    ax_size.text(0.0, 0.80, "Dot area = accepted / available replicates", ha="left", va="center", fontsize=9.8, fontweight="bold")
    for position, fraction, label in zip((1.7, 2.55, 3.45), (1 / 3, 2 / 3, 1.0), ("1/3", "2/3", "3/3")):
        ax_size.scatter(position, 0.34, s=42 + 330 * fraction, color="#111111", edgecolors="none")
        ax_size.text(position, 0.02, label, ha="center", va="bottom", fontsize=8.8)
    ax_size.text(4.25, 0.34, "Foundation cells use 1/1", ha="left", va="center", fontsize=8.8, color="#444444")

    ax_color.set_axis_off()
    gradient = np.linspace(1, 0, 256)[np.newaxis, :]
    ax_color.imshow(gradient, aspect="auto", cmap="plasma", extent=(0, 1, 0.28, 0.56))
    ax_color.text(0.0, 0.04, "High score", ha="left", va="bottom", fontsize=8.8)
    ax_color.text(1.0, 0.04, "Low score", ha="right", va="bottom", fontsize=8.8)
    ax_color.text(0.0, 0.83, "Color = column-normalized performance", ha="left", va="center", fontsize=9.8, fontweight="bold")

    ax_group.set_xlim(0, len(column_order))
    ax_group.set_ylim(0, 1)
    group_specs = [
        (0, 3, "Migration / fixed pathway"),
        (3, 6, "Migration / self-relaxed"),
        (6, 8, "Sliding"),
    ]
    for start, end, label in group_specs:
        if end <= len(column_order):
            ax_group.plot([start + 0.03, end - 0.03], [0.20, 0.20], color="#111111", linewidth=2.3, solid_capstyle="butt")
            ax_group.text((start + end) / 2, 0.72, label, ha="center", va="center", fontsize=10.2, fontweight="bold")
    ax_group.axis("off")
    ax_row_header.axis("off")
    ax_row_header.text(1.0, 0.20, "Model", ha="right", va="center", fontsize=10.0, fontweight="bold")

    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(len(row_order), 0)
    for index, label in enumerate(row_labels.tolist()):
        weight = "bold" if row_kinds.iloc[index] == "trained_family" else "normal"
        ax_labels.text(1.0, index + 0.5, label, ha="right", va="center", fontsize=10.2, fontweight=weight)
    ax_labels.axis("off")

    yy, xx = np.indices(scores.shape)
    score_values = scores.to_numpy(float)
    coverage_values = coverage.to_numpy(float)
    finite = np.isfinite(score_values)
    cmap = plt.get_cmap("plasma")
    if finite.any():
        ax.scatter(
            xx[finite],
            yy[finite],
            s=42 + 330 * coverage_values[finite],
            c=score_values[finite],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            edgecolors="none",
            alpha=0.96,
            zorder=3,
        )
    missing_cells = ~finite
    if missing_cells.any():
        ax.scatter(
            xx[missing_cells],
            yy[missing_cells],
            s=28 + 40 * np.nan_to_num(coverage_values[missing_cells]),
            facecolors="#E7E7E7",
            edgecolors="#B9B9B9",
            linewidths=0.9,
            zorder=2,
        )
    ax.set_xlim(-0.55, len(column_order) - 0.45)
    ax.set_ylim(len(row_order) - 0.55, -0.55)
    ax.set_xticks(np.arange(len(column_order)), labels, rotation=48, ha="right", rotation_mode="anchor", fontsize=9.4, fontweight="bold")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks([])
    ax.set_facecolor("white")
    for boundary in (2.5, 5.5):
        if boundary < len(column_order):
            ax.axvline(boundary, color="#111111", linewidth=1.35, zorder=4)
    if len(row_order) > 4:
        ax.axhline(3.5, color="#111111", linewidth=1.25, zorder=4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap),
        cax=ax_cbar,
    )
    colorbar.set_label("Performance score\n[0, 1]", fontsize=9.3, fontweight="bold")
    colorbar.ax.tick_params(labelsize=8.4, length=3)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    return _save_figure(fig, out_dir / "fig_neb_seed_performance_heatmap")


def render_lineage_clustermap(errors: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Render a ranked signed-error heatmap with an overall error summary row."""
    _publication_style()
    matrix, labels, paths = _signature_matrix(errors)
    raw = matrix.to_numpy(float)
    if raw.shape[0] < 3:
        raise ValueError("an error heatmap needs at least three model-seed units")
    row_mean_abs = np.mean(np.abs(raw), axis=1)
    row_order = np.argsort(-row_mean_abs, kind="stable")
    ordered = matrix.iloc[row_order]
    ordered_labels = labels.reindex(ordered.index)
    maximum = max(0.05, float(np.nanquantile(np.abs(raw), 0.98)))
    maximum = max(maximum, 0.10)
    overall_mean_abs = np.mean(np.abs(raw), axis=0)
    fig = plt.figure(figsize=(14.2, 7.4), layout="constrained")
    grid = fig.add_gridspec(
        3,
        6,
        height_ratios=(0.14, 0.18, 1.0),
        width_ratios=(0.12, 0.20, 1.4, 4.9, 0.64, 0.18),
        hspace=0.03,
        wspace=0.03,
    )
    ax_task = fig.add_subplot(grid[0, 3])
    ax_overall_label = fig.add_subplot(grid[1, 2])
    ax_overall = fig.add_subplot(grid[1, 3])
    ax_overall_cbar = fig.add_subplot(grid[1, 4])
    ax_family = fig.add_subplot(grid[2, 1])
    ax_labels = fig.add_subplot(grid[2, 2])
    ax = fig.add_subplot(grid[2, 3])
    ax_row_summary = fig.add_subplot(grid[2, 4])
    cax = fig.add_subplot(grid[2, 5])
    families = _family_order(errors)
    palette = _family_palette(families)
    row_families = list(ordered_labels["model_family"].astype(str))
    _draw_family_strip(ax_family, row_families, palette)
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(len(ordered), 0)
    for index, record in enumerate(ordered_labels.itertuples()):
        ax_labels.text(1.0, index + 0.5, record.unit, ha="right", va="center", fontsize=9.7, fontweight="bold")
    ax_labels.set_axis_off()
    norm = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
    image = ax.imshow(ordered.to_numpy(float), cmap="RdBu_r", norm=norm, interpolation="nearest", aspect="auto")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xticks(np.arange(-0.5, ordered.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
    ax.grid(which="minor", color="#D8D8D8", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    for index in range(len(row_families) - 1):
        if row_families[index] != row_families[index + 1]:
            ax.axhline(index + 0.5, color="#222222", linewidth=1.15)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#222222")
    task_colors = _task_palette(paths)
    ax_task.set_xlim(0, ordered.shape[1])
    ax_task.set_ylim(0, 1)
    start = 0
    for path in paths:
        count = sum(column[0] == path for column in ordered.columns)
        path_label = errors.loc[errors["path_id"] == path, "path_label"].iloc[0]
        ax_task.add_patch(Rectangle((start, 0.33), count, 0.32, facecolor=task_colors[path], edgecolor="white", linewidth=0.8))
        ax_task.text(start + count / 2, 0.93, path_label, ha="center", va="top", fontsize=10.3, fontweight="bold")
        ax.axvline(start - 0.5, color="#333333", linewidth=1.0)
        start += count
    ax.axvline(start - 0.5, color="#333333", linewidth=1.0)
    ax_task.set_axis_off()

    overall_image = ax_overall.imshow(overall_mean_abs[np.newaxis, :], cmap="OrRd", aspect="auto", interpolation="nearest")
    ax_overall.set_xticks([])
    ax_overall.set_yticks([0], [""], fontsize=8)
    ax_overall_label.set_axis_off()
    ax_overall_label.text(1.0, 0.5, "Overall mean\n|error| (eV)", ha="right", va="center", fontsize=9.1, fontweight="bold")
    overall_cbar = fig.colorbar(overall_image, cax=ax_overall_cbar, orientation="vertical")
    overall_cbar.ax.tick_params(labelsize=8, length=2)
    ax_row_summary.imshow(row_mean_abs[row_order, np.newaxis], cmap="OrRd", aspect="auto", interpolation="nearest")
    ax_row_summary.set_xticks([0], ["Mean\n|error|"], fontsize=8.8, fontweight="bold")
    ax_row_summary.xaxis.tick_top()
    ax_row_summary.tick_params(axis="x", length=0, pad=28)
    ax_row_summary.set_yticks([])
    for index, value in enumerate(row_mean_abs[row_order]):
        ax_row_summary.text(0, index, f"{value:.2f}", ha="center", va="center", fontsize=8.0, fontweight="bold")
    for spine in ax_row_summary.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#222222")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Signed relative-energy error (eV)", fontsize=10.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=9.3, length=3)
    return _save_figure(fig, out_dir / "fig_neb_seed_error_lineage_clustermap")


def _bubble_matrix(
    matrix: np.ndarray,
    labels: list[str],
    *,
    output: Path,
    title: str,
    subtitle: str,
    cmap: str = "RdBu",
    vmin: float = -1.0,
    vmax: float = 1.0,
    colorbar_label: str = "Correlation",
) -> list[Path]:
    _publication_style()
    count = len(labels)
    yy, xx = np.indices(matrix.shape)
    values = matrix.ravel()
    fig, ax = plt.subplots(figsize=(max(8.2, count * 0.62), max(7.2, count * 0.54)), layout="constrained")
    scale = (min(680, 16000 / max(count, 1)))
    scatter = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        s=np.square(np.abs(values)) * scale + 4,
        c=values,
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        marker="o",
        edgecolors="none",
        alpha=0.97,
    )
    ax.set_xticks(np.arange(count), labels, rotation=90, ha="center", fontsize=8.3)
    ax.set_yticks(np.arange(count), labels, fontsize=8.3)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=5)
    ax.set_xlim(-0.5, count - 0.5)
    ax.set_ylim(count - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, count, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, count, 1), minor=True)
    ax.grid(which="minor", color="#E1E1E1", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label(colorbar_label, fontsize=10.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=9.5, length=3)
    return _save_figure(fig, output)


def _normalized_similarity(errors: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    matrix, labels, _ = _signature_matrix(errors)
    correlations = np.corrcoef(matrix.to_numpy(float))
    off_diagonal = correlations[~np.eye(correlations.shape[0], dtype=bool)]
    minimum = float(np.nanmin(off_diagonal))
    maximum = float(np.nanmax(off_diagonal))
    if maximum <= minimum:
        similarity = np.ones_like(correlations)
    else:
        similarity = (correlations - minimum) / (maximum - minimum)
    np.fill_diagonal(similarity, 1.0)
    return similarity, list(labels.reindex(matrix.index)["unit"])


def render_similarity_bubble_matrix(errors: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Render Figure-1-style normalized error-signature similarity without stars."""
    similarity, display_labels = _normalized_similarity(errors)
    return _bubble_matrix(
        similarity,
        display_labels,
        output=out_dir / "fig_neb_seed_error_similarity_bubble_matrix",
        title="NEB error-signature similarity",
        subtitle="Circle area and color encode off-diagonal correlation normalized to [0, 1].",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Normalized correlation similarity [0, 1]",
    )


def render_reference_bubble_demo(out_dir: Path) -> list[Path]:
    """Recreate the first attachment's bubble-matrix grammar using deterministic mock values."""
    rng = np.random.default_rng(20261031)
    group = np.repeat(np.arange(4), 5)
    latent = rng.normal(size=(20, 6))
    latent += group[:, None] * np.array([1.2, -0.7, 0.5, -0.4, 0.25, -0.2])
    correlations = np.corrcoef(latent)
    return _bubble_matrix(
        correlations,
        [f"specimen {index + 1:02d}" for index in range(len(correlations))],
        output=out_dir / "fig_reference1_bubble_matrix_synthetic",
        title="Reference bubble-matrix grammar",
        subtitle="Deterministic synthetic illustration only; color and area encode correlation direction and magnitude.",
    )


def render_reference_clustered_heatmap_demo(out_dir: Path) -> list[Path]:
    """Recreate the second attachment's clustered-heatmap grammar with synthetic data."""
    _publication_style()
    rng = np.random.default_rng(20261031)
    row_group = np.repeat(np.arange(4), 12)
    col_group = np.repeat(np.arange(3), 6)
    matrix = rng.normal(0.0, 0.45, size=(48, 18))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if row_group[row] == col_group[column] % 4:
                matrix[row, column] += 1.45
            elif (row_group[row] - col_group[column]) % 3 == 0:
                matrix[row, column] -= 0.75
    row_linkage = linkage(pdist(matrix, metric="correlation"), method="average")
    col_linkage = linkage(pdist(matrix.T, metric="correlation"), method="average")
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)
    shown = matrix[np.ix_(row_order, col_order)]
    fig = plt.figure(figsize=(11.2, 8.1), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(0.95, 4.5), width_ratios=(1.25, 4.6, 0.16), hspace=0.03, wspace=0.03)
    ax_col = fig.add_subplot(grid[0, 1])
    ax_row = fig.add_subplot(grid[1, 0])
    ax = fig.add_subplot(grid[1, 1])
    cax = fig.add_subplot(grid[1, 2])
    dendrogram(col_linkage, no_labels=True, color_threshold=0, above_threshold_color="#666666", ax=ax_col)
    dendrogram(row_linkage, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="#666666", ax=ax_row)
    ax_col.set_axis_off()
    ax_row.set_axis_off()
    image = ax.imshow(shown, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5), aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Feature value (z score)", fontsize=10.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=9.5, length=3)
    return _save_figure(fig, out_dir / "fig_reference2_clustered_heatmap_synthetic")


def render_bundle(
    input_path: Path,
    output_dir: Path,
    processed_dir: Path,
    *,
    expected_families: int | None = 4,
    expected_seeds_per_family: int | None = 3,
    expected_paths: int | None = 3,
    baseline_family: str | None = None,
    allow_synthetic_demo: bool = False,
    total_metrics_path: Path | None = None,
    dot_metrics_path: Path | None = None,
) -> dict[str, object]:
    """Build all source tables and vector/raster figures from one input table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame, audit = read_and_validate(
        input_path,
        expected_families=expected_families,
        expected_seeds_per_family=expected_seeds_per_family,
        expected_paths=expected_paths,
        allow_synthetic_demo=allow_synthetic_demo,
    )
    errors, metrics, effects = derive_tables(frame, baseline_family)
    errors.to_csv(processed_dir / "NEB_IMAGEWISE_ENERGY_ERRORS.tsv", sep="\t", index=False)
    metrics.to_csv(processed_dir / "NEB_MODEL_SEED_ROUTE_METRICS.tsv", sep="\t", index=False)
    effects.to_csv(processed_dir / "NEB_SEED_MATCHED_FAMILY_EFFECTS.tsv", sep="\t", index=False)
    similarity, similarity_labels = _normalized_similarity(errors)
    pd.DataFrame(similarity, index=similarity_labels, columns=similarity_labels).to_csv(
        processed_dir / "NEB_ERROR_SIGNATURE_SIMILARITY_01.tsv", sep="\t"
    )
    if total_metrics_path is not None:
        total_metrics = pd.read_csv(total_metrics_path, sep="\t")
        total_metrics.to_csv(processed_dir / "NEB_TOTAL_PERFORMANCE_METRICS.tsv", sep="\t", index=False)
        if dot_metrics_path is not None:
            dot_metrics = pd.read_csv(dot_metrics_path, sep="\t")
            dot_metrics.to_csv(processed_dir / "NEB_TOTAL_PERFORMANCE_DOT_METRICS.tsv", sep="\t", index=False)
            performance_products = render_total_performance_heatmap(dot_metrics, output_dir)
        else:
            performance_products = render_total_performance_heatmap(total_metrics, output_dir)
    else:
        performance_products = render_performance_heatmap(metrics, output_dir)
    additional_inputs: list[dict[str, object]] = []
    if total_metrics_path is not None:
        total_metrics = pd.read_csv(total_metrics_path, sep="\t")
        source_cells = [
            token
            for cell in total_metrics["source_artifact"].dropna().astype(str).unique()
            for token in cell.split(";")
            if token
        ]
        sources = [total_metrics_path, *source_cells]
        if dot_metrics_path is not None:
            dot_metrics = pd.read_csv(dot_metrics_path, sep="\t")
            dot_source_cells = [
                token
                for cell in dot_metrics["source_artifact"].dropna().astype(str).unique()
                for token in cell.split(";")
                if token
            ]
            sources.extend([dot_metrics_path, *dot_source_cells])
        for source in dict.fromkeys(sources):
            path = Path(source)
            additional_inputs.append(
                {
                    "path": str(path.resolve()) if path.exists() else str(path),
                    "exists": path.exists(),
                    "sha256": _sha256(path) if path.is_file() else None,
                }
            )
    products = (
        render_reference_bubble_demo(output_dir)
        + render_reference_clustered_heatmap_demo(output_dir)
        + performance_products
        + render_lineage_clustermap(errors, output_dir)
        + render_similarity_bubble_matrix(errors, output_dir)
    )
    manifest = {
        "schema_version": "migrationbench_neb_seed_paired_heatmap_v2",
        "code": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "input": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        "additional_inputs": additional_inputs,
        "input_audit": asdict(audit),
        "analytic_boundary": (
            "The sliding block is fixed-path DFT-matched MAE. The migration block is "
            "a separately sourced barrier-proxy summary; fixed-path and self-relaxed "
            "migration values are never pooled or called DFT errors."
        ),
        "total_performance_visual_contract": (
            "Rows are model families/checkpoint variants. Columns are ordered migration "
            "fixed pathway, migration self-relaxed, and sliding. Color is a per-column "
            "lower-is-better min-max score in [0, 1]; circle area is accepted/available "
            "replicate coverage. Main trained-family values are medians over seeds 123/234/345."
        ),
        "cluster_interpretation": (
            "rows are sorted by mean absolute image-wise error; the descriptive map "
            "is not causal training lineage"
        ),
        "products": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in products],
        "processed_tables": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in sorted(processed_dir.glob("NEB_*.tsv"))
        ],
    }
    manifest_path = output_dir / "NEB_SEED_FIGURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
