"""Seed-resolved, fixed-path NEB error heatmaps with provenance checks.

This module deliberately keeps a fixed DFT image path separate from
self-relaxed MLFF-NEB.  It accepts one row per
model/checkpoint/route/image on an identical fixed geometry and produces:

* a seed-preserving task-performance heatmap (four families x three seeds);
* a descriptive, hierarchical similarity map of the full image-wise errors;
* a bubble similarity matrix using the reference figure's visual grammar.

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


def render_lineage_clustermap(errors: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Render a correlation-distance dendrogram of image-wise signed errors."""
    _publication_style()
    matrix, labels, paths = _signature_matrix(errors)
    raw = matrix.to_numpy(float)
    if raw.shape[0] < 3:
        raise ValueError("a lineage heatmap needs at least three model-seed units")
    if np.any(np.std(raw, axis=1) == 0):
        raise ValueError("cannot cluster a constant image-wise error signature")
    row_linkage = linkage(pdist(raw, metric="correlation"), method="average")
    row_order = leaves_list(row_linkage)
    ordered = matrix.iloc[row_order]
    ordered_labels = labels.reindex(ordered.index)
    maximum = max(0.05, float(np.nanquantile(np.abs(raw), 0.98)))
    maximum = max(maximum, 0.10)
    fig = plt.figure(figsize=(14.0, 7.0), layout="constrained")
    grid = fig.add_gridspec(
        3,
        5,
        height_ratios=(0.13, 0.07, 1.0),
        width_ratios=(0.95, 0.12, 0.20, 4.6, 0.16),
        hspace=0.03,
        wspace=0.03,
    )
    ax_task = fig.add_subplot(grid[0, 3])
    ax_dendrogram = fig.add_subplot(grid[2, 0])
    ax_family = fig.add_subplot(grid[2, 1])
    ax_labels = fig.add_subplot(grid[2, 2])
    ax = fig.add_subplot(grid[2, 3])
    cax = fig.add_subplot(grid[2, 4])
    dendrogram(row_linkage, orientation="left", no_labels=True, color_threshold=0, above_threshold_color="#555555", ax=ax_dendrogram)
    ax_dendrogram.invert_yaxis()
    ax_dendrogram.set_axis_off()
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
        cmap="RdBu",
        norm=Normalize(vmin=-1, vmax=1),
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
    colorbar.set_label("Correlation", fontsize=10.5, fontweight="bold")
    colorbar.ax.tick_params(labelsize=9.5, length=3)
    return _save_figure(fig, output)


def render_similarity_bubble_matrix(errors: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Render Figure-1-style error-signature correlations without p-value stars."""
    matrix, labels, _ = _signature_matrix(errors)
    correlations = np.corrcoef(matrix.to_numpy(float))
    display_labels = list(labels.reindex(matrix.index)["unit"])
    return _bubble_matrix(
        correlations,
        display_labels,
        output=out_dir / "fig_neb_seed_error_similarity_bubble_matrix",
        title="NEB error-signature similarity",
        subtitle="Circle area is |r|; color is r. No significance symbols: NEB images are not independent samples.",
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
    products = (
        render_reference_bubble_demo(output_dir)
        + render_reference_clustered_heatmap_demo(output_dir)
        + render_performance_heatmap(metrics, output_dir)
        + render_lineage_clustermap(errors, output_dir)
        + render_similarity_bubble_matrix(errors, output_dir)
    )
    manifest = {
        "schema_version": "migrationbench_neb_seed_paired_heatmap_v1",
        "code": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "input": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        "input_audit": asdict(audit),
        "analytic_boundary": (
            "Fixed-path DFT-retained image scores only. Self-relaxed MLFF-NEB and "
            "fixed-path values are never pooled."
        ),
        "cluster_interpretation": "descriptive error-signature similarity, not causal training lineage",
        "products": [{"path": str(path.resolve()), "sha256": _sha256(path)} for path in products],
        "processed_tables": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in sorted(processed_dir.glob("NEB_*.tsv"))
        ],
    }
    manifest_path = output_dir / "NEB_SEED_FIGURE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
