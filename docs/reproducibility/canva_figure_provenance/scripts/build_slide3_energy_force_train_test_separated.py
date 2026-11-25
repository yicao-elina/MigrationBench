#!/usr/bin/env python3
"""Build separate train/test energy and force parity rows for Slide 3.

The four row blocks are, in order: energy train, energy test, force train,
and force test.  Every finite observation contributes to a vector hexbin;
train and test are never overlaid in the same panel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_slide3_nature_density as base  # noqa: E402
import build_slide3_energy_train_test_cluster_centered_heatmap as energy_source  # noqa: E402

OUT_ROOT = base.OUT_ROOT.parent / "R04_slide3_energy_force_train_test_separated_v5"
FONT = base.FONT
FONT_INCREMENT_PT = 4
MIN_TEXT = base.MIN_TEXT + FONT_INCREMENT_PT
TITLE_SIZE = base.TITLE_SIZE + FONT_INCREMENT_PT
PANEL_TITLE = base.PANEL_TITLE + FONT_INCREMENT_PT
AXIS_LABEL = base.AXIS_LABEL + FONT_INCREMENT_PT
TICK_SIZE = base.TICK_SIZE + FONT_INCREMENT_PT
ANNOTATION = base.ANNOTATION + FONT_INCREMENT_PT
LEGEND_SIZE = base.LEGEND_SIZE + FONT_INCREMENT_PT
COLORBAR_SIZE = base.COLORBAR_SIZE + FONT_INCREMENT_PT
COLORS = base.COLORS
FAMILIES = base.FAMILIES
SEEDS = base.SEEDS
TRAIN_ALPHA = 0.50
TEST_ALPHA = 0.90
FORCE_VIEW_LIMIT = 10.0
FAMILY_PREFIX = energy_source.FAMILY_PREFIX


def style_axis(ax: plt.Axes) -> None:
    energy_source.style_axis(ax)
    ax.tick_params(axis="both", which="major", direction="out", length=7, width=1.35, pad=6, labelsize=TICK_SIZE)
    for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        tick.set_fontfamily(FONT)
        tick.set_fontsize(TICK_SIZE)
        tick.set_fontweight("bold")


def load_force_family_split(family: str, split: str) -> dict[str, np.ndarray]:
    refs: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    prefix = FAMILY_PREFIX[family]
    for seed in SEEDS:
        npz_path = base.RAW_ROOT / f"{prefix}_seed{seed}_{split}__parity.npz"
        with np.load(npz_path, allow_pickle=False) as data:
            ref = np.asarray(data["force_ref_eV_A"], dtype=float)
            pred = np.asarray(data["force_pred_eV_A"], dtype=float)
        if ref.shape != pred.shape:
            raise RuntimeError(f"force reference/prediction shape mismatch: {npz_path.name}")
        finite = np.isfinite(ref) & np.isfinite(pred)
        refs.append(ref[finite])
        preds.append(pred[finite])
    return {"x": np.concatenate(refs), "y": np.concatenate(preds)}


def finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    return np.asarray(x[finite], dtype=float), np.asarray(y[finite], dtype=float)


def load_panels() -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, dict[str, dict[str, int]]], dict[str, dict[str, dict[str, dict[str, float]]]]]:
    panels: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    stats: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for family in FAMILIES:
        train_energy = energy_source.load_family_split(family, "train")
        test_energy = energy_source.load_family_split(family, "test")
        normalized, _, _, _ = energy_source.shared_cluster_center(train_energy, test_energy)
        panels[family] = {"energy": {}, "force": {}}
        counts[family] = {"energy": {}, "force": {}}
        stats[family] = {"energy": {}, "force": {}}
        for split in ("train", "test"):
            key = f"{split}_x"
            energy_x, energy_y = finite_xy(normalized[f"{split}_x"], normalized[f"{split}_y"])
            force = load_force_family_split(family, split)
            force_x, force_y = finite_xy(force["x"], force["y"])
            panels[family]["energy"][split] = {"x": energy_x, "y": energy_y}
            panels[family]["force"][split] = {"x": force_x, "y": force_y}
            counts[family]["energy"][split] = int(energy_x.size)
            counts[family]["force"][split] = int(force_x.size)
            stats[family]["energy"][split] = {
                "spearman": float(base.spearman(energy_x, energy_y)),
                "pearson": float(base.pearson(energy_x, energy_y)),
            }
            stats[family]["force"][split] = {
                "spearman": float(base.spearman(force_x, force_y)),
                "pearson": float(base.pearson(force_x, force_y)),
            }
    return panels, counts, stats


def metric_extent(panels: dict[str, dict[str, dict[str, np.ndarray]]], metric: str) -> tuple[float, float, float, float]:
    if metric == "force":
        return (-FORCE_VIEW_LIMIT, FORCE_VIEW_LIMIT, -FORCE_VIEW_LIMIT, FORCE_VIEW_LIMIT)
    values: list[np.ndarray] = []
    for family in FAMILIES:
        for split in ("train", "test"):
            values.extend((panels[family][metric][split]["x"], panels[family][metric][split]["y"]))
    merged = np.concatenate(values)
    lo = float(np.min(merged))
    hi = float(np.max(merged))
    if metric == "force":
        bound = max(abs(lo), abs(hi))
        lo, hi = -bound, bound
    span = hi - lo
    pad = 0.08 * span if span > 0 else 0.02
    return (lo - pad, hi + pad, lo - pad, hi + pad)


def metric_norm(panels: dict[str, dict[str, dict[str, np.ndarray]]], metric: str, extent: tuple[float, float, float, float]) -> LogNorm:
    max_count = 1.0
    for family in FAMILIES:
        for split in ("train", "test"):
            panel = panels[family][metric][split]
            counts = energy_source.actual_hexbin_counts(panel["x"], panel["y"], extent, gridsize=48)
            if counts.size:
                max_count = max(max_count, float(np.max(counts)))
    return LogNorm(vmin=1.0, vmax=max(2.0, max_count))


def add_vector_colorbar(fig: plt.Figure, axes: np.ndarray, norm: LogNorm, label: str) -> None:
    top = axes[0, -1].get_position().y1
    bottom = axes[-1, -1].get_position().y0
    cax = fig.add_axes([0.925, bottom, 0.017, top - bottom])
    cax.set_yscale("log")
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(norm.vmin, norm.vmax)
    cmap = mpl.colormaps["magma"]
    edges = np.geomspace(norm.vmin, norm.vmax, 72)
    for lower, upper in zip(edges[:-1], edges[1:]):
        cax.add_patch(
            mpl.patches.Rectangle(
                (0.0, lower),
                1.0,
                upper - lower,
                facecolor=cmap(norm(np.sqrt(lower * upper))),
                edgecolor="none",
                transform=cax.transData,
            )
        )
    cax.xaxis.set_visible(False)
    cax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    cax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    cax.yaxis.tick_right()
    cax.tick_params(axis="y", which="major", labelsize=COLORBAR_SIZE, width=1.0, length=5, pad=5)
    cax.set_ylabel(label, fontsize=COLORBAR_SIZE, fontweight="bold", labelpad=32, rotation=270)
    cax.yaxis.set_label_position("right")
    for tick in cax.get_yticklabels():
        tick.set_fontfamily(FONT)
        tick.set_fontsize(COLORBAR_SIZE)
        tick.set_fontweight("bold")
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
        spine.set_color("#111111")


def render(
    panels: dict[str, dict[str, dict[str, np.ndarray]]],
    counts: dict[str, dict[str, dict[str, int]]],
    stats: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    out_root: Path,
) -> tuple[dict[str, str], dict[str, list[float]], dict[str, dict[str, float]], dict[str, dict[str, dict[str, int]]]]:
    extents = {metric: metric_extent(panels, metric) for metric in ("energy", "force")}
    norms = {metric: metric_norm(panels, metric, extents[metric]) for metric in ("energy", "force")}
    visible_counts: dict[str, dict[str, dict[str, int]]] = {}
    for family in FAMILIES:
        visible_counts[family] = {"energy": {}, "force": {}}
        for metric in ("energy", "force"):
            extent = extents[metric]
            for split in ("train", "test"):
                panel = panels[family][metric][split]
                visible = (
                    (panel["x"] >= extent[0])
                    & (panel["x"] <= extent[1])
                    & (panel["y"] >= extent[2])
                    & (panel["y"] <= extent[3])
                )
                visible_counts[family][metric][split] = int(np.count_nonzero(visible))
    fig = plt.figure(figsize=(21.2, 21.5))
    grid = fig.add_gridspec(
        5,
        4,
        left=0.095,
        right=0.905,
        bottom=0.085,
        top=0.935,
        wspace=0.030,
        hspace=0.16,
        height_ratios=[1.0, 1.0, 0.42, 1.0, 1.0],
    )
    axes = np.empty((4, 4), dtype=object)
    grid_rows = (0, 1, 3, 4)
    for row, grid_row in enumerate(grid_rows):
        for col in range(4):
            axes[row, col] = fig.add_subplot(grid[grid_row, col])
    row_specs = (("energy", "train"), ("energy", "test"), ("force", "train"), ("force", "test"))
    for row, (metric, split) in enumerate(row_specs):
        for col, family in enumerate(FAMILIES):
            ax = axes[row, col]
            panel = panels[family][metric][split]
            extent = extents[metric]
            alpha = TRAIN_ALPHA if split == "train" else TEST_ALPHA
            ax.hexbin(
                panel["x"],
                panel["y"],
                gridsize=48,
                extent=extent,
                mincnt=1,
                cmap="magma",
                norm=norms[metric],
                alpha=alpha,
                linewidths=0.0,
                edgecolors="none",
                rasterized=False,
            )
            ax.plot(
                [extent[0], extent[1]],
                [extent[0], extent[1]],
                linestyle=(0, (6, 4)),
                color="#cf2e2e",
                linewidth=2.0,
                zorder=3,
            )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box")
            if row == 0:
                ax.set_title(
                    f"{chr(65 + col)}  {family}",
                    loc="left",
                    color=COLORS[family],
                    fontsize=PANEL_TITLE,
                    fontweight="bold",
                    pad=12,
                )
            note = (
                f"n={counts[family][metric][split]:,}\n"
                f"ρ={stats[family][metric][split]['spearman']:.2f}, r={stats[family][metric][split]['pearson']:.2f}"
            )
            ax.text(
                0.045,
                0.95,
                note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=ANNOTATION,
                linespacing=1.12,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3.0},
            )
            style_axis(ax)
            if col > 0:
                ax.tick_params(labelleft=False)
            if row not in (1, 3):
                ax.tick_params(labelbottom=False)
        row_label = split
        y_center = float((axes[row, 0].get_position().y0 + axes[row, 0].get_position().y1) / 2.0)
        fig.text(0.058, y_center, row_label, ha="center", va="center", rotation=90, fontsize=AXIS_LABEL, fontweight="bold")

    energy_y = r"MLIP $E/N-\langle E/N\rangle_{\mathrm{cluster}}$ (eV atom$^{-1}$)"
    energy_x = r"DFT $E/N-\langle E/N\rangle_{\mathrm{cluster}}$ (eV atom$^{-1}$)"
    force_y = r"MLIP force $F$ (eV Å$^{-1}$)"
    force_x = r"DFT force $F$ (eV Å$^{-1}$)"
    energy_center_y = float(
        (axes[0, 0].get_position().y1 + axes[1, 0].get_position().y0) / 2.0
    )
    force_center_y = float(
        (axes[2, 0].get_position().y1 + axes[3, 0].get_position().y0) / 2.0
    )
    energy_x_y = float(axes[1, 0].get_position().y0 - 0.032)
    force_x_y = float(axes[3, 0].get_position().y0 - 0.028)
    fig.text(0.020, energy_center_y, energy_y, ha="center", va="center", rotation=90, fontsize=AXIS_LABEL, fontweight="bold")
    fig.text(0.020, force_center_y, force_y, ha="center", va="center", rotation=90, fontsize=AXIS_LABEL, fontweight="bold")
    fig.text(0.5, energy_x_y, energy_x, ha="center", va="center", fontsize=AXIS_LABEL, fontweight="bold")
    fig.text(0.5, force_x_y, force_x, ha="center", va="center", fontsize=AXIS_LABEL, fontweight="bold")

    fig.text(
        0.5,
        0.985,
        "DFT reference vs MLIP prediction | complete energy and force train/test parity",
        ha="center",
        va="top",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    handles = [
        Patch(facecolor=mpl.colormaps["magma"](0.62), edgecolor="none", alpha=TRAIN_ALPHA, label="train"),
        Patch(facecolor=mpl.colormaps["magma"](0.62), edgecolor="none", alpha=TEST_ALPHA, label="test"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        handletextpad=0.55,
        columnspacing=1.6,
    )
    add_vector_colorbar(fig, axes[0:2, :], norms["energy"], "Energy count per hexbin")
    add_vector_colorbar(fig, axes[2:4, :], norms["force"], "Force count per hexbin")

    stem = "slide3_energy_force_train_test_separated"
    paths = {ext: out_root / "figures" / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")}
    paths["png"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["png"], dpi=320, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.18, facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)
    return (
        {key: str(value) for key, value in paths.items()},
        {key: list(value) for key, value in extents.items()},
        {key: {"vmin": float(value.vmin), "vmax": float(value.vmax)} for key, value in norms.items()},
        visible_counts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    base.setup_style()
    mpl.rcParams.update({"font.size": MIN_TEXT, "axes.labelsize": AXIS_LABEL, "axes.titlesize": PANEL_TITLE})
    panels, counts, stats = load_panels()
    outputs, extents, norms, visible_counts = render(panels, counts, stats, out_root=out_root)
    manifest = {
        "render_revision": "R04_energy_force_train_test_separated_v5",
        "publication_status": "RECONSTRUCTION",
        "scientific_scope": "within-cluster energy parity plus direct force-component parity, with train and held-out test rendered in separate rows",
        "processing_script": str(Path(__file__).resolve()),
        "processing_script_sha256": base.sha256(Path(__file__).resolve()),
        "data_root": str(base.DATA_ROOT),
        "raw_root": str(base.RAW_ROOT),
        "row_order": ["energy train", "energy test", "force train", "force test"],
        "input_matrix": "24 parity NPZ outputs plus matching per-frame CSV labels: 4 families x 3 seeds x train/test",
        "energy_normalization": "within each family and trajectory cluster, subtract the pooled train+test DFT E/N mean from both train and test DFT/MLIP E/N",
        "force_definition": "direct DFT and MLIP force-component parity from force_ref_eV_A and force_pred_eV_A; all finite components are included",
        "complete_data": "all finite energy frames and force components are read; energy is fully in view, while force is intentionally displayed on the requested -10 to +10 eV Å^-1 window and out-of-window components are counted explicitly",
        "rendering": "train and held-out test are never overlaid in the same panel; each row uses a separate split-specific magma hexbin; force display is fixed to [-10, 10] eV Å^-1 on both axes",
        "layout": {
            "blocks": ["energy", "force"],
            "within_block_rows": "train and test rows are compactly spaced",
            "between_block_spacing": "dedicated spacer row separates energy and force blocks",
            "shared_axis_labels": "one shared x and y label per metric block; train/test row labels retained",
        },
        "train_alpha": TRAIN_ALPHA,
        "test_alpha": TEST_ALPHA,
        "font_increment_pt": FONT_INCREMENT_PT,
        "column_spacing_target": "visible square-panel gaps are reduced to approximately the within-Energy-row gap by narrowing the canvas while preserving equal panel aspect",
        "families": FAMILIES,
        "counts": counts,
        "visible_counts": visible_counts,
        "force_out_of_window_counts": {
            family: {
                split: counts[family]["force"][split] - visible_counts[family]["force"][split]
                for split in ("train", "test")
            }
            for family in FAMILIES
        },
        "statistics": stats,
        "limits": extents,
        "count_norm": norms,
        "min_text_size_pt": MIN_TEXT,
        "outputs": outputs,
        "input_hashes": {
            str(path): base.sha256(path)
            for path in [base.SUMMARY_PATH, *sorted(base.RAW_ROOT.glob("*_parity.npz")), *sorted(base.RAW_ROOT.glob("*_per_frame.csv"))]
            if path.exists()
        },
    }
    manifest["output_hashes"] = {key: base.sha256(Path(value)) for key, value in outputs.items()}
    manifest_path = out_root / "slide3_energy_force_train_test_separated_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"outputs": outputs, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
