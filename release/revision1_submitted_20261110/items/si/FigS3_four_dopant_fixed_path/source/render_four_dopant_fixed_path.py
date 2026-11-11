#!/usr/bin/env python3
"""Re-render the four-dopant, three-path Foundation fixed-path SI figure.

Run from the repository root:
    python analysis/four_dopant_fixed_path_palette_20261109/render_four_dopant_fixed_path.py

The plotting inputs are the local R03 aggregate and screened QE tables copied
to source_data/. No energies or image coordinates are recomputed here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


PALETTE_REL = "reproducibility/canva_palette_vector_bundle_20261105/MIGRATIONBENCH_FIGURE_PALETTE.json"
# Repository root: the first parent holding the shared palette (the script moved from analysis/ into
# release/.../items/si/FigS3_four_dopant_fixed_path/source/, so a fixed parents[N] no longer works).
ROOT = next(p for p in Path(__file__).resolve().parents if (p / PALETTE_REL).is_file())
HERE = Path(__file__).resolve().parent
DATA = HERE / "source_data"
# FOUR_DOPANT_OUT (a file stem) redirects the figures and the render manifest; default is the committed copy.
OUT = Path(os.environ.get("FOUR_DOPANT_OUT", str(HERE / "figures" / "fig_NEB-fixed-path_Cu-Li-Na-Cr_palette_updated")))
MANIFEST = OUT.parent / "RENDER_MANIFEST.json" if "FOUR_DOPANT_OUT" in os.environ else HERE / "RENDER_MANIFEST.json"
PALETTE_PATH = ROOT / PALETTE_REL
PALETTE = json.loads(PALETTE_PATH.read_text())

ELEMENTS = ["Cr", "Cu", "Li", "Na"]
PATHS = [
    ("within_gap", "in-gap diffusion"),
    ("gap_to_ql", "gap→QL"),
    ("within_ql", "within-QL"),
]
MODEL_CLASSES = ["MACE-MP", "MACE-OMAT-0", "MACE-MH"]
MODEL_COLORS = PALETTE["foundation_model_classes"]
MODEL_MARKERS = {"MACE-MP": "o", "MACE-OMAT-0": "^", "MACE-MH": "s"}
MODEL_STYLES = {"MACE-MP": "-", "MACE-OMAT-0": "--", "MACE-MH": "-."}
PATH_COLORS = {key: val["color"] for key, val in PALETTE["path_class_heading_accents"].items()}
INK = PALETTE["model_families"]["DFT_reference"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": PALETTE["typography"]["font_family"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(INK)
        spine.set_linewidth(1.05)
    ax.tick_params(direction="out", width=1.05, length=4.2, labelsize=11.2, colors=INK)
    ax.minorticks_on()


def main() -> None:
    grouped_path = DATA / "foundation_model_class_profiles_4x3_R03.tsv"
    qe_path = DATA / "foundation_model_qe_reference_profiles_4x3_R03.tsv"
    grouped = read_tsv(grouped_path)
    qe_rows = read_tsv(qe_path)

    if set(row["element"] for row in grouped) != set(ELEMENTS):
        raise ValueError("Unexpected elements in grouped source data")
    if set(row["path_key"] for row in grouped) != {item[0] for item in PATHS}:
        raise ValueError("Unexpected path classes in grouped source data")
    if set(row["model_class"] for row in grouped) != set(MODEL_CLASSES):
        raise ValueError("Unexpected model classes in grouped source data")

    fig, axs = plt.subplots(4, 3, figsize=(12.0, 12.6), squeeze=False)
    panel = iter("abcdefghijkl")
    for row_index, element in enumerate(ELEMENTS):
        for col_index, (path_key, path_label) in enumerate(PATHS):
            ax = axs[row_index, col_index]
            style_axes(ax)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, .25, .5, .75, 1], ["0", ".25", ".5", ".75", "1"])
            ax.text(
                -0.16, 1.055, next(panel), transform=ax.transAxes,
                fontsize=14.5, fontweight="bold", color=INK, va="bottom",
            )
            all_y: list[np.ndarray] = []
            for model_class in MODEL_CLASSES:
                records = [
                    r for r in grouped
                    if r["element"] == element and r["path_key"] == path_key and r["model_class"] == model_class
                ]
                records.sort(key=lambda r: int(r["image_index"]))
                if not records:
                    raise ValueError(f"Missing data: {element} / {path_key} / {model_class}")
                x = np.asarray([float(r["s"]) for r in records])
                y = np.asarray([float(r["mean_relative_energy_eV"]) for r in records])
                sd = np.asarray([float(r["sd_relative_energy_eV"] or 0) for r in records])
                color = MODEL_COLORS[model_class]
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=.20, linewidth=0, zorder=2)
                ax.plot(
                    x, y, color=color, linewidth=2.15, linestyle=MODEL_STYLES[model_class],
                    marker=MODEL_MARKERS[model_class], markersize=5.0,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.0, zorder=5,
                )
                all_y.extend([y - sd, y + sd])

            qe = [r for r in qe_rows if r["element"] == element and r["path_key"] == path_key]
            qe.sort(key=lambda r: int(r["image_index"]))
            if qe:
                xq = np.asarray([float(r["s"]) for r in qe])
                yq = np.asarray([float(r["relative_energy_eV"]) for r in qe])
                ax.plot(
                    xq, yq, color=INK, linewidth=1.8, linestyle=(0, (4, 2)),
                    marker="D", markersize=4.3, markerfacecolor="white",
                    markeredgecolor=INK, markeredgewidth=.9, zorder=6,
                )
                all_y.append(yq)

            if not all_y:
                raise ValueError(f"No plotted values for {element} / {path_key}")
            vals = np.concatenate(all_y)
            low, high = float(np.nanmin(vals)), float(np.nanmax(vals))
            pad = max(.06 * (high - low), .04)
            ax.set_ylim(low - pad, high + pad)

            if row_index == 0:
                ax.set_title(
                    path_label, fontsize=15.5, fontweight="bold", color=PATH_COLORS[path_key], pad=10,
                )
            if col_index == 0:
                ax.set_ylabel(f"{element}\nΔE (eV)", fontsize=13.3, fontweight="bold", labelpad=8, color=INK)
            if row_index == len(ELEMENTS) - 1:
                ax.set_xlabel("Normalized image coordinate, s", fontsize=12.7, fontweight="bold", labelpad=8, color=INK)

    handles = [
        Line2D(
            [], [], color=MODEL_COLORS[name], linewidth=2.15, linestyle=MODEL_STYLES[name],
            marker=MODEL_MARKERS[name], markerfacecolor="white", markeredgecolor=MODEL_COLORS[name],
            markersize=6.0, label=name,
        )
        for name in MODEL_CLASSES
    ]
    handles.append(
        Line2D([], [], color=INK, linewidth=1.8, linestyle=(0, (4, 2)), marker="D",
               markerfacecolor="white", markeredgecolor=INK, markersize=5.2,
               label="Screened QE path table")
    )
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(.5, .995), ncol=4,
        frameon=False, fontsize=11.5, handlelength=2.5, columnspacing=1.6, handletextpad=.6,
    )
    fig.subplots_adjust(left=.095, right=.985, bottom=.075, top=.925, wspace=.34, hspace=.46)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"), dpi=350 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    original_pdf = ROOT / "output/revision1_camera_ready/paper/revision1/figures/main/SI/fig_NEB-fixed-path_Cu-Li-Na-Cr.pdf"
    manifest = {
        "figure_id": "fig_NEB-fixed-path_Cu-Li-Na-Cr",
        "status": "RENDERED_FROM_LOCAL_R03_AGGREGATE_AND_SCREENED_QE_TABLES_WITH_PALETTE_AND_FONT_UPDATE",
        "original_figure_sha256": sha256(original_pdf) if original_pdf.exists() else None,
        "source_script": "analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/build_foundation_class_4x3_r03.py",
        "source_data": {
            grouped_path.name: sha256(grouped_path),
            qe_path.name: sha256(qe_path),
        },
        "palette_reference": str(PALETTE_PATH.relative_to(ROOT)),
        "path_heading_colors": PATH_COLORS,
        "font_sizes_pt": {"ticks": 11.2, "axis_labels": 12.7, "row_labels": 13.3, "path_headings": 15.5, "panel_letters": 14.5, "legend": 11.5},
        "outputs": {
            ext: {"path": str(OUT.with_suffix(f".{ext}")), "sha256": sha256(OUT.with_suffix(f".{ext}"))}
            for ext in ("pdf", "svg", "png")
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
