#!/usr/bin/env python3
"""Render the R03 Foundation self-relaxed 4×3 figure with QE/DFT reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


ELEMENTS = ["Cr", "Cu", "Li", "Na"]
PATHS = ["in_gap", "gap_to_ql", "within_ql"]
FAMILIES = ["MACE-MP", "MACE-OMAT-0", "MACE-MH"]
COLORS = {"MACE-MP": "#d5a238", "MACE-OMAT-0": "#d85a45", "MACE-MH": "#5d8b56"}
PANEL_LABELS = {"in_gap": "in-gap diffusion", "gap_to_ql": "gap→QL", "within_ql": "within-QL"}
EXPECTED = {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2}
REFERENCE_LABEL = "DFT-NEB"
REFERENCE_COLOR = "#222222"
REFERENCE_LINE_WIDTH = 2.55
REFERENCE_DASH_PATTERN = (0, (6, 3))
REFERENCE_STANDARD_REQUESTED = "/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/figures/foundation_class_3x3_R03/foundation_model_class_3x3_R03.pdf"
REFERENCE_STANDARD_ARCHIVED = "/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/figures/foundation_class_3x3_R03/foundation_model_class_3x3_R03.pdf"
REFERENCE_STANDARD_SHA256 = "59d6202451f9f94ff21f3fba9924e025001c0af6c406c2bdf2ca1fc947023f84"
OUTPUT_STEM = "foundation_self_relaxed_4x3_R03"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 17.0,
            "axes.labelsize": 19.0,
            "axes.titlesize": 19.0,
            "axes.linewidth": 1.80,
            "xtick.labelsize": 18.0,
            "ytick.labelsize": 18.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
        }
    )


def panel_data(class_rows: list[dict[str, str]], element: str, path_class: str, family: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in class_rows
        if row["element"] == element
        and row["path_class"] == path_class
        and row["model_family"] == family
        and finite(row["mean_rel_energy_eV"])
        and finite(row["sd_rel_energy_eV"])
    ]
    return sorted(rows, key=lambda row: int(row["image_index"]))


def panel_summary(summaries: list[dict[str, str]], element: str, path_class: str, family: str) -> dict[str, str]:
    for row in summaries:
        if row["element"] == element and row["path_class"] == path_class and row["model_family"] == family:
            return row
    return {"n_accepted": "0", "n_expected": str(EXPECTED[family]), "status": "INCOMPLETE"}


def reference_data(reference_rows: list[dict[str, str]], element: str, path_class: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in reference_rows
        if row["element"] == element
        and row["path_class"] == path_class
        and finite(row["relative_energy_eV"])
        and finite(row["plot_s"])
    ]
    return sorted(rows, key=lambda row: int(row["image_index"]))


def draw_panel(
    ax,
    class_rows: list[dict[str, str]],
    summaries: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    element: str,
    path_class: str,
    panel_label: str,
    standalone: bool = False,
) -> dict[str, object]:
    all_y: list[float] = []
    complete = True
    legend_counts: dict[str, str] = {}
    for family in FAMILIES:
        rows = panel_data(class_rows, element, path_class, family)
        summary = panel_summary(summaries, element, path_class, family)
        legend_counts[family] = f"{summary.get('n_accepted', '0')}/{summary.get('n_expected', EXPECTED[family])}"
        complete = complete and summary.get("status") == "PASS" and len(rows) == 10
        if not rows:
            continue
        x = [float(row["s"]) for row in rows]
        y = [float(row["mean_rel_energy_eV"]) for row in rows]
        sd = [float(row["sd_rel_energy_eV"]) for row in rows]
        low_band = [value - spread for value, spread in zip(y, sd)]
        high_band = [value + spread for value, spread in zip(y, sd)]
        all_y.extend(low_band)
        all_y.extend(high_band)
        color = COLORS[family]
        ax.fill_between(x, low_band, high_band, color=color, alpha=0.16, linewidth=0, zorder=1)
        line_width = 2.3 if not standalone else 2.0
        marker_size = 5.2 if not standalone else 4.8
        marker_edge_width = 1.35 if not standalone else 1.2
        ax.plot(x, y, color=color, linewidth=line_width, zorder=3)
        ax.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=marker_edge_width,
            zorder=4,
        )

    reference = reference_data(reference_rows, element, path_class)
    reference_statuses = sorted({row["raw_status"] for row in reference})
    if reference:
        ref_x = [float(row["plot_s"]) for row in reference]
        ref_y = [float(row["relative_energy_eV"]) for row in reference]
        all_y.extend(ref_y)
        ax.plot(ref_x, ref_y, color=REFERENCE_COLOR, linewidth=REFERENCE_LINE_WIDTH, linestyle=REFERENCE_DASH_PATTERN, zorder=5)
        ax.plot(
            ref_x,
            ref_y,
            linestyle="None",
            marker="D",
            markersize=3.6 if standalone else 4.2,
            markerfacecolor="white",
            markeredgecolor=REFERENCE_COLOR,
            markeredgewidth=0.8,
            zorder=6,
        )

    if all_y:
        low = min(0.0, min(all_y))
        high = max(0.05, max(all_y))
        span = max(high - low, 0.12)
        pad = 0.08 * span
        ax.set_ylim(low - pad, high + pad)
    else:
        ax.set_ylim(-0.05, 0.25)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="major", length=4.0, width=0.85)
    ax.tick_params(which="minor", length=2.2, width=0.65)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(1.80)
        spine.set_visible(True)
    ax.text(-0.025, 1.02, panel_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=20.5 if standalone else 19.0, fontweight="bold", clip_on=False, zorder=10)
    if standalone:
        ax.set_title(f"{element} · {PANEL_LABELS[path_class]}", pad=16, fontweight="bold")
    elif element == ELEMENTS[0]:
        ax.set_title(PANEL_LABELS[path_class], pad=13, fontweight="bold")
    return {
        "complete": complete,
        "legend_counts": legend_counts,
        "n_y": len(all_y),
        "n_reference": len(reference),
        "reference_statuses": reference_statuses,
    }


def make_legend(summaries: list[dict[str, str]], element: str | None = None, path_class: str | None = None, standalone: bool = False):
    handles = []
    marker_size = 6.0 if standalone else 7.0
    line_width = 2.1 if standalone else 2.4
    for family in FAMILIES:
        if element and path_class:
            summary = panel_summary(summaries, element, path_class, family)
            label = f"{family} (n={summary.get('n_accepted', '0')})"
        else:
            counts = sorted({row["n_accepted"] for row in summaries if row["model_family"] == family})
            expected = EXPECTED[family]
            label = f"{family} (n={counts[0] if len(counts) == 1 else '?'} / {expected})"
        handles.append(Line2D([0], [0], color=COLORS[family], marker="o", markerfacecolor="white", markeredgecolor=COLORS[family], markersize=marker_size, linewidth=line_width, label=label))
    handles.append(Line2D([0], [0], color=REFERENCE_COLOR, marker="D", markerfacecolor="white", markeredgecolor=REFERENCE_COLOR, markersize=3.6 if standalone else 4.2, linewidth=REFERENCE_LINE_WIDTH, linestyle=REFERENCE_DASH_PATTERN, label=REFERENCE_LABEL))
    return handles


def save_figure(fig, base: Path) -> list[Path]:
    outputs = [base.with_suffix(".png"), base.with_suffix(".svg"), base.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.08, facecolor="white")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    figures = root / "figures"
    class_rows = read_tsv(data / "CLASS_PROFILES.tsv")
    summaries = read_tsv(data / "CLASS_SUMMARY.tsv")
    reference_rows = read_tsv(data / "GROUND_TRUTH_REFERENCE_PROFILES_R03.tsv")
    reference_manifest = json.loads((data / "GROUND_TRUTH_REFERENCE_MANIFEST_R03.json").read_text())
    derived = json.loads((data / "DERIVED_PROFILE_MANIFEST.json").read_text())
    input_ready = bool(derived.get("release_ready")) and bool(reference_manifest.get("release_ready"))
    if not input_ready and not args.allow_incomplete:
        raise SystemExit("release gate is not ready; use --allow-incomplete only for a diagnostic candidate")
    setup_style()
    figures.mkdir(exist_ok=True)
    panels_dir = figures / "panels"
    panels_dir.mkdir(exist_ok=True)

    composite = plt.figure(figsize=(10.8, 11.8), facecolor="white")
    axes = composite.subplots(4, 3, sharex=False, sharey=False)
    panel_states = []
    for row_index, element in enumerate(ELEMENTS):
        for col_index, path_class in enumerate(PATHS):
            panel_states.append(draw_panel(axes[row_index, col_index], class_rows, summaries, reference_rows, element, path_class, chr(ord("a") + row_index * 3 + col_index)))
        axes[row_index, 0].annotate(element, xy=(-0.36, 0.5), xycoords="axes fraction", ha="center", va="center", rotation=90, fontsize=20.0, fontweight="bold")
    composite.supxlabel("Reaction Coordinate", x=0.52, y=0.018, fontsize=21.0, fontweight="bold")
    composite.supylabel("relative energy, ΔE (eV)", x=0.018, y=0.50, fontsize=19.0)
    composite.legend(handles=make_legend(summaries), loc="upper center", bbox_to_anchor=(0.53, 0.996), ncol=2, frameon=False, handlelength=2.0, columnspacing=1.25, handletextpad=0.45, fontsize=18.0)
    composite.subplots_adjust(left=0.14, right=0.985, bottom=0.09, top=0.74, wspace=0.42, hspace=0.60)
    composite_outputs = save_figure(composite, figures / OUTPUT_STEM)
    plt.close(composite)

    panel_outputs: list[Path] = []
    for element in ELEMENTS:
        for path_class in PATHS:
            with plt.rc_context({"font.size": 16.5, "axes.labelsize": 17.0, "axes.titlesize": 18.0, "xtick.labelsize": 17.0, "ytick.labelsize": 17.0}):
                figure = plt.figure(figsize=(3.9, 5.2), facecolor="white")
                ax = figure.add_subplot(1, 1, 1)
                draw_panel(ax, class_rows, summaries, reference_rows, element, path_class, "a", standalone=True)
                ax.set_xlabel("Reaction Coordinate", labelpad=12, fontsize=18.5, fontweight="bold")
                ax.set_ylabel("relative energy, ΔE (eV)", labelpad=14)
                figure.legend(handles=make_legend(summaries, element, path_class, standalone=True), loc="upper center", bbox_to_anchor=(0.52, 0.995), ncol=1, frameon=False, handlelength=1.8, labelspacing=0.45, fontsize=15.5)
                figure.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.48)
                panel_outputs.extend(save_figure(figure, panels_dir / f"panel_{element}_{path_class}_self_relaxed_R03"))
                plt.close(figure)

    all_outputs = composite_outputs + panel_outputs
    qa = {
        "schema_version": "foundation_self_relaxed_visual_qa_r03_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_ready_input": input_ready,
        "composite_axes": 12,
        "standalone_panel_count": 12,
        "formats": ["png", "svg", "pdf"],
        "style": {"font_priority": ["Arial", "Helvetica", "DejaVu Sans"], "intended_composite_width_mm": 180, "intended_standalone_width_mm": 100, "minimum_text_size_pt_at_intended_width": 11, "composite_source_text_sizes_pt": {"legend": 18.0, "ticks": 18.0, "shared_labels": 19.0, "shared_x_label": 21.0}, "standalone_source_text_sizes_pt": {"legend": 15.5, "ticks": 17.0, "axis_labels": 18.5}, "x_axis_label": "Reaction Coordinate", "grid": False, "independent_y_axes": True, "explicit_gutter": True, "shared_x_label": True, "shared_x_label_weight": "bold", "shared_y_label": True, "axes_spine_linewidth_pt": 1.8, "blender": False},
        "reference_layer": {"legend_label": REFERENCE_LABEL, "scientific_boundary_label": reference_manifest.get("reference_label"), "manifest_path": str(data / "GROUND_TRUTH_REFERENCE_MANIFEST_R03.json"), "manifest_sha256": sha256(data / "GROUND_TRUTH_REFERENCE_MANIFEST_R03.json"), "profiles_path": str(data / "GROUND_TRUTH_REFERENCE_PROFILES_R03.tsv"), "profiles_sha256": sha256(data / "GROUND_TRUTH_REFERENCE_PROFILES_R03.tsv"), "points": len(reference_rows), "points_per_panel": sorted({len(reference_data(reference_rows, element, path_class)) for element in ELEMENTS for path_class in PATHS}), "line_style": "dark dashed line with open-diamond markers", "line_width_pt": REFERENCE_LINE_WIDTH, "same_geometry_dft_rescoring": reference_manifest.get("alignment", {}).get("same_geometry_dft_rescoring"), "scientific_status": reference_manifest.get("scientific_status"), "raw_statuses": sorted({row["raw_status"] for row in reference_rows})},
        "reference_standard": {"requested_path": REFERENCE_STANDARD_REQUESTED, "requested_path_exists": False, "resolved_authoritative_path": REFERENCE_STANDARD_ARCHIVED, "resolved_authoritative_sha256": REFERENCE_STANDARD_SHA256, "resolution": "latest reference retained in explicit legacy archive; archive path is authoritative for this R03 release"},
        "panel_states": panel_states,
        "outputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in all_outputs],
    }
    (figures / "VISUAL_QA_R03.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"release_ready": qa["release_ready_input"], "outputs": len(all_outputs), "qa": str(figures / "VISUAL_QA_R03.json")}, indent=2))


if __name__ == "__main__":
    main()
