#!/usr/bin/env python3
"""Render the two standalone fixed-path Fig. 4b/c panels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, root_path, sha256_file, system, write_json


STYLE_FAMILY = {
    "foundation": "foundation_omat",
    "ft_600k": "ft_600k",
    "ft_multit": "ft_multit",
    "scratch": "from_scratch",
}


def family_color(style: dict, family: str, item: dict | None = None) -> str:
    """Resolve a named model color, allowing scratch subfamilies to diverge."""
    if family == "scratch" and item is not None:
        subfamily = item.get("scratch_subfamily")
        subfamily_colors = style.get("subfamily_colors", {})
        if subfamily in subfamily_colors:
            return subfamily_colors[subfamily]
    return style["families"][STYLE_FAMILY[family]]["color"]


def resolve_linestyle(value):
    """Convert JSON dash arrays into Matplotlib's tuple form."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], list):
        return (value[0], tuple(value[1]))
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def dft_profile(root: Path, config: dict, system_id: str) -> tuple[np.ndarray, np.ndarray]:
    spec = system(config, system_id)
    rows = read_rows(root_path(root, spec["dft_csv"]))
    return (
        np.asarray([float(row["Reaction_Coord"]) for row in rows], dtype=float),
        np.asarray([float(row[spec["dft_column"]]) for row in rows], dtype=float),
    )


def marker_for(model_id: str, item: dict, style: dict, ordinal: int) -> str:
    """Keep the original family marker mapping in the publication panel."""
    return style.get("family_markers", {}).get(item.get("family", "foundation"), "o")


def line_style_for(item: dict, style: dict, ordinal: int) -> str:
    """Use line style for seed/checkpoint identity so markers remain family-stable."""
    seed = item.get("training_seed", "NA")
    if seed in (None, "NA"):
        ordered = style.get("seed_order", [123, 234, 345])
        seed = ordered[ordinal % len(ordered)]
    return style.get("seed_line_styles", {}).get(str(seed), "--")


def plot_one(root: Path, config: dict, style: dict, profiles: list[dict], system_id: str, panel_label: str, output_stem: str) -> dict:
    x, dft = dft_profile(root, config, system_id)
    config_models = config["models"]
    model_ids = [mid for mid, item in config_models.items() if item["role"].startswith(("A1_", "A2_", "A3_", "A4_"))]
    by_model = {}
    for row in profiles:
        if row["system_id"] != system_id:
            continue
        by_model.setdefault(row["model_id"], []).append(row)
    missing = [model_id for model_id in model_ids if model_id not in by_model]
    if missing:
        raise RuntimeError(f"missing current fixed-path profiles for {system_id}: {missing}")
    max_value = float(np.max(dft))
    min_value = float(np.min(dft))
    for rows in by_model.values():
        values = np.asarray([float(row["relative_energy_eV"]) for row in sorted(rows, key=lambda item: int(item["image_index"]))])
        max_value = max(max_value, float(np.max(values)))
        min_value = min(min_value, float(np.min(values)))
    requested_min, requested_max = config["plot"]["axis_ylim"]
    y_min = min(float(requested_min), math_floor_half(min_value))
    y_max = max(float(requested_max), math_ceil_half(max_value))
    convention = style.get("plotting_convention", {})
    plt.rcParams.update({
        "font.family": config["plot"].get("font_family", "Arial"),
        "font.size": config["plot"].get("font_size", 11),
        "axes.linewidth": config["plot"].get("axes_linewidth", 1.1),
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=tuple(config["plot"]["figsize_inches"]))
    reference_trace = style.get("reference_trace", {})
    ax.plot(
        x, dft,
        color=config["plot"]["dft_color"],
        linewidth=reference_trace.get("linewidth", 1.35),
        linestyle=resolve_linestyle(reference_trace.get("linestyle", "-")),
        marker=reference_trace.get("marker", "o"),
        markersize=reference_trace.get("markersize", 4.2),
        markerfacecolor=reference_trace.get("markerfacecolor", config["plot"]["dft_color"]),
        markeredgecolor=reference_trace.get("markeredgecolor", config["plot"]["dft_color"]),
        markeredgewidth=reference_trace.get("markeredgewidth", 0.7),
        label=style.get("legend_labels", {}).get("dft", "DFT reference"),
        zorder=reference_trace.get("zorder", 8),
    )
    family_items = {"foundation": [], "ft_600k": [], "ft_multit": [], "scratch": []}
    for ordinal, model_id in enumerate(model_ids):
        item = config_models[model_id]
        rows = sorted(by_model[model_id], key=lambda value: int(value["image_index"]))
        values = np.asarray([float(row["relative_energy_eV"]) for row in rows], dtype=float)
        family = item["family"]
        family_items.setdefault(family, []).append((item, values, ordinal, model_id))
    summary_only = style.get("summary_only", False)
    for family, members in family_items.items():
        if not members:
            continue
        if summary_only:
            if family == "scratch":
                grouped_members = {}
                for member in members:
                    subgroup = member[0].get("scratch_subfamily", "scratch")
                    grouped_members.setdefault(subgroup, []).append(member)
            else:
                grouped_members = {family: members}
            for subgroup, subgroup_rows in grouped_members.items():
                array = np.vstack([values for _, values, _, _ in subgroup_rows])
                mean = array.mean(axis=0)
                std = array.std(axis=0, ddof=1) if len(array) > 1 else np.zeros_like(mean)
                color = family_color(style, family, subgroup_rows[0][0])
                ax.fill_between(
                    x, mean - std, mean + std, color=color,
                    alpha=style["uncertainty"]["band_alpha"], linewidth=0, zorder=2,
                )
                line_style = style.get("summary_line_styles", {}).get(subgroup, style.get("summary_line_style", "-"))
                ax.plot(
                    x, mean, color=color, linestyle=line_style,
                    linewidth=style["uncertainty"]["summary_line_width"],
                    marker=style.get("summary_marker", "o"),
                    markersize=style.get("summary_marker_size", 4.3),
                    markerfacecolor=style.get("summary_markerfacecolor", "white"),
                    markeredgecolor=color,
                    markeredgewidth=style.get("summary_markeredgewidth", 0.95),
                    zorder=5,
                )
            continue
        aggregate_foundation = family == "foundation" and style.get("aggregate_foundation", False)
        if not aggregate_foundation:
            for item, values, ordinal, model_id in members:
                color = family_color(style, family, item)
                marker = marker_for(model_id, item, style, ordinal)
                alpha = 0.88 if family == "foundation" else style["uncertainty"]["individual_seed_alpha"]
                ax.plot(
                    x, values, color=color, linestyle=line_style_for(item, style, ordinal),
                    linewidth=style["uncertainty"]["individual_seed_line_width"],
                    marker=marker, markersize=style["uncertainty"]["marker_size"], markerfacecolor=color,
                    markeredgecolor=color, markeredgewidth=0.65, alpha=alpha, zorder=4,
                )
        if family == "scratch":
            subgroup_members = {}
            for member in members:
                subgroup = member[0].get("scratch_subfamily", "scratch")
                subgroup_members.setdefault(subgroup, []).append(member)
            for subgroup, subgroup_rows in subgroup_members.items():
                array = np.vstack([values for _, values, _, _ in subgroup_rows])
                mean = array.mean(axis=0)
                std = array.std(axis=0, ddof=1) if len(array) > 1 else np.zeros_like(mean)
                color = family_color(style, family, subgroup_rows[0][0])
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=style["uncertainty"]["band_alpha"], linewidth=0, zorder=2)
                mean_style = style.get("subfamily_mean_line_styles", {}).get(subgroup, "-")
                ax.plot(x, mean, color=color, linestyle=mean_style, linewidth=style["uncertainty"]["summary_line_width"], zorder=6)
        elif family != "foundation" or aggregate_foundation:
            array = np.vstack([values for _, values, _, _ in members])
            mean = array.mean(axis=0)
            std = array.std(axis=0, ddof=1) if len(array) > 1 else np.zeros_like(mean)
            color = family_color(style, family, members[0][0])
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=style["uncertainty"]["band_alpha"], linewidth=0, zorder=2)
            ax.plot(x, mean, color=color, linestyle="-", linewidth=style["uncertainty"]["summary_line_width"], zorder=6)
    ax.set_xlim(*config["plot"]["axis_xlim"])
    ax.set_ylim(y_min, y_max)
    x_ticks = convention.get("x_major_ticks", list(np.linspace(0.0, 1.0, 6)))
    ax.set_xticks(x_ticks)
    if "x_tick_labels" in convention:
        ax.set_xticklabels(convention["x_tick_labels"])
    tick_min = math.ceil(y_min)
    tick_max = math.floor(y_max)
    if tick_min <= tick_max:
        ax.set_yticks(np.arange(tick_min, tick_max + 1.0, 1.0))
    ax.set_xlabel("Reaction Coordinate", labelpad=convention.get("axis_label_pad", 6), fontsize=convention.get("axis_label_size"))
    ax.set_ylabel("Relative Energy (eV)", labelpad=convention.get("axis_label_pad", 6), fontsize=convention.get("axis_label_size"))
    if convention.get("figure_level_title", False):
        fig.suptitle("Fixed-path DFT vs. MACE models", fontsize=convention.get("title_font_size", 12), fontweight=convention.get("title_font_weight", "bold"), y=convention.get("title_y", 0.99))
    else:
        ax.set_title("Fixed-path DFT vs. MACE models", fontsize=convention.get("title_font_size", 10.5), pad=7)
    ax.text(-0.16, convention.get("panel_label_y", 1.04), panel_label, transform=ax.transAxes, fontsize=convention.get("panel_label_size", 14), fontweight="bold", va="bottom")
    ax.tick_params(axis="both", which="major", direction=convention.get("tick_direction", "out"), width=convention.get("major_tick_width", config["plot"].get("tick_width", 1.0)), length=convention.get("major_tick_length", 4), pad=convention.get("tick_pad", 3))
    ax.tick_params(axis="both", which="minor", direction=convention.get("tick_direction", "out"), width=convention.get("minor_tick_width", 0.8), length=convention.get("minor_tick_length", 2.5))
    if convention.get("minor_ticks", False):
        ax.minorticks_on()
    ax.grid(False)
    if "show_all_spines" in convention:
        for spine in ax.spines.values():
            spine.set_visible(convention["show_all_spines"])
            spine.set_color(convention.get("spine_color", "#111111"))
            spine.set_linewidth(convention.get("spine_width", config["plot"].get("axes_linewidth", 1.1)))
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    legend_labels = style.get("legend_labels", {})
    handles = [Line2D(
        [0], [0], color=config["plot"]["dft_color"],
        linestyle=resolve_linestyle(reference_trace.get("linestyle", "-")),
        marker=reference_trace.get("marker", "o"),
        linewidth=reference_trace.get("linewidth", 1.35),
        markersize=reference_trace.get("markersize", 4.2),
        markerfacecolor=reference_trace.get("markerfacecolor", config["plot"]["dft_color"]),
        markeredgecolor=reference_trace.get("markeredgecolor", config["plot"]["dft_color"]),
        markeredgewidth=reference_trace.get("markeredgewidth", 0.7),
        label=legend_labels.get("dft", "DFT reference"),
    )]
    family_labels = [
        ("foundation", None),
        ("ft_600k", None),
        ("ft_multit", None),
        ("scratch", "scratch_600k"),
        ("scratch", "scratch_multit"),
    ]
    default_labels = {
        ("foundation", None): "Foundation (3 checkpoint variants)",
        ("ft_600k", None): "FT-600K (mean +/- 1 SD, n=3)",
        ("ft_multit", None): "FT-MultiT (mean +/- 1 SD, n=3)",
        ("scratch", "scratch_600k"): "Scratch-600K (mean +/- 1 SD, n=3)",
        ("scratch", "scratch_multit"): "Scratch-MultiT (mean +/- 1 SD, n=3)",
    }
    for family, subgroup in family_labels:
        if family not in family_items or not family_items[family]:
            continue
        if subgroup is not None and not any(item.get("scratch_subfamily") == subgroup for item, _, _, _ in family_items[family]):
            continue
        label_key = subgroup if subgroup is not None else family
        label = legend_labels.get(label_key, default_labels[(family, subgroup)])
        marker = None if summary_only else style.get("family_markers", {}).get(family, "o")
        mean_style = style.get("summary_line_styles", {}).get(subgroup, style.get("summary_line_style", style.get("subfamily_mean_line_styles", {}).get(subgroup, "-")))
        legend_item = {"scratch_subfamily": subgroup} if subgroup is not None else None
        handles.append(Line2D(
            [0], [0], color=family_color(style, family, legend_item), marker=marker,
            linestyle=mean_style,
            linewidth=style["uncertainty"]["summary_line_width"] if summary_only else 1.3,
            markersize=style.get("summary_marker_size", 4.3) if summary_only else 3.4,
            markerfacecolor=style.get("summary_markerfacecolor", "white") if summary_only else None,
            markeredgecolor=family_color(style, family, legend_item) if summary_only else None,
            markeredgewidth=style.get("summary_markeredgewidth", 0.95) if summary_only else None,
            label=label,
        ))
    if convention.get("figure_level_legend", False):
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=tuple(convention.get("legend_anchor", [0.5, 0.94])), fontsize=convention.get("legend_font_size", 7.0),
                   frameon=False, ncol=convention.get("legend_ncol", 2), handlelength=2.2, borderpad=0.2, handletextpad=0.45,
                   columnspacing=1.1, labelspacing=0.25)
    else:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize=7.0,
                  frameon=True, framealpha=0.94, ncol=2, borderpad=0.35, handletextpad=0.45,
                  columnspacing=0.8, labelspacing=0.25)
    if convention.get("figure_level_layout", False):
        fig.subplots_adjust(**convention.get("subplots_adjust", {"left": 0.17, "right": 0.985, "bottom": 0.19, "top": 0.80}))
    else:
        fig.tight_layout(pad=0.8)
    out_dir = root / "results/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{output_stem}.pdf"
    png = out_dir / f"{output_stem}.png"
    svg = out_dir / f"{output_stem}.svg"
    metadata = {"Title": f"{panel_label} interlayer sliding fixed-path relative-energy profile", "Creator": "MigrationBench sliding fixed-path campaign"}
    fig.savefig(pdf, metadata=metadata)
    fig.savefig(png, dpi=config["plot"]["dpi"], metadata=metadata)
    fig.savefig(svg, metadata=metadata)
    plt.close(fig)
    return {"system_id": system_id, "panel": panel_label, "pdf": str(pdf), "png": str(png), "svg": str(svg), "pdf_sha256": sha256_file(pdf), "png_sha256": sha256_file(png), "svg_sha256": sha256_file(svg), "y_limits": [y_min, y_max], "n_model_profiles": len(model_ids)}


def math_floor_half(value: float) -> float:
    return math.floor(value * 2.0) / 2.0


def math_ceil_half(value: float) -> float:
    return math.ceil(value * 2.0) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--style-file", type=Path, help="Optional style JSON override, relative to the campaign root")
    parser.add_argument("--output-tag", default=None, help="Optional suffix appended to both figure stems")
    parser.add_argument("--manifest-name", default="FIG4_FIXED_PATH_PLOT_MANIFEST.json", help="Manifest filename under results/figures")
    args = parser.parse_args()
    config, root = load_config(args.config)
    style_path = root_path(root, args.style_file) if args.style_file else root_path(root, config["plot"]["style_file"])
    style = json.loads(style_path.read_text())
    profile_path = root / "results/tables/fixed_profiles.csv"
    if not profile_path.is_file():
        raise SystemExit(f"missing fixed profile table: {profile_path}")
    profiles = read_rows(profile_path)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    outputs = [
        plot_one(root, config, style, profiles, "pristine", "(b)", f"fig4b_interlayer_sliding_pristine_fixed_path_20261024{tag}"),
        plot_one(root, config, style, profiles, "cr_in_gap", "(c)", f"fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024{tag}"),
    ]
    manifest = {
        "schema_version": "sliding_fixed_path_figure_manifest_v1",
        "campaign_id": config["campaign_id"],
        "observable": "FIXED_PATH_SINGLE_POINT",
        "data_table": str(profile_path),
        "data_table_sha256": sha256_file(profile_path),
        "style_file": str(style_path),
        "style_file_sha256": sha256_file(style_path),
        "outputs": outputs,
        "dft_reference_status": config["protocol"]["dft_reference_status"],
        "dft_reference_method": config["protocol"]["dft_reference_method"],
        "publication_boundary": "fixed-path relative-energy diagnostic; not a self-relaxed or validated migration barrier",
    }
    write_json(root / "results/figures" / args.manifest_name, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

