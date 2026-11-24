#!/usr/bin/env python3
"""Render the frozen 4x3 Foundation self-relaxed NEB figure and panels."""

from __future__ import annotations

import argparse
import csv
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "axes.linewidth": 1.05,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
        }
    )


def panel_data(class_rows: list[dict[str, str]], element: str, path_class: str, family: str) -> list[dict[str, str]]:
    return [
        row
        for row in class_rows
        if row["element"] == element
        and row["path_class"] == path_class
        and row["model_family"] == family
        and finite(row["mean_rel_energy_eV"])
        and finite(row["sd_rel_energy_eV"])
    ]


def panel_summary(summaries: list[dict[str, str]], element: str, path_class: str, family: str) -> dict[str, str]:
    for row in summaries:
        if row["element"] == element and row["path_class"] == path_class and row["model_family"] == family:
            return row
    return {"n_accepted": "0", "n_expected": str(EXPECTED[family]), "status": "INCOMPLETE"}


def draw_panel(
    ax,
    class_rows: list[dict[str, str]],
    summaries: list[dict[str, str]],
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
        all_y.extend([value - spread for value, spread in zip(y, sd)])
        all_y.extend([value + spread for value, spread in zip(y, sd)])
        color = COLORS[family]
        ax.fill_between(x, [value - spread for value, spread in zip(y, sd)], [value + spread for value, spread in zip(y, sd)], color=color, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(x, y, color=color, linewidth=1.65, zorder=3)
        ax.plot(x, y, linestyle="None", marker="o", markersize=4.1, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.05, zorder=4)

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
        spine.set_linewidth(1.05)
        spine.set_visible(True)
    ax.text(0.025, 0.965, panel_label, transform=ax.transAxes, ha="left", va="top", fontsize=11.5, fontweight="bold")
    if standalone:
        ax.set_title(f"{element} · {PANEL_LABELS[path_class]}", pad=12, fontweight="bold")
    elif element == ELEMENTS[0]:
        ax.set_title(PANEL_LABELS[path_class], pad=13, fontweight="bold")
    return {"complete": complete, "legend_counts": legend_counts, "n_y": len(all_y)}


def make_legend(summaries: list[dict[str, str]], element: str | None = None, path_class: str | None = None):
    handles = []
    for family in FAMILIES:
        if element and path_class:
            summary = panel_summary(summaries, element, path_class, family)
            label = f"{family} (n={summary.get('n_accepted', '0')})"
        else:
            counts = sorted({row["n_accepted"] for row in summaries if row["model_family"] == family})
            expected = EXPECTED[family]
            label = f"{family} (n={counts[0] if len(counts) == 1 else '?'} / {expected})"
        handles.append(Line2D([0], [0], color=COLORS[family], marker="o", markerfacecolor="white", markeredgecolor=COLORS[family], markersize=5, linewidth=1.7, label=label))
    statuses = {row.get("status", "") for row in summaries}
    if statuses == {"PASS"}:
        handles.append(Line2D([0], [0], color="#202020", marker="✓", markersize=8, linewidth=0, label="converged / included"))
    else:
        handles.append(Line2D([0], [0], color="#9a2d2d", marker="×", markersize=8, linewidth=0, label="failure / excluded"))
    return handles


def save_figure(fig, base: Path) -> list[Path]:
    outputs = [base.with_suffix(".png"), base.with_suffix(".svg"), base.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=400, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(outputs[2], bbox_inches="tight", pad_inches=0.08)
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
    derived = json.loads((data / "DERIVED_PROFILE_MANIFEST.json").read_text())
    if not derived.get("release_ready") and not args.allow_incomplete:
        raise SystemExit("release gate is not ready; use --allow-incomplete only for a diagnostic candidate")
    setup_style()
    figures.mkdir(exist_ok=True)
    panels_dir = figures / "panels"
    panels_dir.mkdir(exist_ok=True)

    composite = plt.figure(figsize=(10.8, 11.8))
    axes = composite.subplots(4, 3, sharex=False, sharey=False)
    panel_states = []
    for row_index, element in enumerate(ELEMENTS):
        for col_index, path_class in enumerate(PATHS):
            panel_states.append(draw_panel(axes[row_index, col_index], class_rows, summaries, element, path_class, chr(ord("a") + row_index * 3 + col_index)))
        axes[row_index, 0].annotate(element, xy=(-0.29, 0.5), xycoords="axes fraction", ha="center", va="center", rotation=90, fontsize=12.5, fontweight="bold")
    composite.supxlabel("normalized image coordinate, s", x=0.52, y=0.018, fontsize=11.5)
    composite.supylabel("relative energy, ΔE (eV)", x=0.018, y=0.50, fontsize=11.5)
    composite.legend(handles=make_legend(summaries), loc="upper center", bbox_to_anchor=(0.53, 0.996), ncol=4, frameon=False, handlelength=2.0, columnspacing=1.4, fontsize=9.4)
    composite.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.925, wspace=0.36, hspace=0.46)
    composite_outputs = save_figure(composite, figures / "foundation_self_relaxed_4x3_R01")
    plt.close(composite)

    panel_outputs: list[Path] = []
    for element in ELEMENTS:
        for path_class in PATHS:
            figure = plt.figure(figsize=(3.7, 3.35))
            ax = figure.add_subplot(1, 1, 1)
            draw_panel(ax, class_rows, summaries, element, path_class, "a", standalone=True)
            figure.supxlabel("normalized image coordinate, s", x=0.54, y=0.015, fontsize=10.5)
            figure.supylabel("relative energy, ΔE (eV)", x=0.012, y=0.50, fontsize=10.5)
            figure.legend(handles=make_legend(summaries, element, path_class), loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=2, frameon=False, handlelength=1.8, columnspacing=1.0, fontsize=8.2)
            figure.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.78)
            panel_outputs.extend(save_figure(figure, panels_dir / f"panel_{element}_{path_class}_self_relaxed_R01"))
            plt.close(figure)

    all_outputs = composite_outputs + panel_outputs
    qa = {
        "schema_version": "foundation_self_relaxed_visual_qa_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_ready_input": bool(derived.get("release_ready")),
        "composite_axes": 12,
        "standalone_panel_count": 12,
        "formats": ["png", "svg", "pdf"],
        "style": {
            "font_priority": ["Arial", "Helvetica", "DejaVu Sans"],
            "grid": False,
            "independent_y_axes": True,
            "explicit_gutter": True,
            "shared_x_label": True,
            "shared_y_label": True,
            "blender": False,
        },
        "outputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in all_outputs
        ],
    }
    (figures / "VISUAL_QA_R01.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"release_ready": derived.get("release_ready"), "outputs": len(all_outputs), "qa": str(figures / 'VISUAL_QA_R01.json')}, indent=2))


if __name__ == "__main__":
    main()

