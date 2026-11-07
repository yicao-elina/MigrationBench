"""Render the task-conditioned model-selection schematic for Figure 1.

This is a conceptual figure: its arrows describe the evaluation logic and do
not encode a quantitative model ranking.  Quantitative results remain in the
main figures and evidence tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

try:
    from plot_atlas import apply_publication_style, get_palette, save_figure
except ImportError:
    # Keep this conceptual figure reproducible on the manuscript build host
    # even when the optional Plot Atlas environment is unavailable.
    def apply_publication_style(font_size=8.4):
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "savefig.bbox": "tight",
        })

    def get_palette(_name):
        return {
            "ink": "#222222",
            "end": "#C44E52",
            "roles": {
                "global": "#2F5597",
                "retrained": "#4AAF4F",
                "1000": "#FB7D00",
            },
        }

    def save_figure(fig, output):
        fig.savefig(output, bbox_inches="tight", facecolor="white")


def rounded_box(ax, x, y, w, h, *, face, edge, title, body="", title_color=None,
                lw=1.0, title_size=9.0, body_size=7.4, alpha=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.045,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(x + 0.12, y + h - 0.16, title, ha="left", va="top",
            fontsize=title_size, fontweight="bold", color=title_color or edge)
    if body:
        ax.text(x + 0.12, y + h - 0.42, body, ha="left", va="top",
                fontsize=body_size, color="#262626", linespacing=1.25)
    return patch


def arrow(ax, start, end, *, color="#555555", lw=1.0, mutation=10,
          style="-|>", connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        linewidth=lw, color=color, connectionstyle=connectionstyle,
    ))


def layer_stack(ax, x, y, *, scale=1.0, dopant_x=None, path=None,
                host="#B9C3CC", te="#E6C27A", cr="#8A5DB7"):
    """Small layered Sb2Te3 cartoon with an optional Cr migration arrow."""
    widths = [1.24, 0.94, 1.24]
    ys = [y + 0.38, y + 0.14, y - 0.10]
    for row, yy in enumerate(ys):
        ax.plot([x, x + 1.25 * scale], [yy, yy], color=host, lw=1.0 * scale,
                solid_capstyle="round")
        for j in range(5):
            xx = x + (0.10 + j * 0.24) * scale
            ax.add_patch(Circle((xx, yy), 0.045 * scale, facecolor=te,
                                edgecolor="#8A6A32", linewidth=0.35))
    if dopant_x is not None:
        ax.add_patch(Circle((dopant_x, y + 0.26 * scale), 0.09 * scale,
                            facecolor=cr, edgecolor="#4D3266", linewidth=0.5,
                            zorder=4))
    if path == "gap":
        arrow(ax, (x + 0.20, y + 0.26), (x + 1.10, y + 0.26),
              color="#4AAF4F", lw=1.5, mutation=8)
    elif path == "deep":
        arrow(ax, (x + 0.22, y + 0.26), (x + 0.86, y - 0.05),
              color="#FB7D00", lw=1.5, mutation=8,
              connectionstyle="arc3,rad=-0.15")


def render(output: Path) -> None:
    apply_publication_style(8.4)
    palette = get_palette("nature_feature_analysis")
    orange = get_palette("nature_gdml_accuracy")["roles"]["1000"]
    warning = get_palette("paired_graphite_red")["end"]
    ink = palette["ink"]
    neutral = "#F5F6F7"
    blue = palette["roles"]["global"]
    green = palette["roles"]["retrained"]
    purple = "#98489F"

    fig, ax = plt.subplots(figsize=(12.2, 7.0))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # (a) The target process is a ladder of distributions, not a single test.
    ax.text(0.20, 6.82, "(a) Target process defines the relevant distribution",
            fontsize=10.5, fontweight="bold", color=ink, va="top")
    ax.text(0.20, 6.55,
            "The same material contains tasks with increasing distance from the training support",
            fontsize=8.0, color="#4B4B4B", va="top")
    arrow(ax, (0.55, 5.86), (11.65, 5.86), color="#666666", lw=1.1, mutation=12)
    ax.text(6.08, 6.03, "task difficulty / distribution shift  $\\longrightarrow$",
            fontsize=8.0, color="#4B4B4B", ha="center", va="bottom")

    task_regions = [
        (0.42, 4.95, 3.10, 0.72, "#EFF6FB", blue, "near equilibrium", "pointwise accuracy"),
        (3.72, 4.95, 3.26, 0.72, "#EEF8EC", green, "in-gap migration", "controlled process probe"),
        (7.22, 4.95, 4.42, 0.72, "#FFF4E7", orange, "within-QL / sliding", "hard extrapolation probe"),
    ]
    for x, y, w, h, face, edge, title, body in task_regions:
        rounded_box(ax, x, y, w, h, face=face, edge=edge, title=title,
                    body=body, title_size=8.2, body_size=7.0, lw=0.9)

    layer_stack(ax, 2.05, 5.15, scale=0.68, dopant_x=2.42)
    layer_stack(ax, 5.48, 5.15, scale=0.68, dopant_x=5.86, path="gap")
    layer_stack(ax, 9.62, 5.15, scale=0.68, dopant_x=10.00, path="deep")
    ax.text(11.10, 5.30, "collective registry\nchanges", fontsize=7.2,
            ha="center", va="center", color="#5B4A32")

    # (b) Model routes: prior, target data, and their conditional combination.
    ax.text(0.20, 4.50, "(b) Model routes encode different priors and calibration risks",
            fontsize=10.5, fontweight="bold", color=ink, va="top")
    rounded_box(ax, 0.52, 3.55, 2.20, 0.58, face="#EEF6FB", edge=blue,
                title="target DFT / AIMD", body="task-relevant data", title_size=8.0,
                body_size=7.0)
    rounded_box(ax, 4.90, 3.55, 2.35, 0.58, face="#F5ECF6", edge=purple,
                title="foundation prior", body="broad pre-training", title_size=8.0,
                body_size=7.0)

    rounded_box(ax, 0.38, 2.60, 3.10, 0.76, face="#EFF6FB", edge=blue,
                title="specialist: train from scratch",
                body="target calibration  +  extrapolation risk",
                title_size=8.3, body_size=7.2)
    rounded_box(ax, 4.55, 2.60, 3.10, 0.76, face="#F5ECF6", edge=purple,
                title="generalist: zero-shot",
                body="broad prior  +  target offset / softening",
                title_size=8.3, body_size=7.2)
    rounded_box(ax, 8.72, 2.60, 3.10, 0.76, face="#EEF8EC", edge=green,
                title="foundation + target adaptation",
                body="prior  +  sparse data  →  conditional alignment",
                title_size=8.3, body_size=7.2)

    arrow(ax, (1.62, 3.53), (1.62, 3.38), color=blue, lw=1.0, mutation=9)
    arrow(ax, (6.08, 3.53), (6.08, 3.38), color=purple, lw=1.0, mutation=9)
    arrow(ax, (2.72, 3.84), (9.28, 3.38), color="#777777", lw=0.9, mutation=9,
          connectionstyle="arc3,rad=-0.18")
    arrow(ax, (5.18, 3.54), (9.55, 3.38), color=green, lw=1.0, mutation=9,
          connectionstyle="arc3,rad=0.10")

    # (c) Shared evaluation criteria make the comparison process-level and
    # evidence-bound without exposing internal audit language to readers.
    ax.text(0.20, 2.20, "(c) Shared evaluation criteria produce a task-conditioned decision",
            fontsize=10.5, fontweight="bold", color=ink, va="top")
    gate_y, gate_h = 1.18, 0.72
    gates = [
        (0.34, 2.28, "equilibrium\nRMSE", "baseline", blue),
        (3.05, 2.28, "fixed-path\ncalibration", "common reference", green),
        (5.76, 2.28, "self-relaxed NEB /\nMD stability", "force feedback", warning),
        (8.47, 2.90, "collective /\nhard-OOD response", "stress test", orange),
    ]
    for x, w, title, body, edge in gates:
        rounded_box(ax, x, gate_y, w, gate_h, face=neutral, edge=edge,
                    title=title, body=body, title_size=8.0, body_size=6.8,
                    lw=1.1)
    for x in (2.66, 5.37, 8.08):
        arrow(ax, (x, 1.54), (x + 0.30, 1.54), color="#666666", lw=0.9, mutation=8)

    # The bottom row is a decision rule, not a claim of universal ranking.
    decision_y, decision_h = 0.18, 0.63
    decisions = [
        (0.34, 3.34, "near support", "adapted prior can help", "#EEF8EC", green),
        (4.00, 3.34, "moderate shift", "compare grouped seeds", "#EFF6FB", blue),
        (7.60, 4.35, "far shift", "no automatic winner  →  diagnose coverage gaps", "#FFF4E7", orange),
    ]
    for x, w, title, body, face, edge in decisions:
        rounded_box(ax, x, decision_y, w, decision_h, face=face, edge=edge,
                    title=title, body=body, title_size=8.0, body_size=7.0,
                    lw=1.0)
    ax.text(11.72, 2.02, "A model can pass one task\nand fail another.",
            fontsize=7.0, color="#4B4B4B", ha="right", va="bottom")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output)
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)


if __name__ == "__main__":
    main()
