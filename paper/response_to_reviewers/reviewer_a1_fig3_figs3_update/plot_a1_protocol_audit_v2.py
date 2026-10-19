"""Publication redraw for Reviewer A1: fixed-path versus self-relaxed NEB.

The figure uses a concrete atomic-image schematic rather than a workflow diagram.
The left panel reports the audited fixed-path barriers.  The right panel shows the
valid self-relaxed NEB operator schematically and separates it from the invalidated
historical Fig. S3 record.
"""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from plot_atlas import apply_publication_style, save_figure


HERE = Path(__file__).resolve().parent
CSV = HERE / "fig3d_figS3_protocol_side_by_side_audited.csv"
OUT = HERE / "fig3d_figS3_protocol_side_by_side_audited"

BLACK = "#111111"
BLUE = "#2F6FB0"
RED = "#C83E4D"
GOLD = "#D79B1E"
PURPLE = "#A991C6"
SLATE = "#6A84A8"
ATOM_BLUE = "#6D8FB9"
ATOM_PURPLE = "#B9A8D1"
PALE_BLUE = "#EAF2FA"
PALE_RED = "#FDECEF"


def load_fixed_rows():
    with CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row["operator"] == "fixed_geometry_single_point"]
    order = ["DFT candidate", "Foundation", "Scratch", "FT-600K", "FT-MultiT"]
    lookup = {
        "DFT candidate": "DFT_reference",
        "Foundation": "Foundation_fixed",
        "Scratch": "Scratch_fixed",
        "FT-600K": "FT600K_fixed",
        "FT-MultiT": "FTMultiT_fixed",
    }
    by_record = {row["record"]: row for row in rows}
    return [(label, float(by_record[lookup[label]]["barrier_eV"])) for label in order]


def draw_atom(ax, x, y, radius, color, edge=BLACK, lw=0.45, z=5):
    ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor=edge, linewidth=lw, zorder=z))


def draw_snapshot(ax, cx, cy, dopant_x, dopant_y, width=0.82, height=0.55,
                  edge=BLACK, path_color=BLUE, node_color=None):
    """Draw a small layered-slab snapshot with one migrating dopant."""
    ax.add_patch(FancyBboxPatch(
        (cx - width / 2, cy - height / 2), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="white", edgecolor=edge, linewidth=0.75, zorder=1,
    ))
    left = cx - width * 0.39
    bottom = cy - height * 0.28
    dx = width * 0.19
    dy = height * 0.22
    for row in range(2):
        for col in range(4):
            jitter = 0.5 * (row % 2)
            draw_atom(ax, left + (col + jitter) * dx, bottom + row * dy,
                      width * 0.052, ATOM_PURPLE, lw=0.35, z=3)
    for row in range(2):
        for col in range(4):
            jitter = 0.5 * ((row + 1) % 2)
            draw_atom(ax, left + (col + jitter) * dx, bottom + height * 0.26 + row * dy,
                      width * 0.052, ATOM_BLUE, lw=0.35, z=4)
    draw_atom(ax, cx + dopant_x * width * 0.30,
              cy + dopant_y * height * 0.45,
              width * 0.078, GOLD, edge=BLACK, lw=0.55, z=7)


def bezier(ax, points, color, lw=1.8, ls="-", z=2):
    verts = [points[0]]
    codes = [MplPath.MOVETO]
    for i in range(1, len(points), 3):
        verts.extend(points[i:i + 3])
        codes.extend([MplPath.CURVE4] * 3)
    ax.add_patch(PathPatch(MplPath(verts, codes), fill=False, edgecolor=color,
                           linewidth=lw, linestyle=ls, zorder=z))


def path_nodes(ax, xs, ys, color, node_color=None, dashed=True):
    node_color = node_color or color
    for x, y in zip(xs, ys):
        ax.add_patch(Circle((x, y), 0.055, facecolor=node_color, edgecolor=BLACK,
                            linewidth=0.65, zorder=8))
    for x, y in zip(xs, ys):
        ax.plot([x, x], [y - 0.055, y - 0.50], color="#7B8794", linewidth=0.55,
                linestyle=(0, (2, 2)), zorder=1)


def draw_energy_axis(ax, x0, x1, y0, y1, ymax=5.0):
    ax.plot([x0, x0], [y0, y1], color=BLACK, linewidth=0.9, zorder=1)
    ax.plot([x0, x1], [y0, y0], color=BLACK, linewidth=0.9, zorder=1)
    for tick in range(6):
        y = y0 + (y1 - y0) * tick / ymax
        ax.plot([x0 - 0.05, x0], [y, y], color=BLACK, linewidth=0.75, zorder=1)
        ax.text(x0 - 0.10, y, str(tick), ha="right", va="center", fontsize=7.5,
                color=BLACK)


def draw_fixed_bar_panel(ax, values):
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 5.65)
    ax.axis("off")
    ax.text(0.10, 5.48, "(a) Fixed-path scoring", fontsize=13, fontweight="bold",
            color=BLACK, va="top")

    # Concrete atomic-image chain above the quantitative panel.
    xs = np.array([0.62, 1.62, 2.62, 3.62, 4.62])
    ys = np.array([4.12, 4.32, 4.55, 4.32, 4.12])
    points = [(xs[0], ys[0]), (0.95, 4.14), (1.28, 4.28), (xs[1], ys[1]),
              (1.95, 4.39), (2.28, 4.52), (xs[2], ys[2]),
              (2.95, 4.52), (3.28, 4.39), (xs[3], ys[3]),
              (3.95, 4.28), (4.28, 4.14), (xs[4], ys[4])]
    bezier(ax, points, BLUE, lw=1.7, ls=(0, (3, 2)), z=2)
    path_nodes(ax, xs, ys, BLUE, node_color="#B9D0E8")
    dopants = [(-0.20, -0.05), (-0.06, 0.16), (0.10, 0.24), (0.27, 0.16), (0.39, -0.05)]
    for x, y, (dx, dy) in zip(xs, ys, dopants):
        draw_snapshot(ax, x, y + 0.60, dx, dy, path_color=BLUE)

    # Bar chart, manually laid out to avoid annotation collisions.
    x0, x1, y0, y1 = 0.62, 5.52, 0.55, 2.35
    ymax = 5.0
    draw_energy_axis(ax, x0, x1, y0, y1, ymax=ymax)
    scale = (y1 - y0) / ymax
    colors = [BLACK, RED, GOLD, BLUE, SLATE]
    labels = ["DFT", "Foundation", "Scratch", "FT-600K", "FT-MultiT"]
    bar_x = np.linspace(0.92, 5.12, 5)
    width = 0.48
    for (label, value), x, color in zip(values, bar_x, colors):
        top = y0 + value * scale
        ax.add_patch(Rectangle((x - width / 2, y0), width, top - y0,
                               facecolor=color, edgecolor="white", linewidth=0.7, zorder=4))
        ax.text(x, top + 0.10, f"{value:.3f}", ha="center", va="bottom",
                fontsize=8.4, fontweight="bold", color=BLACK)
        ax.text(x, y0 - 0.12, label, ha="center", va="top", fontsize=7.3,
                color=BLACK)
    ref = 0.336050
    ref_y = y0 + ref * scale
    ax.plot([x0, x1], [ref_y, ref_y], color=BLACK, linewidth=1.0,
            linestyle=(0, (3, 2)), zorder=3)
    ax.text(x0 + 0.10, y1 + 0.05, "DFT reference = 0.336 eV", ha="left",
            va="bottom", fontsize=8.4, fontweight="bold", color=BLACK)
    ax.text(0.05, (y0 + y1) / 2, "fixed-path barrier (eV)", ha="center", va="center",
            rotation=90, fontsize=8.5, color=BLACK)
    ax.text(3.02, 0.16, "quantitative comparison", fontsize=8.6, fontweight="bold",
            ha="center", color=BLACK)


def draw_self_panel(ax):
    ax.set_xlim(0, 6.0)
    ax.set_ylim(0, 5.65)
    ax.axis("off")
    ax.text(0.10, 5.48, "(b) Self-relaxed NEB", fontsize=13, fontweight="bold",
            color=BLACK, va="top")

    xs = np.array([0.65, 1.55, 2.45, 3.35, 4.25, 5.15])
    ys = np.array([4.05, 4.32, 4.15, 4.62, 4.25, 4.05])
    points = [(xs[0], ys[0]), (0.95, 4.08), (1.22, 4.27), (xs[1], ys[1]),
              (1.82, 4.30), (2.18, 4.15), (xs[2], ys[2]),
              (2.72, 4.12), (3.03, 4.54), (xs[3], ys[3]),
              (3.67, 4.61), (3.96, 4.28), (xs[4], ys[4]),
              (4.55, 4.20), (4.88, 4.08), (xs[5], ys[5])]
    bezier(ax, points, RED, lw=1.8, ls=(0, (3, 2)), z=2)
    path_nodes(ax, xs, ys, RED, node_color="#F4A5AD")
    dopants = [(-0.20, -0.05), (-0.04, 0.10), (0.08, -0.03), (0.18, 0.24),
               (0.34, 0.10), (0.42, -0.05)]
    for x, y, (dx, dy) in zip(xs, ys, dopants):
        draw_snapshot(ax, x, y + 0.60, dx, dy, width=0.70, height=0.50, edge=BLACK)

    # Model energy profile below the image chain.
    ex = np.linspace(0.65, 5.15, 100)
    ey = 2.40 + 0.42 * np.sin((ex - 0.65) / 4.5 * np.pi) ** 1.7
    ey += 0.12 * np.sin((ex - 0.65) / 4.5 * 3 * np.pi)
    ax.plot(ex, ey, color=RED, linewidth=1.9, zorder=2)
    ax.scatter(xs, np.interp(xs, ex, ey), s=22, color="#F4A5AD", edgecolor=BLACK,
               linewidth=0.6, zorder=5)
    ax.plot([0.55, 5.35], [2.25, 2.25], color=BLACK, linewidth=0.8, zorder=1)
    ax.plot([0.55, 0.55], [2.20, 2.95], color=BLACK, linewidth=0.8, zorder=1)
    ax.text(0.43, 2.57, "MLIP energy", fontsize=8.1, color=BLACK, rotation=90,
            ha="center", va="center")
    ax.text(2.95, 2.06, "model relaxes its own image sequence", fontsize=8.4,
            fontweight="bold", color=BLACK, ha="center")

    # Historical record: concise, black text, with a single red status accent.
    card = FancyBboxPatch((0.46, 0.42), 5.08, 1.28,
                          boxstyle="round,pad=0.025,rounding_size=0.04",
                          facecolor=PALE_RED, edgecolor=RED, linewidth=1.15, zorder=1)
    ax.add_patch(card)
    ax.text(0.66, 1.48, "Historical Fig. S3 record", fontsize=9.7, fontweight="bold",
            color=BLACK, va="top")
    ax.text(0.66, 1.19, "submitted: 0.410 eV    |    82 atoms    |    12 images",
            fontsize=8.8, color=BLACK, va="top")
    ax.text(0.66, 0.94, "uniform rescoring: 0.000 / 0.379 / 1.037 eV",
            fontsize=8.8, color=BLACK, va="top")
    ax.text(0.66, 0.68, "different system + calculator/serialization defects",
            fontsize=8.8, color=BLACK, va="top")
    ax.text(5.26, 0.68, "EXCLUDED", fontsize=9.0, fontweight="bold", color=RED,
            ha="right", va="top")


def main():
    apply_publication_style(font_size=9.0)
    values = load_fixed_rows()
    fig = plt.figure(figsize=(12.0, 6.0), constrained_layout=False)
    gs = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.90, bottom=0.055,
                          wspace=0.045)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    draw_fixed_bar_panel(ax_a, values)
    draw_self_panel(ax_b)
    fig.text(0.055, 0.965, "Fixed-path scoring versus self-relaxed NEB",
             fontsize=15, fontweight="bold", color=BLACK, ha="left", va="top")
    fig.add_artist(plt.Line2D([0.505, 0.505], [0.075, 0.89], transform=fig.transFigure,
                              color="#B8C0C8", linewidth=0.8))
    for ext in ("pdf", "svg", "png"):
        save_figure(fig, OUT.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"wrote {OUT}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
