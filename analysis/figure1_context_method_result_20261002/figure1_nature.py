"""Nature-style Figure 1: context, diffusion-aware method, and evidence.

The conceptual panel is explicitly schematic. Quantitative panels use the
repository's audited Phase-1 MD/force-disagreement and exploratory NEB-path
artifacts; the latter are labelled exploratory rather than converged barriers.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#202124"
MUTED = "#65717c"
BLUE = "#2b6cb0"
TEAL = "#008c95"
ORANGE = "#d97925"
MAGENTA = "#a34e86"
RED = "#c44747"
PALE_BLUE = "#e9f1fa"
PALE_TEAL = "#e6f4f2"
PALE_ORANGE = "#fbefe3"
PALE_GRAY = "#f3f5f6"


def style_ax(ax, *, grid=False):
    if grid:
        ax.grid(color="#dfe4e8", lw=0.55, zorder=0)
    else:
        ax.grid(False)
    for s in ax.spines.values():
        s.set_color(INK); s.set_linewidth(0.85)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors=INK, labelsize=7)
    for t in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        t.set_fontweight("bold")


def panel_letter(ax, letter):
    ax.text(-0.16, 1.015, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="left", color=INK)


def panel_a(ax):
    """Schematic: similar static errors need not imply similar task reliability."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(.02, .97, "The evaluation gap", fontsize=9, fontweight="bold", va="top")
    ax.text(.46, .86, "similar single-point error", fontsize=7.0, color=MUTED, ha="center")
    ax.text(.50, .08, "different diffusion-task reliability", fontsize=7.0, color=MUTED, ha="center")
    ax.plot([.12, .90], [.17, .17], color=INK, lw=.8)
    ax.plot([.12, .12], [.17, .82], color=INK, lw=.8)
    ax.text(.88, .11, "single-point\nerror", fontsize=6.2, ha="right", color=MUTED)
    ax.text(.06, .50, "task-level\nreliability", fontsize=6.2, ha="center", color=MUTED, rotation=90)
    ax.scatter([.46, .53], [.74, .32], s=62, c=[BLUE, ORANGE], edgecolor="white", lw=1, zorder=3)
    ax.text(.49, .79, "Model A", fontsize=7, color=BLUE, ha="center")
    ax.text(.58, .27, "Model B", fontsize=7, color=ORANGE, ha="center")
    ax.annotate("same RMSE\ncan hide different\ndownstream behaviour", xy=(.49, .53), xytext=(.72, .61),
                fontsize=7, color=INK, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=.8, color=MUTED))
    ax.text(.52, .01, "schematic", fontsize=6, color=MUTED, ha="center", style="italic")


def rounded_box(ax, xy, wh, text, face, edge, text_color=INK, fs=7.2, weight="bold"):
    x, y = xy; w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                       facecolor=face, edgecolor=edge, linewidth=.9)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            color=text_color, fontweight=weight, linespacing=1.15)
    return p


def panel_b(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(.02, .97, "Diffusion-aware evaluation", fontsize=9, fontweight="bold", va="top")
    ax.text(.02, .87, "from local errors to task reliability", fontsize=7.2, color=MUTED)
    rounded_box(ax, (.08, .67), (.84, .14), "1  Static reference\nenergy • forces • stress", PALE_GRAY, "#89939b")
    rounded_box(ax, (.08, .42), (.84, .14), "2  MD trajectory\nfinite geometry • force spread • stability", PALE_BLUE, BLUE)
    rounded_box(ax, (.08, .17), (.84, .14), "3  Non-equilibrium paths\nrelative profile • perturbation response • path sensitivity", PALE_ORANGE, ORANGE, fs=6.9)
    for y in [.67, .42]:
        ax.add_patch(FancyArrowPatch((.50, y), (.50, y-.10), arrowstyle="-|>", mutation_scale=9,
                                     lw=.85, color=MUTED))
    ax.text(.06, .06, "difficulty / non-equilibrium character  →", fontsize=6.5, color=MUTED)
    ax.plot([.45, .92], [.075, .075], color=RED, lw=1.1, alpha=.75)
    ax.text(.94, .075, "higher", fontsize=6.2, color=RED, va="center", ha="left")


def panel_c(ax):
    """Audited MD geometry/force-integrity warning."""
    steps = np.array([460000, 461000, 462000, 462200])
    min_dist = np.array([2.393, 2.360, 1.979, .629])
    force_spread = np.array([1.29, 5.07, 55.22, 4565.24])
    ax2 = ax.twinx()
    ax.plot(steps, min_dist, color=TEAL, lw=1.8, marker="o", ms=4.2, label="minimum pair distance")
    ax2.plot(steps, force_spread, color=RED, lw=1.8, marker="s", ms=3.8, label="max force spread")
    ax.axvline(461000, color="#b5bdc4", lw=.8, ls=(0, (3, 2)))
    ax.axvspan(462000, 462200, color=RED, alpha=.055, zorder=0)
    ax.set_xlabel("shared trajectory step", fontsize=7.5)
    ax.set_ylabel("minimum pair distance (Å)", color=TEAL, fontsize=7.5)
    ax2.set_ylabel("maximum pairwise force spread (eV Å$^{-1}$)", color=RED, fontsize=7.5)
    ax.set_ylim(.35, 2.65); ax2.set_yscale("log"); ax2.set_ylim(.8, 1.2e4)
    ax.set_xticks(steps); ax.set_xticklabels(["460k", "461k", "462k", "462.2k"])
    ax2.tick_params(axis="y", colors=RED, labelsize=7, width=.8, length=3.5)
    ax.tick_params(axis="y", colors=TEAL)
    ax.text(.96, .91, "geometry collapse", transform=ax.transAxes, ha="right", fontsize=6.7, color=RED)
    ax.text(.96, .84, "not a model-accuracy ranking", transform=ax.transAxes, ha="right", fontsize=6.3, color=MUTED, style="italic")
    style_ax(ax); style_ax(ax2)
    ax2.spines["right"].set_color(RED); ax.spines["left"].set_color(TEAL)
    handles = [Line2D([], [], color=TEAL, marker="o", lw=1.5, ms=3.7, label="minimum distance"),
               Line2D([], [], color=RED, marker="s", lw=1.5, ms=3.4, label="force spread")]
    ax.legend(handles=handles, frameon=False, fontsize=6.4, loc="upper left", handletextpad=.35)


def panel_d(ax):
    """Exploratory path-sampling summary, using audited path-level values."""
    models = ["Foundation", "MPA0"]
    profile = [0.1567, 0.1338]
    barrier = [0.3004, 0.2775]
    y = np.arange(2)
    h = .27
    ax.barh(y+h/2, profile, height=h, color=[BLUE, TEAL], alpha=.92, label="median profile MAE")
    ax.barh(y-h/2, barrier, height=h, color=[MAGENTA, ORANGE], alpha=.82, label="median |barrier difference|")
    for i, (p, b) in enumerate(zip(profile, barrier)):
        ax.text(p+.008, i+h/2, f"{p:.3f}", va="center", fontsize=6.6)
        ax.text(b+.008, i-h/2, f"{b:.3f}", va="center", fontsize=6.6)
    ax.set_yticks(y, models); ax.invert_yaxis(); ax.set_xlim(0, .40)
    ax.set_xlabel("error relative to screened path (eV)", fontsize=7.5)
    ax.set_title("exploratory non-equilibrium path sampling", fontsize=8, loc="left", pad=5, fontweight="bold")
    ax.text(.98, .98, "8/10 paths passed energy-stability screen\n240 DFT-labelled frames; 2 paths excluded", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.3, color=MUTED, linespacing=1.2)
    ax.text(.98, .04, "screening diagnostic, not converged NEB barrier", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.1, color=ORANGE, style="italic")
    ax.legend(frameon=False, fontsize=6.2, loc="lower left", bbox_to_anchor=(.02, .02), handlelength=1.2)
    style_ax(ax)


def main():
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(7.25, 5.7), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[.98, 1.08], width_ratios=[.82, 1.18],
                          left=.09, right=.96, bottom=.10, top=.87, wspace=.47, hspace=.62)
    a = fig.add_subplot(gs[0, 0]); panel_a(a); panel_letter(a, "a")
    b = fig.add_subplot(gs[0, 1]); panel_b(b); panel_letter(b, "b")
    c = fig.add_subplot(gs[1, 0]); panel_c(c); panel_letter(c, "c")
    d = fig.add_subplot(gs[1, 1]); panel_d(d); panel_letter(d, "d")
    fig.suptitle("Diffusion-aware evaluation exposes model differences hidden by single-point errors",
                 fontsize=11.2, fontweight="bold", x=.09, ha="left", y=.965)
    fig.text(.09, .932, "A benchmark should test the dynamical and path-level regimes in which a machine-learning potential is used.",
             fontsize=7.4, color=MUTED, ha="left")
    fig.savefig(OUT / "figure1_diffusion_aware_evaluation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figure1_diffusion_aware_evaluation.svg", bbox_inches="tight")
    fig.savefig(OUT / "figure1_diffusion_aware_evaluation.png", dpi=450, bbox_inches="tight")
    plt.close(fig)
    (OUT / "figure1_caption.md").write_text(
        """# Figure 1 | Diffusion-aware evaluation of machine-learning potentials

**a, Evaluation gap.** Schematic illustration that similar single-point errors need not imply similar reliability in downstream diffusion tasks. This panel is conceptual, not a quantitative result. **b, Evaluation framework.** The proposed hierarchy combines static reference quantities with MD trajectory integrity and non-equilibrium path diagnostics, while increasing the difficulty and non-equilibrium character of the test. **c, MD stress test.** In the common trajectory, the minimum pair distance decreases from 2.393 to 0.629 Å while the maximum pairwise force spread increases from 1.29 to 4,565.24 eV Å−1. The panel documents a geometry/force-integrity failure regime and is not a model-accuracy ranking. **d, Non-equilibrium path sampling.** Across eight paths passing the exploratory energy-stability screen and 240 DFT-labelled frames, Foundation and MPA0 show median relative-profile MAEs of 0.1567 and 0.1338 eV and median absolute barrier differences of 0.3004 and 0.2775 eV, respectively. These are screening diagnostics based on fixed-geometry exploratory path samples; they are not converged NEB barriers or universal transfer guarantees.

**Take-home message.** Single-point error metrics are necessary but insufficient for diffusion-focused model assessment; MD and path-level tests reveal failure modes and model differences that static metrics can miss.
""")


if __name__ == "__main__":
    main()
