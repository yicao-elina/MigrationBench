"""Shared bits for the six figure builders."""

from pathlib import Path

from figkit import icons as I
from figkit import layout as L
from figkit import palette as P
from figkit.atoms import sphere_fill
from figkit.svgdoc import Svg

OUT = Path(__file__).resolve().parents[1] / "out"
WIDTH = 183.0
MARGIN = 3.0

PROBE_CAPTION = {
    "static": "Pointwise E / F",
    "md": "Rollout stability",
    "within_gap": "Local motion",
    "gap_to_ql": "Entry environment",
    "within_ql": "Intralayer motion",
    "sliding": "Collective motion",
}


def atom_legend(svg: Svg, x, y, gap=9.0):
    for i, sp in enumerate(("Te", "Sb", "Cr")):
        cx = x + i * gap
        svg.circle(cx, y - 0.8, 1.1, fill=sphere_fill(svg, P.ATOM[sp][0]))
        svg.text(cx + 1.8, y, sp, size="small")


def footnote(svg: Svg, text: str):
    svg.text(MARGIN, svg.h - 1.6, text, size="note", color=P.MUTED, italic=True)


def save(svg: Svg, name: str):
    svg.save(OUT / name)
    print("wrote", OUT / name)


def slot_row(svg: Svg, y: float, h: float, specs, x0: float = MARGIN, x1: float = WIDTH - MARGIN, gap: float = 3.5):
    """Equal-width data slots: specs = [(letter, title, xlabel, ylabel, note), ...]."""
    w = (x1 - x0 - gap * (len(specs) - 1)) / len(specs)
    for i, (letter, title, xlabel, ylabel, note) in enumerate(specs):
        x = x0 + i * (w + gap)
        L.panel(svg, x, y, letter, title)
        I.plot_slot(svg, x, y + 2.2, w, h - 2.2, xlabel, ylabel, note)
