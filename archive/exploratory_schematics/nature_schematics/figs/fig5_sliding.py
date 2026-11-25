"""Fig. 5 | Interlayer sliding: how do collective displacement and doping change performance?"""

from figkit import components as C
from figkit import icons as I
from figkit import layout as L
from figkit import palette as P
from figkit.svgdoc import Svg, arrow

from .common import WIDTH, footnote, save, slot_row


def build():
    svg = Svg(WIDTH, 118)
    L.panel(svg, 3, 6, "a", "Same shear, with and without the dopant")
    for i, (task, title) in enumerate((("sliding_pristine", "Pristine Sb_{2}Te_{3}"), ("sliding", "Cr in the gap"))):
        cx = 48 + i * 88
        svg.text(cx, 12, title, size="label", weight="bold", anchor="middle")
        C.vignette(svg, cx, 36, 1.7, task, labels=(i == 0))
        for k, kind in enumerate(("fixed", "self")):
            x = cx - 26 + k * 28
            svg.rect(x, 56, 1.0, 9, fill=P.OPERATOR[kind][0])
            I.path_operator(svg, x + 2, 56, 14, 8, kind)
            svg.text(x + 17, 61.5, P.OPERATOR[kind][1], size="note", color=P.MUTED)
    arrow(svg, (78, 34), (104, 34), sw=0.35)
    svg.text(91, 31.5, "+ Cr", size="label", anchor="middle", weight="bold", color=P.ATOM["Cr"][0])
    svg.text(91, 38.5, "doping response", size="note", anchor="middle", color=P.MUTED)
    slot_row(svg, 74, 39, [
        ("b", "Pristine peak", "model", "peak energy (eV)", "fixed / self-relaxed"),
        ("c", "Cr-in-gap peak", "model", "peak energy (eV)", "fixed / self-relaxed"),
        ("d", "Doped − pristine", "model", "Δ peak (eV)", "doping response"),
    ])
    footnote(svg, "Slabs and upper-QL streaks follow the DFT sliding path (upper QL shifts, lower QL fixed).")
    save(svg, "fig5_sliding")


if __name__ == "__main__":
    build()
