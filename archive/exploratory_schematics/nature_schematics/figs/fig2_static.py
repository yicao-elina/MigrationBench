"""Fig. 2 | Static accuracy: how accurate are E and F on the same configurations?"""

from figkit import components as C
from figkit import icons as I
from figkit import layout as L
from figkit import palette as P
from figkit.svgdoc import Svg, arrow

from .common import WIDTH, footnote, save, slot_row

MODELS = ("foundation", "finetune", "scratch")


def panel_schematic(svg: Svg):
    L.panel(svg, 3, 6, "a", "One shared configuration set, two observables")
    L.container(svg, 4, 10, 66, 58, rx=2)
    C.md_frame(svg, 18, 38, 1.0, 0.1, seed=21)
    for k, (kind, label) in enumerate((("600K", "600 K"), ("multiT", "Multi-T"))):
        I.thermal_glyph(svg, 10 + 15 * k, 21, 4.2, kind)
        svg.text(10 + 15 * k + (2.0 if kind == "600K" else 3.6), 22, label, size="note", color=P.MUTED)
    svg.text(18, 57, "AIMD test frames", size="small", anchor="middle", weight="bold")
    svg.text(18, 60.2, "600 K and multi-T", size="note", anchor="middle", color=P.MUTED)
    for i, key in enumerate(MODELS):
        y = 22 + i * 14
        I.network(svg, 40, y, 5.2, P.INIT[key][0], key)
        svg.text(40, y + 6.6, P.INIT[key][1], size="note", anchor="middle", color=P.INIT[key][0])
        arrow(svg, (29.5, 38), (35.5, y), bend=0.0, sw=0.25, head=0.9)
        arrow(svg, (44.8, y), (52, 25 if i == 0 else 47 if i == 2 else 36), sw=0.25, head=0.9, color=P.MUTED)
    svg.circle(58, 24, 4.2, fill="#fbe3ec", stroke=P.TASK_INK["static"], sw=0.25)
    svg.text(58, 25.6, "*E*", size=9, anchor="middle", color=P.TASK_INK["static"])
    svg.text(58, 31.2, "per atom", size="note", anchor="middle", color=P.MUTED)
    C.force_cluster(svg, 59, 46, 2.0)
    svg.text(59, 60.2, "*F* per atom", size="note", anchor="middle", color=P.MUTED)


def build():
    svg = Svg(WIDTH, 92)
    panel_schematic(svg)
    slot_row(svg, 6, 62, [
        ("b", "RMSE", "initialization × thermal", "RMSE", "E and F RMSE, seed replicates"),
        ("c", "Energy parity", "DFT energy", "Predicted energy", "parity"),
        ("d", "Force parity", "DFT force", "Predicted force", "parity"),
    ], x0=76)
    from figkit import layout
    layout.legend_init_thermal(svg, 76, 72.5, row=3.2)
    footnote(svg, "Force directions in a are schematic; full train/test distributions are in the SI.")
    save(svg, "fig2_static")


if __name__ == "__main__":
    build()
