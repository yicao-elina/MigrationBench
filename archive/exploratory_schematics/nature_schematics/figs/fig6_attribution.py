"""Fig. 6 | Distribution & attribution: which coverage differences and features accompany error?"""

from figkit import components as C
from figkit import icons as I
from figkit import layout as L
from figkit import palette as P
from figkit.svgdoc import Svg, arrow

from .common import WIDTH, footnote, save, slot_row

STEPS = (
    ("Configurations", "all tasks, all models"),
    ("Local environments", "SOAP descriptors"),
    ("Shared map", "PCA, training vs task"),
    ("Coverage", "distance to training"),
    ("Attribution", "SHAP on error"),
)


def build():
    svg = Svg(WIDTH, 118)
    L.panel(svg, 3, 6, "a", "From configurations to explanations")
    bw, gap, x0, y0, bh = 31.5, 5.0, 4, 10, 34
    for i, (title, sub) in enumerate(STEPS):
        bx = x0 + i * (bw + gap)
        L.container(svg, bx, y0, bw, bh, rx=1.6)
        cx, cy = bx + bw / 2, y0 + 15
        if i == 0:
            C.vignette(svg, cx, cy + 1, 0.95, "within_ql")
        elif i == 1:
            I.soap_env(svg, cx, cy, 8.5)
        elif i == 2:
            I.pca_cloud(svg, cx, cy, 26, 20)
        elif i == 3:
            I.distance_glyph(svg, cx, cy, 26, 20)
        else:
            I.shap_bars(svg, cx - 11, cy - 9, 24, 18)
        svg.text(cx, y0 + bh - 5.0, title, size="small", weight="bold", anchor="middle")
        svg.text(cx, y0 + bh - 1.8, sub, size="note", anchor="middle", color=P.MUTED)
        if i < len(STEPS) - 1:
            arrow(svg, (bx + bw + 0.6, y0 + bh / 2), (bx + bw + gap - 0.6, y0 + bh / 2), sw=0.3, head=1.0)
    slot_row(svg, 54, 58, [
        ("b", "Task map", "task", "model", "dot matrix: deviation"),
        ("c", "SOAP-PCA", "PC1", "PC2", "training cloud + task points"),
        ("d", "Training distance", "distance to training", "error", "per configuration"),
        ("e", "SHAP", "SHAP value", "feature", "summary"),
    ])
    footnote(svg, "Panel a glyphs are schematic; b–e are built from the registry data.")
    save(svg, "fig6_attribution")


if __name__ == "__main__":
    build()
