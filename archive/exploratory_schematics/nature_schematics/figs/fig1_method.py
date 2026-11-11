"""Fig. 1 | Method: what is compared, and how."""

from figkit import components as C
from figkit import icons as I
from figkit import layout as L
from figkit import palette as P
from figkit.svgdoc import Svg, arrow, darken, lighten

from .common import PROBE_CAPTION, WIDTH, atom_legend, footnote, save

TASKS = ["static", "md", "within_gap", "gap_to_ql", "within_ql", "sliding"]
ZERO_SHOT = "MP-0B3 · OMAT-0 · MPA0"


def panel_system(svg: Svg):
    L.panel(svg, 3, 6, "a", "Layered target system")
    C.hero_bilayer(svg, 33, 33, 1.6)
    atom_legend(svg, 20, 59.5)
    svg.text(33, 64, "Cr-doped Sb_{2}Te_{3} (81 atoms) · Cr path from NEB images", size="note", anchor="middle", color=P.MUTED)


def panel_design(svg: Svg):
    x0 = 62
    L.panel(svg, x0, 6, "b", "Two controlled training factors")
    # Inputs: target AIMD (two thermal coverages) and the foundation prior.
    I.database(svg, x0 + 7, 21, 7, 8, "#7d93b8")
    svg.text(x0 + 7, 28.5, "Target AIMD", size="small", anchor="middle")
    svg.text(x0 + 7, 31, "E / F labels", size="note", anchor="middle", color=P.MUTED)
    I.network(svg, x0 + 7, 44, 6.5, P.INIT["foundation"][0], "foundation")
    svg.text(x0 + 7, 51.5, "Foundation prior *θ*_{FM}", size="small", anchor="middle")

    gx, gy, cw, ch = x0 + 26, 13, 15.5, 10.5
    for j, kind in enumerate(("600K", "multiT")):
        cx = gx + 14 + j * cw + cw / 2
        I.thermal_glyph(svg, cx - 4.2, gy - 0.6, 4.2, kind)
        svg.text(cx + 0.2, gy + 0.6, "600 K" if kind == "600K" else "Multi-T", size="small")
    rows = (("scratch", "Scratch", "random init."), ("finetune", "Fine-tuning", "FM init."))
    for i, (key, name, how) in enumerate(rows):
        y = gy + 3.5 + i * (ch + 1.5)
        color = P.INIT[key][0]
        L.container(svg, gx, y, 14 + 2 * cw, ch, fill=lighten(color, 0.9), edge=lighten(color, 0.7))
        svg.text(gx + 1.5, y + 4.6, name, size="small", weight="bold", color=darken(color, 0.1))
        svg.text(gx + 1.5, y + 7.4, how, size="note", color=P.MUTED)
        for j, filled in enumerate((False, True)):
            cx = gx + 14 + j * cw + cw / 2
            for k, shape in enumerate(("circle", "triangle", "square")):
                I.marker(svg, cx - 3 + 3 * k, y + ch / 2, 0.85, shape, color, filled)
    arrow(svg, (x0 + 11.5, 21), (gx - 0.6, gy + 6), bend=-0.15, sw=0.3)
    arrow(svg, (x0 + 11.5, 44), (gx - 0.6, gy + 3.5 + ch + 1.5 + ch / 2), bend=0.12, sw=0.3)
    zy = gy + 3.5 + 2 * (ch + 1.5)
    L.container(svg, gx, zy, 14 + 2 * cw, 7.5, fill=lighten(P.INIT["foundation"][0], 0.9),
                edge=lighten(P.INIT["foundation"][0], 0.7))
    svg.text(gx + 1.5, zy + 3.2, "Zero-shot", size="small", weight="bold", color=darken(P.INIT["foundation"][0], 0.15))
    svg.text(gx + 1.5, zy + 5.9, "no target fitting", size="note", color=P.MUTED)
    for k in range(3):
        I.marker(svg, gx + 18 + 3.2 * k, zy + 3.75, 0.85, "diamond", P.INIT["foundation"][0], False)
    svg.text(gx + 27.5, zy + 4.4, ZERO_SHOT, size=4.2, color=P.MUTED)
    arrow(svg, (x0 + 7, 49.2 + 4.5), (gx - 0.6, zy + 3.75), bend=0.25, sw=0.3, dash="0.8 0.6",
          color=P.INIT["foundation"][0])
    svg.text(gx + 7 + cw * 1.5, gy + 50.5, "3 training seeds per trained cell", size="note", anchor="middle", color=P.MUTED)


def panel_operators(svg: Svg):
    x0 = 135
    L.panel(svg, x0, 6, "c", "Two measurement operators")
    for i, (kind, label, formula, note) in enumerate((
        ("fixed", "Fixed path", "*E*_{θ}(*q*_{s}^{ref})", "same reference geometries"),
        ("self", "Self-relaxed", "*E*_{θ}(*q*_{s}^{θ})", "model-relaxed geometries"),
    )):
        y = 10 + i * 24
        color = P.OPERATOR[kind][0]
        svg.rect(x0, y, 1.2, 21, fill=color)
        L.container(svg, x0 + 1.6, y, 44, 21, rx=1.2)
        svg.text(x0 + 4, y + 4.5, label, size="label", weight="bold")
        I.path_operator(svg, x0 + 4, y + 6.5, 20, 10, kind)
        svg.text(x0 + 33.5, y + 12.8, formula, size=7.5, anchor="middle")
        svg.text(x0 + 4, y + 19.3, note, size="note", color=P.MUTED)


def panel_atlas(svg: Svg):
    y0 = 70
    L.panel(svg, 3, y0, "d", "A task atlas, rather than one global ranking")
    x0, w = 6, WIDTH - 9
    centres = L.task_bar(svg, x0, y0 + 3.5, w, 4.2, TASKS)
    for key, cx in zip(TASKS, centres):
        C.vignette(svg, cx, y0 + 23.0, 1.32, key)
        svg.text(cx, y0 + 38.0, PROBE_CAPTION[key], size="small", anchor="middle", color=darken(P.TASK_INK[key], 0.1))
    ay = y0 + 40.5
    grad = svg.define("taskramp", '<linearGradient id="taskramp">' + "".join(
        f'<stop offset="{i / (len(TASKS) - 1):.2f}" stop-color="{P.TASK[k][0]}"/>' for i, k in enumerate(TASKS)) +
        "</linearGradient>")
    svg.rect(x0, ay, w, 1.3, fill=f"url(#{grad})", rx=0.65)
    svg.text(x0, ay + 4.4, "pointwise", size="note", color=P.MUTED)
    svg.text(x0 + w, ay + 4.4, "collective", size="note", anchor="end", color=P.MUTED)
    svg.text(x0 + w / 2, ay + 4.4, "probes are task labels, not a calibrated scale of distribution shift",
             size="note", anchor="middle", color=P.MUTED, italic=True)


def panel_diagnostics(svg: Svg):
    y0 = 121
    L.panel(svg, 3, y0, "e", "Explaining differences")
    steps = [("Local environments", "SOAP"), ("Shared map", "PCA"), ("Coverage", "distance to training"),
             ("Attribution", "SHAP")]
    bw, gap, x = 36, 9.5, 8
    for i, (title, sub) in enumerate(steps):
        bx = x + i * (bw + gap)
        L.container(svg, bx, y0 + 3, bw, 17, rx=1.5)
        cx, cy = bx + 8.5, y0 + 11.5
        if i == 0:
            I.soap_env(svg, cx, cy, 5.2)
        elif i == 1:
            I.pca_cloud(svg, cx + 1, cy, 15, 13)
        elif i == 2:
            I.distance_glyph(svg, cx + 1, cy, 15, 13)
        else:
            I.shap_bars(svg, cx - 6, cy - 6, 14, 12)
        svg.text(bx + 18.5, y0 + 10.5, title, size="small", weight="bold")
        svg.text(bx + 18.5, y0 + 13.6, sub, size="note", color=P.MUTED)
        if i < len(steps) - 1:
            arrow(svg, (bx + bw + 1.2, y0 + 11.5), (bx + bw + gap - 1.2, y0 + 11.5), sw=0.3)
    svg.text(WIDTH - 4, y0 + 23.5, "→ Fig. 6", size="small", anchor="end", color=P.MUTED)


def build():
    svg = Svg(WIDTH, 150)
    panel_system(svg)
    panel_design(svg)
    panel_operators(svg)
    panel_atlas(svg)
    panel_diagnostics(svg)
    footnote(svg, "Structures and Cr paths are projected from the Cr81 NEB / DFT sliding coordinates; "
                  "the MD vignette's thermal displacements are schematic.")
    save(svg, "fig1_method")


if __name__ == "__main__":
    build()
