"""Vector glyphs: model initializations, thermal coverage, operators, diagnostics."""

import math

import numpy as np

from . import palette as P
from .svgdoc import Svg, _num, arrow, darken, lighten


def _node(svg: Svg, x, y, r, color, filled=True):
    if filled:
        svg.circle(x, y, r, fill=color, stroke=darken(color, 0.2), sw=0.12)
    else:
        svg.circle(x, y, r, fill=P.WHITE, stroke=color, sw=0.22)


def network(svg: Svg, cx, cy, size, color, style="foundation"):
    """Neural-network glyph. foundation: stacked prior; finetune: prior + target arrow; scratch: random init."""
    layers = [2, 3, 2]
    w, h = size, size * 0.8
    if style in ("foundation", "finetune"):
        for k in (2, 1):  # stacked "pretrained" sheets behind
            off = 0.9 * k * size / 10
            svg.rect(cx - w / 2 - 1 + off, cy - h / 2 - 0.8 - off, w + 2, h + 1.6, rx=1.0,
                     fill=lighten(color, 0.82 - 0.06 * k), stroke=lighten(color, 0.55), sw=0.12)
    svg.rect(cx - w / 2 - 1, cy - h / 2 - 0.8, w + 2, h + 1.6, rx=1.0, fill=P.WHITE,
             stroke=lighten(color, 0.35), sw=0.18)
    pts = []
    for li, n in enumerate(layers):
        x = cx - w / 2 + w * li / (len(layers) - 1)
        pts.append([(x, cy - h / 2 + h * (k + 0.5) / n) for k in range(n)])
    rng = np.random.default_rng(3)
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                if style == "scratch":
                    svg.line(*p, *q, stroke=lighten(color, 0.3), sw=0.12 + 0.12 * rng.random(), dash="0.5 0.4")
                else:
                    svg.line(*p, *q, stroke=lighten(color, 0.25), sw=0.16)
    r = size * 0.085
    for layer in pts:
        for p in layer:
            _node(svg, *p, r, color, filled=style != "scratch")


def database(svg: Svg, cx, cy, w, h, color):
    """Stacked cylinder (dataset)."""
    disc = h / 3
    ry = w * 0.18
    for k in range(3):
        y = cy + h / 2 - (k + 1) * disc
        c = lighten(color, 0.12 * k)
        svg.path(f"M {_num(cx - w / 2)} {_num(y)} L {_num(cx - w / 2)} {_num(y + disc)} "
                 f"A {_num(w / 2)} {_num(ry)} 0 0 0 {_num(cx + w / 2)} {_num(y + disc)} "
                 f"L {_num(cx + w / 2)} {_num(y)} Z", fill=darken(c, 0.08), stroke=P.WHITE, sw=0.12)
        svg.ellipse(cx, y, w / 2, ry, fill=lighten(c, 0.25), stroke=P.WHITE, sw=0.12)


def thermometer(svg: Svg, cx, cy, h, level, color):
    stem_w, bulb = h * 0.22, h * 0.2
    top, bottom = cy - h / 2, cy + h / 2 - bulb
    svg.rect(cx - stem_w / 2, top, stem_w, bottom - top + bulb * 0.5, rx=stem_w / 2, fill=P.WHITE, stroke=P.LINE, sw=0.16)
    fill_top = bottom - (bottom - top - stem_w * 0.3) * level
    svg.rect(cx - stem_w * 0.22, fill_top, stem_w * 0.44, bottom - fill_top + bulb * 0.4, rx=stem_w * 0.2, fill=color)
    svg.circle(cx, bottom + bulb * 0.35, bulb, fill=color, stroke=P.LINE, sw=0.16)


def thermal_glyph(svg: Svg, cx, cy, h, kind):
    if kind == "600K":
        thermometer(svg, cx, cy, h, 0.6, "#e07a4f")
        return
    for dx, lvl, col in ((-h * 0.34, 0.3, "#5b8fd1"), (0, 0.6, "#e0a64f"), (h * 0.34, 0.9, "#d4544a")):
        thermometer(svg, cx + dx, cy, h * 0.92, lvl, col)


def marker(svg: Svg, x, y, r, shape, color, filled):
    fill = color if filled else P.WHITE
    sw = 0.22
    if shape == "circle":
        svg.circle(x, y, r, fill=fill, stroke=color, sw=sw)
    elif shape == "square":
        svg.rect(x - r * 0.88, y - r * 0.88, r * 1.76, r * 1.76, fill=fill, stroke=color, sw=sw)
    elif shape == "triangle":
        svg.path(f"M {_num(x)} {_num(y - r * 1.05)} L {_num(x + r * 1.0)} {_num(y + r * 0.7)} "
                 f"L {_num(x - r * 1.0)} {_num(y + r * 0.7)} Z", fill=fill, stroke=color, sw=sw)
    elif shape == "diamond":
        svg.path(f"M {_num(x)} {_num(y - r * 1.1)} L {_num(x + r * 1.1)} {_num(y)} L {_num(x)} {_num(y + r * 1.1)} "
                 f"L {_num(x - r * 1.1)} {_num(y)} Z", fill=fill, stroke=color, sw=sw)


def path_operator(svg: Svg, x, y, w, h, kind):
    """Mini NEB-path glyph. fixed: reference images scored as given; self: images relaxed by the model."""
    n = 6
    xs = [x + w * (0.08 + 0.84 * i / (n - 1)) for i in range(n)]
    ref = [y + h * (0.78 - 0.5 * math.sin(math.pi * i / (n - 1))) for i in range(n)]
    color = P.OPERATOR[kind][0] if kind == "fixed" else P.OPERATOR["self"][0]
    svg.polyline(list(zip(xs, ref)), stroke=P.MUTED if kind == "fixed" else P.FAINT, sw=0.25,
                 dash=None if kind == "fixed" else "0.6 0.5")
    if kind == "fixed":
        for px, py in zip(xs, ref):
            svg.circle(px, py, 0.62, fill=P.ATOM["Cr"][0], stroke=darken(P.ATOM["Cr"][0], 0.25), sw=0.12)
        return
    moved = [py + (h * 0.16 * math.sin(math.pi * i / (n - 1))) for i, py in enumerate(ref)]
    svg.polyline(list(zip(xs, moved)), stroke=darken(color, 0.1), sw=0.3)
    for i, (px, py, my) in enumerate(zip(xs, ref, moved)):
        if 0 < i < n - 1:
            arrow(svg, (px, py + 0.3), (px, my - 0.5), color=darken(color, 0.15), sw=0.16, head=0.55)
        svg.circle(px, my, 0.62, fill=P.ATOM["Cr"][0], stroke=darken(P.ATOM["Cr"][0], 0.25), sw=0.12)


def pca_cloud(svg: Svg, cx, cy, w, h, seed=11, task_color=P.TASK_INK["within_ql"]):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (70, 2)) @ np.array([[1.0, 0.35], [0.0, 0.55]])
    for px, py in pts:
        svg.circle(cx - w * 0.12 + px * w * 0.16, cy + py * h * 0.2, 0.32, fill="#9aa3ae", opacity=0.75)
    for px, py in rng.normal(0, 0.35, (9, 2)):
        svg.circle(cx + w * 0.3 + px * w * 0.06, cy - h * 0.22 + py * h * 0.1, 0.42, fill=task_color)


def soap_env(svg: Svg, cx, cy, r):
    gid = svg.define("soapg", '<radialGradient id="soapg"><stop offset="0" stop-color="#e9a3c0" stop-opacity="0.55"/>'
                              '<stop offset="1" stop-color="#e9a3c0" stop-opacity="0"/></radialGradient>')
    svg.circle(cx, cy, r, fill=f"url(#{gid})")
    for k in (0.45, 0.75, 1.0):
        svg.circle(cx, cy, r * k, stroke=lighten(P.TASK_INK["within_gap"], 0.3), sw=0.12, dash="0.4 0.35")
    for ang in range(0, 360, 60):
        a = math.radians(ang + 15)
        svg.circle(cx + 0.62 * r * math.cos(a), cy + 0.62 * r * math.sin(a), r * 0.12,
                   fill=P.ATOM["Te"][0] if ang % 120 else P.ATOM["Sb"][0])
    svg.circle(cx, cy, r * 0.16, fill=P.ATOM["Cr"][0])


def shap_bars(svg: Svg, x, y, w, h, rows=4, seed=5):
    rng = np.random.default_rng(seed)
    axis_x = x + w * 0.35
    svg.line(axis_x, y, axis_x, y + h, stroke=P.FAINT, sw=0.16)
    for k in range(rows):
        yy = y + h * (k + 0.5) / rows
        svg.line(x, yy, x + w * 0.2, yy, stroke=P.FAINT, sw=0.5, cap="butt")
        for _ in range(9):
            v = rng.normal(0.18 * (rows - k), 0.12)
            t = rng.random()
            col = "#2a9d9a" if t < 0.5 else "#e46aa2"
            svg.circle(axis_x + v * w * 0.4, yy + rng.normal(0, h / rows * 0.12), 0.32, fill=col, opacity=0.85)


def distance_glyph(svg: Svg, cx, cy, w, h):
    rng = np.random.default_rng(2)
    for px, py in rng.normal(0, 1, (40, 2)):
        svg.circle(cx - w * 0.22 + px * w * 0.1, cy + py * h * 0.14, 0.3, fill="#9aa3ae", opacity=0.8)
    tx, ty = cx + w * 0.3, cy - h * 0.15
    svg.line(cx - w * 0.12, cy - h * 0.04, tx, ty, stroke=P.TASK_INK["within_ql"], sw=0.2, dash="0.6 0.4")
    svg.circle(tx, ty, 0.55, fill=P.TASK_INK["within_ql"])
    svg.text((cx - w * 0.12 + tx) / 2 + 0.5, (cy + ty) / 2 - 1.0, "*d*_{train}", size="small", color=P.TASK_INK["within_ql"])


def plot_slot(svg: Svg, x, y, w, h, xlabel="", ylabel="", note="data panel"):
    """Empty axes frame reserving space for a quantitative plot made from source data."""
    svg.rect(x, y, w, h, fill="#fafbfc", rx=0.6)
    svg.line(x + 4.5, y + 1.5, x + 4.5, y + h - 4.0, stroke=P.MUTED, sw=0.2)
    svg.line(x + 4.5, y + h - 4.0, x + w - 1.5, y + h - 4.0, stroke=P.MUTED, sw=0.2)
    if xlabel:
        svg.text(x + 4.5 + (w - 6) / 2, y + h - 1.0, xlabel, size="note", anchor="middle", color=P.MUTED)
    if ylabel:
        svg.text(x + 2.2, y + (h - 4) / 2 + 1, ylabel, size="note", anchor="middle", color=P.MUTED, rotate=-90)
    svg.text(x + 4.5 + (w - 6) / 2, y + (h - 4) / 2 + 1.5, note, size="note", anchor="middle", color=P.FAINT, italic=True)


def check(svg: Svg, cx, cy, r, color=P.STATUS["complete"]):
    svg.circle(cx, cy, r, fill=color)
    svg.path(f"M {_num(cx - r * 0.45)} {_num(cy + r * 0.02)} L {_num(cx - r * 0.1)} {_num(cy + r * 0.38)} "
             f"L {_num(cx + r * 0.5)} {_num(cy - r * 0.35)}", stroke=P.WHITE, sw=r * 0.28)


def cross(svg: Svg, cx, cy, r, color=P.STATUS["failed"]):
    svg.circle(cx, cy, r, fill=color)
    k = r * 0.4
    svg.line(cx - k, cy - k, cx + k, cy + k, stroke=P.WHITE, sw=r * 0.28)
    svg.line(cx - k, cy + k, cx + k, cy - k, stroke=P.WHITE, sw=r * 0.28)
