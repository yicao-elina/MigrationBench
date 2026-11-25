"""Layout helpers: panel letters, headings, containers, chips, legends."""

from . import icons
from . import palette as P
from .svgdoc import Svg, darken


def panel(svg: Svg, x, y, letter, title=""):
    svg.text(x, y, letter, size="letter", weight="bold")
    if title:
        svg.text(x + 4.2, y, title, size="title")


def container(svg: Svg, x, y, w, h, fill=P.CONTAINER, edge=P.CONTAINER_EDGE, rx=2.0):
    svg.rect(x, y, w, h, fill=fill, stroke=edge, sw=0.15, rx=rx)


def chip(svg: Svg, x, y, w, h, fill, label, color=P.INK, weight="bold", size="small"):
    svg.rect(x, y, w, h, fill=fill, rx=0.8)
    svg.text(x + w / 2, y + h / 2 + 0.7, label, size=size, weight=weight, anchor="middle", color=color)


def task_bar(svg: Svg, x, y, w, h, keys, gap=1.2):
    """Row of task chips on the pink→purple ramp; returns chip centres."""
    n = len(keys)
    cw = (w - gap * (n - 1)) / n
    centres = []
    for i, k in enumerate(keys):
        fill, label = P.TASK[k]
        dark = i >= n - 2
        chip(svg, x + i * (cw + gap), y, cw, h, fill, label, color=P.WHITE if dark else darken(P.TASK_INK[k], 0.15))
        centres.append(x + i * (cw + gap) + cw / 2)
    return centres


def legend_init_thermal(svg: Svg, x, y, row=4.2):
    """Initialization colours (open = 600 K, filled = Multi-T) and seed shapes."""
    svg.text(x, y, "Initialization", size="small", weight="bold")
    for i, (key, (color, name, _)) in enumerate(P.INIT.items()):
        yy = y + row * (i + 1)
        icons.marker(svg, x + 1.2, yy - 0.8, 0.9, "circle", color, False)
        icons.marker(svg, x + 4.0, yy - 0.8, 0.9, "circle", color, True)
        svg.text(x + 6.2, yy, name, size="small")
    xx = x + 30
    svg.text(xx, y, "Thermal coverage", size="small", weight="bold")
    icons.marker(svg, xx + 1.2, y + row - 0.8, 0.9, "circle", P.INK, False)
    svg.text(xx + 3.4, y + row, "600 K (single T)", size="small")
    icons.marker(svg, xx + 1.2, y + 2 * row - 0.8, 0.9, "circle", P.INK, True)
    svg.text(xx + 3.4, y + 2 * row, "Multi-T", size="small")
    xs = x + 62
    svg.text(xs, y, "Training seed", size="small", weight="bold")
    for i, (shape, label) in enumerate((("circle", "s123"), ("triangle", "s234"), ("square", "s345"), ("diamond", "zero-shot"))):
        icons.marker(svg, xs + 1.2, y + row * (i + 1) - 0.8, 0.9, shape, P.MUTED, False)
        svg.text(xs + 3.4, y + row * (i + 1), label, size="small")
