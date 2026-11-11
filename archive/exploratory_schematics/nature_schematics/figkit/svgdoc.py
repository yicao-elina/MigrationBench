"""Minimal SVG document in millimetre units, exported to SVG / PDF / PNG.

User units are millimetres (viewBox == physical size), so a 183 mm figure is
183 units wide. Text sizes are given in points and converted here.
"""

import math
import os
import re
from html import escape
from pathlib import Path

from . import palette as P

PT_MM = 25.4 / 72  # 1 pt in mm
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")


def _num(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def mix(color: str, other: str, t: float) -> str:
    """Linear blend of two hex colours (t=0 → color, t=1 → other)."""
    a = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def lighten(color: str, t: float) -> str:
    return mix(color, "#ffffff", t)


def darken(color: str, t: float) -> str:
    return mix(color, "#000000", t)


class Svg:
    def __init__(self, width_mm: float, height_mm: float):
        self.w, self.h = width_mm, height_mm
        self.defs: dict[str, str] = {}
        self.body: list[str] = []

    # -- plumbing ---------------------------------------------------------
    def add(self, xml: str) -> None:
        self.body.append(xml)

    def define(self, key: str, xml: str) -> str:
        self.defs.setdefault(key, xml)
        return key

    def to_string(self) -> str:
        defs = "".join(self.defs.values())
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_num(self.w)}mm" height="{_num(self.h)}mm" '
            f'viewBox="0 0 {_num(self.w)} {_num(self.h)}" font-family="{P.FONT}">'
            f'<rect width="100%" height="100%" fill="#ffffff"/><defs>{defs}</defs>{"".join(self.body)}</svg>'
        )

    def save(self, stem: Path, png_dpi: int = 600) -> None:
        import cairosvg

        stem.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_string().encode()
        stem.with_suffix(".svg").write_bytes(data)
        cairosvg.svg2pdf(bytestring=data, write_to=str(stem.with_suffix(".pdf")))
        cairosvg.svg2png(bytestring=data, write_to=str(stem.with_suffix(".png")), dpi=png_dpi,
                         output_width=round(self.w / 25.4 * png_dpi))

    # -- primitives -------------------------------------------------------
    def rect(self, x, y, w, h, fill="none", stroke="none", sw=0.25, rx=0.0, opacity=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{_num(x)}" y="{_num(y)}" width="{_num(w)}" height="{_num(h)}" rx="{_num(rx)}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{_num(sw)}" opacity="{_num(opacity)}"{d}/>')

    def circle(self, cx, cy, r, fill="none", stroke="none", sw=0.2, opacity=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<circle cx="{_num(cx)}" cy="{_num(cy)}" r="{_num(r)}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{_num(sw)}" opacity="{_num(opacity)}"{d}/>')

    def ellipse(self, cx, cy, rx, ry, fill="none", stroke="none", sw=0.2, opacity=1.0):
        self.add(f'<ellipse cx="{_num(cx)}" cy="{_num(cy)}" rx="{_num(rx)}" ry="{_num(ry)}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{_num(sw)}" opacity="{_num(opacity)}"/>')

    def line(self, x1, y1, x2, y2, stroke=P.LINE, sw=0.25, dash=None, cap="round", opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{_num(x1)}" y1="{_num(y1)}" x2="{_num(x2)}" y2="{_num(y2)}" stroke="{stroke}" '
                 f'stroke-width="{_num(sw)}" stroke-linecap="{cap}" opacity="{_num(opacity)}"{d}/>')

    def path(self, d, fill="none", stroke=P.LINE, sw=0.25, dash=None, opacity=1.0, join="round", cap="round"):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{_num(sw)}" '
                 f'stroke-linejoin="{join}" stroke-linecap="{cap}" opacity="{_num(opacity)}"{dd}/>')

    def polyline(self, pts, **kw):
        if len(pts) < 2:
            return
        self.path("M " + " L ".join(f"{_num(x)} {_num(y)}" for x, y in pts), **kw)

    def text(self, x, y, s, size="label", weight="normal", anchor="start", color=P.INK, italic=False,
             rotate=0.0, baseline="auto"):
        """Text with markup x_{sub}, x^{sup}, *italic*.

        Markup is laid out as separate, metric-positioned runs (cairosvg mis-places
        tspan offsets); plain strings stay a single editable text element.
        """
        pt = P.SIZE[size] if isinstance(size, str) else size
        fs = pt * PT_MM
        runs = _runs(s, fs, italic)
        rot = f' transform="rotate({_num(rotate)} {_num(x)} {_num(y)})"' if rotate else ""
        dom = f' dominant-baseline="{baseline}"' if baseline != "auto" else ""
        if len(runs) == 1 and runs[0][2] == 0:
            style = ' font-style="italic"' if runs[0][1] else ""
            self.add(f'<text x="{_num(x)}" y="{_num(y)}" font-size="{_num(fs)}" font-weight="{weight}" '
                     f'text-anchor="{anchor}" fill="{color}"{style}{rot}{dom}>{escape(runs[0][0])}</text>')
            return
        widths = [_measure(t, size_mm, ital, weight) for t, ital, _dy, size_mm in runs]
        total = sum(widths)
        cx = x - (total if anchor == "end" else total / 2 if anchor == "middle" else 0)
        self.add(f"<g{rot}>")
        for (t, ital, dy, size_mm), w in zip(runs, widths):
            style = ' font-style="italic"' if ital else ""
            self.add(f'<text xml:space="preserve" x="{_num(cx)}" y="{_num(y + dy)}" font-size="{_num(size_mm)}" '
                     f'font-weight="{weight}" fill="{color}"{style}{dom}>{escape(t)}</text>')
            cx += w
        self.add("</g>")

    def group_open(self, transform="", opacity=1.0):
        t = f' transform="{transform}"' if transform else ""
        self.add(f'<g{t} opacity="{_num(opacity)}">')

    def group_close(self):
        self.add("</g>")


_TOKEN = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\}|\*[^*]+\*)")
_FONT_DIR = Path("/System/Library/Fonts/Supplemental")
_FONT_FILES = {(False, False): "Arial.ttf", (False, True): "Arial Italic.ttf",
               (True, False): "Arial Bold.ttf", (True, True): "Arial Bold Italic.ttf"}
_font_cache: dict = {}


def _measure(text: str, size_mm: float, italic: bool, weight: str) -> float:
    """Advance width in mm using the real Arial metrics."""
    from PIL import ImageFont

    key = (weight == "bold", italic)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(str(_FONT_DIR / _FONT_FILES[key]), 1000)
    return _font_cache[key].getlength(text) / 1000 * size_mm


def _runs(s: str, fs: float, base_italic: bool):
    """Split markup into (text, italic, baseline offset mm, size mm) runs."""
    runs = []
    for part in _TOKEN.split(s):
        if not part:
            continue
        if part.startswith("_{"):
            runs.append((part[2:-1], base_italic, 0.3 * fs, fs * 0.68))
        elif part.startswith("^{"):
            runs.append((part[2:-1], base_italic, -0.38 * fs, fs * 0.68))
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            runs.append((part[1:-1], True, 0.0, fs))
        else:
            runs.append((part, base_italic, 0.0, fs))
    return runs or [("", base_italic, 0.0, fs)]


# -- arrows ----------------------------------------------------------------
def arrow_head(svg: Svg, tip, direction, size=1.3, color=P.LINE):
    ang = math.atan2(direction[1], direction[0])
    wing = 0.42
    p1 = (tip[0] - size * math.cos(ang - wing), tip[1] - size * math.sin(ang - wing))
    p2 = (tip[0] - size * math.cos(ang + wing), tip[1] - size * math.sin(ang + wing))
    back = (tip[0] - 0.72 * size * math.cos(ang), tip[1] - 0.72 * size * math.sin(ang))
    svg.path(f"M {_num(tip[0])} {_num(tip[1])} L {_num(p1[0])} {_num(p1[1])} L {_num(back[0])} {_num(back[1])} "
             f"L {_num(p2[0])} {_num(p2[1])} Z", fill=color, stroke="none")


def arrow(svg: Svg, start, end, bend=0.0, color=P.LINE, sw=0.3, head=1.3, dash=None, both=False):
    """Straight (bend=0) or quadratic curved arrow; bend is a fraction of length."""
    (x1, y1), (x2, y2) = start, end
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    nx, ny = -(y2 - y1), x2 - x1
    cx, cy = mx + bend * nx, my + bend * ny
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    trim = 0.6 * head / length
    ex, ey = x2 - (x2 - cx) * trim * 1.6, y2 - (y2 - cy) * trim * 1.6
    sx, sy = (x1 + (cx - x1) * trim * 1.6, y1 + (cy - y1) * trim * 1.6) if both else (x1, y1)
    svg.path(f"M {_num(sx)} {_num(sy)} Q {_num(cx)} {_num(cy)} {_num(ex)} {_num(ey)}", stroke=color, sw=sw, dash=dash)
    arrow_head(svg, (x2, y2), (x2 - cx, y2 - cy), head, color)
    if both:
        arrow_head(svg, (x1, y1), (x1 - cx, y1 - cy), head, color)
