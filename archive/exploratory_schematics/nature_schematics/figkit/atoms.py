"""Matte 3D atom renderer: real coordinates → depth-sorted vector spheres.

Look: soft key light from upper-left, no specular hotspot, gentle rim
darkening, aerial-perspective fade with depth and a soft ground shadow. All
pure SVG gradients (no filters), so PDF/Illustrator keep it vector.
"""

from dataclasses import dataclass, field

import numpy as np

from . import palette as P
from .svgdoc import Svg, _num, darken, lighten, mix


def sphere_fill(svg: Svg, color: str) -> str:
    gid = f"sph{color[1:]}"
    svg.define(
        gid,
        f'<radialGradient id="{gid}" cx="0.5" cy="0.5" r="0.5" fx="0.34" fy="0.3">'
        f'<stop offset="0" stop-color="{lighten(color, 0.62)}"/>'
        f'<stop offset="0.42" stop-color="{lighten(color, 0.2)}"/>'
        f'<stop offset="0.82" stop-color="{color}"/>'
        f'<stop offset="1" stop-color="{darken(color, 0.24)}"/></radialGradient>',
    )
    return f"url(#{gid})"


def glow_fill(svg: Svg, color: str) -> str:
    gid = f"glow{color[1:]}"
    svg.define(
        gid,
        f'<radialGradient id="{gid}" cx="0.5" cy="0.5" r="0.5">'
        f'<stop offset="0" stop-color="{color}" stop-opacity="0.55"/>'
        f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></radialGradient>',
    )
    return f"url(#{gid})"


def shadow_fill(svg: Svg) -> str:
    svg.define(
        "softshadow",
        '<radialGradient id="softshadow" cx="0.5" cy="0.5" r="0.5">'
        '<stop offset="0" stop-color="#1f2329" stop-opacity="0.2"/>'
        '<stop offset="0.6" stop-color="#1f2329" stop-opacity="0.07"/>'
        '<stop offset="1" stop-color="#1f2329" stop-opacity="0"/></radialGradient>',
    )
    return "url(#softshadow)"


@dataclass
class View:
    azimuth: float = -28.0  # degrees about z
    elevation: float = 16.0  # degrees camera tilt (look down)
    scale: float = 1.0  # mm per Å

    def project(self, r: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Return (screen_x, screen_up, depth) in Å for positions r (N×3)."""
        a, e = np.radians(self.azimuth), np.radians(self.elevation)
        v = r - center
        x = v[:, 0] * np.cos(a) - v[:, 1] * np.sin(a)
        y = v[:, 0] * np.sin(a) + v[:, 1] * np.cos(a)
        z = v[:, 2]
        up = z * np.cos(e) + y * np.sin(e)
        depth = y * np.cos(e) - z * np.sin(e)
        return np.column_stack([x, up, depth])


@dataclass
class Atom:
    pos: np.ndarray
    species: str
    alpha: float = 1.0
    scale: float = 1.0
    color: str | None = None
    ghost: bool = False  # dashed outline, no fill
    ring: bool = False  # transition-state ring
    glow: bool = False


@dataclass
class Scene:
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[tuple[int, int]] = field(default_factory=list)
    trails: list[tuple] = field(default_factory=list)  # (N×3 path, colour, width mm[, dashed])

    def add(self, *args, **kw) -> int:
        self.atoms.append(Atom(*args, **kw))
        return len(self.atoms) - 1


def render(svg: Svg, scene: Scene, origin: tuple[float, float], view: View, center: np.ndarray,
           fog: float = 0.32, shadow: bool = True, bond_color: str = "#8d939c", size: float = 1.0,
           hero_on_top: bool = True) -> dict:
    """Draw a scene; returns screen helpers {"to_screen": fn} for annotations."""
    ox, oy = origin
    pos = np.array([a.pos for a in scene.atoms])
    proj = view.project(pos, center)
    dmin, dmax = proj[:, 2].min(), proj[:, 2].max()
    span = max(dmax - dmin, 1e-6)

    def to_screen(r):
        p = view.project(np.atleast_2d(r), center)[0]
        return ox + p[0] * view.scale, oy - p[1] * view.scale, p[2]

    if shadow:
        xs, ups = proj[:, 0] * view.scale, proj[:, 1] * view.scale
        w = xs.max() - xs.min()
        svg.ellipse(ox + (xs.max() + xs.min()) / 2, oy - ups.min() + 1.4 * view.scale, w * 0.56,
                    1.6 * view.scale, fill=shadow_fill(svg))

    # Painter's order: far → near; the hero (Cr) and its images always on top, in sequence.
    items = []  # (sort key, depth, kind, payload)
    for i, a in enumerate(scene.atoms):
        key = 1e6 + i if (hero_on_top and a.species == "Cr") else -proj[i, 2]
        items.append((key, proj[i, 2], "atom", i))
    for i, j in scene.bonds:
        d = (proj[i, 2] + proj[j, 2]) / 2 + 0.01
        items.append((-d, d, "bond", (i, j)))
    for k, (path, *_rest) in enumerate(scene.trails):
        d = view.project(path, center)[:, 2].min() - 0.05
        items.append((1e6 - 1 if hero_on_top else -d, d, "trail", k))
    items.sort(key=lambda t: t[0])

    for _key, depth, kind, payload in items:
        f = fog * min(max((depth - dmin) / span, 0.0), 1.0)
        if kind == "bond":
            i, j = payload
            x1, y1 = ox + proj[i, 0] * view.scale, oy - proj[i, 1] * view.scale
            x2, y2 = ox + proj[j, 0] * view.scale, oy - proj[j, 1] * view.scale
            w = 0.36 * view.scale
            svg.line(x1, y1, x2, y2, stroke=darken(lighten(bond_color, f), 0.18), sw=w)
            svg.line(x1, y1, x2, y2, stroke=lighten(bond_color, 0.35 + f * 0.5), sw=w * 0.45)
            continue
        if kind == "trail":
            path, color, width, *style = scene.trails[payload]
            dashed = style[0] if style else True
            pts = [(ox + p[0] * view.scale, oy - p[1] * view.scale) for p in view.project(path, center)]
            svg.polyline(pts, stroke=color, sw=width, dash=f"{_num(width * 1.6)} {_num(width * 1.8)}" if dashed else None)
            continue
        a = scene.atoms[payload]
        base, radius = P.ATOM[a.species]
        color = a.color or base
        r = radius * size * a.scale * view.scale * (1 - 0.1 * f / max(fog, 1e-6))
        cx, cy = ox + proj[payload, 0] * view.scale, oy - proj[payload, 1] * view.scale
        if a.glow:
            svg.circle(cx, cy, r * 2.3, fill=glow_fill(svg, color))
        if a.ghost:
            svg.circle(cx, cy, r, fill=lighten(color, 0.75), stroke=darken(color, 0.1), sw=0.12,
                       dash=f"{_num(r * 0.35)} {_num(r * 0.25)}", opacity=a.alpha)
        else:
            svg.circle(cx, cy, r, fill=sphere_fill(svg, mix(color, P.WHITE, f)), opacity=a.alpha)
        if a.ring:
            svg.circle(cx, cy, r * 1.55, stroke=P.INK, sw=0.18, dash="0.5 0.35")
            svg.text(cx + r * 1.55, cy - r * 1.25, "‡", size="small", weight="bold")
    return {"to_screen": to_screen}
