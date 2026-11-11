"""Structure components built from real coordinates (plus labelled schematic variants)."""

import numpy as np

from . import palette as P
from . import structures as st
from .atoms import Scene, View, render
from .svgdoc import Svg, arrow, darken, lighten, mix

SLAB_VIEW = dict(azimuth=0.0, elevation=7.0)
ATOM_SIZE = 1.3
CR = P.ATOM["Cr"][0]
GAP_BAND = (22.65, 24.75)
UPPER_QL, LOWER_QL = (25.1, 32.4), (15.0, 22.3)


def cr_ramp(t: float) -> str:
    return mix(lighten(CR, 0.62), darken(CR, 0.3), t)


def _screen(view: View, center, origin, r):
    p = view.project(np.atleast_2d(np.asarray(r, float)), center)[0]
    return origin[0] + p[0] * view.scale, origin[1] - p[1] * view.scale


def gap_guides(svg: Svg, view: View, center, origin, half_x, labels=True, label_side="left"):
    """Pale vdW-gap band behind the slab and QL / gap brackets."""
    x0, _ = _screen(view, center, origin, center + [-half_x - 0.6, 0, 0])
    x1, _ = _screen(view, center, origin, center + [half_x + 0.6, 0, 0])
    _, yb = _screen(view, center, origin, [center[0], center[1], GAP_BAND[0]])
    _, yt = _screen(view, center, origin, [center[0], center[1], GAP_BAND[1]])
    svg.rect(x0, yt, x1 - x0, yb - yt, fill=P.GAP_TINT, opacity=0.75, rx=0.4)
    if not labels:
        return
    bx = x0 - 1.2 if label_side == "left" else x1 + 1.2
    sign = -1 if label_side == "left" else 1
    anchor = "end" if label_side == "left" else "start"
    for (z0, z1), text, color in ((UPPER_QL, "QL", P.MUTED), ((GAP_BAND[0], GAP_BAND[1]), "vdW gap", "#a07a1f"),
                                  (LOWER_QL, "QL", P.MUTED)):
        _, ya = _screen(view, center, origin, [center[0], center[1], z0])
        _, yb2 = _screen(view, center, origin, [center[0], center[1], z1])
        svg.path(f"M {bx - sign * 0.6} {ya} L {bx} {ya} L {bx} {yb2} L {bx - sign * 0.6} {yb2}", stroke=color, sw=0.18)
        svg.text(bx + sign * 0.8, (ya + yb2) / 2 + 0.7, text, size="note", anchor=anchor, color=color)


def bilayer(svg: Svg, cx, cy, scale, key="within_gap", image=0, half_x=9.5, y_range=(-5.5, 1.2),
            z_range=(14.5, 33.0), guides=True, labels=True, decorate=None, offsets=None, include_cr=True,
            label_side="left", shadow=True):
    """Slab of the real structure; `decorate(scene, info)` may add atoms/trails before rendering."""
    scene, center, info = st.slab(key, image, half_x=half_x, y_range=y_range, z_range=z_range,
                                  offsets=offsets, include_cr=include_cr)
    view = View(scale=scale, **SLAB_VIEW)
    origin = (cx, cy)
    if guides:
        gap_guides(svg, view, center, origin, half_x, labels=labels, label_side=label_side)
    if decorate:
        decorate(scene, info)
    render(svg, scene, origin, view, center, size=ATOM_SIZE, shadow=shadow)
    return {"view": view, "center": center, "origin": origin, "info": info, "half_x": half_x,
            "screen": lambda r: _screen(view, center, origin, r)}


# ------------------------------------------------------------------ heroes --
def hero_bilayer(svg: Svg, cx, cy, scale):
    """Fig. 1a: Cr in the gap, with its real journey (in-gap → into the QL) as a dotted trail."""
    gap, ql = st.paths()["within_gap"], st.paths()["within_ql"]
    journey = np.concatenate([gap["images"][:, st.cr_index("within_gap")], ql["images"][:, st.cr_index("within_ql")]])

    def deco(scene: Scene, _):
        scene.trails.append((journey, darken(CR, 0.2), 0.26))
        scene.add(journey[-1], "Cr", ghost=True, scale=1.08)

    out = bilayer(svg, cx, cy, scale, decorate=deco)
    tip = out["screen"](journey[-1])
    prev = out["screen"](journey[-3])
    arrow(svg, prev, (tip[0], tip[1] + 0.9), color=darken(CR, 0.25), sw=0.3, head=1.0)
    return out


def vignette(svg: Svg, cx, cy, scale, task, labels=False):
    """Small real-structure vignette showing one task's motion cue."""
    half_x, zr = 6.2, (17.5, 30.5)
    key = {"static": "within_gap", "md": "within_gap", "sliding": "sliding_cr",
           "sliding_pristine": "sliding_pristine"}.get(task, task)
    peaks = st.dft_peak_image()
    offsets, deco = None, None

    if task in ("within_gap", "within_ql"):
        def deco(scene, _):
            st.stroboscope(scene, task, cr_ramp, ring_image=peaks.get(task), trail_color=darken(CR, 0.2))
    elif task == "md":
        n = len(st.tiled(key, 0)[0])
        offsets = st.thermal_offsets(n, 0.16, seed=4)  # schematic thermal motion

        def deco(scene, _):
            r = st.paths()[key]["images"][0][st.cr_index(key)]
            for k, d in enumerate(st.thermal_offsets(4, 0.45, seed=9)):
                scene.add(r + d, "Cr", color=cr_ramp(0.15 + 0.2 * k), scale=0.95, alpha=0.55)
    out = bilayer(svg, cx, cy, scale, key=key, half_x=half_x, z_range=zr, labels=labels, decorate=deco,
                  offsets=offsets, include_cr=task not in ("within_gap", "within_ql"), shadow=False)

    if task == "md":
        x, y = out["screen"](st.paths()[key]["images"][0][st.cr_index(key)])
        svg.ellipse(x, y, 2.4 * scale, 1.5 * scale, stroke=darken(CR, 0.2), sw=0.18)
        svg.ellipse(x, y, 3.3 * scale, 2.1 * scale, stroke=lighten(CR, 0.2), sw=0.14)
    if task == "static":
        x, y = out["screen"](st.paths()[key]["images"][0][st.cr_index(key)])
        svg.circle(x, y, 1.9 * scale, stroke=P.INK, sw=0.18, dash="0.5 0.35")
    elif task == "gap_to_ql":
        host_arrows(svg, out, key, magnify=10.0, color=P.TASK_INK["gap_to_ql"])
        x, y = out["screen"]([out["center"][0] + 6.2, out["center"][1], 30.5])
        svg.text(x, y, "×10", size="note", anchor="end", color=P.TASK_INK["gap_to_ql"])
    elif task in ("sliding", "sliding_pristine"):
        shear_arrow(svg, out, key)
    return out


def host_arrows(svg: Svg, out, key, magnify, color, radius=6.0, min_disp=0.06):
    info = out["info"]
    pos0 = info["pos"]
    pos1, _ = st.tiled(key, -1)
    r_cr = st.paths()[key]["images"][0][st.cr_index(key)]
    for k in np.flatnonzero(info["keep"]):
        d = pos1[k] - pos0[k]
        if np.linalg.norm(d) < min_disp or np.linalg.norm(pos0[k] - r_cr) > radius:
            continue
        a = out["screen"](pos0[k])
        b = out["screen"](pos0[k] + magnify * d)
        arrow(svg, a, b, color=color, sw=0.28, head=0.75)


def shear_arrow(svg: Svg, out, key, streaks=True):
    imgs = st.paths()[key]["images"]
    upper = imgs[0][:, 2] > UPPER_QL[0] - 0.3
    shift = (imgs[-1][upper] - imgs[0][upper]).mean(axis=0)
    info = out["info"]
    if streaks:
        tiled_all = [st.tiled(key, i)[0] for i in range(len(imgs))]
        half_x = out["half_x"]
        for k in np.flatnonzero(info["keep"]):
            if upper[info["idx"][k]]:
                inside = [t[k] for t in tiled_all if abs(t[k][0] - out["center"][0]) <= half_x]
                svg.polyline([out["screen"](r) for r in inside], stroke=P.TASK_INK["sliding"], sw=0.16, opacity=0.7)
    c = out["center"]
    top = UPPER_QL[1] + 1.3
    a = out["screen"]([c[0] + 3.5, c[1], top])
    b = out["screen"]([c[0] - 3.5, c[1], top])
    if shift[0] > 0:
        a, b = b, a
    arrow(svg, a, b, color=P.TASK_INK["sliding"], sw=0.5, head=1.3)
    return float(np.linalg.norm(shift[:2]))


def md_frame(svg: Svg, cx, cy, scale, amplitude, seed, failed=False):
    """SCHEMATIC MD snapshot: thermal offsets on the real lattice; `failed` disorders around Cr."""
    key = "within_gap"
    pos, idx = st.tiled(key, 0)
    offsets = st.thermal_offsets(len(pos), amplitude, seed)
    if failed:
        r_cr = st.paths()[key]["images"][0][st.cr_index(key)]
        near = np.linalg.norm(pos - r_cr, axis=1)
        offsets = offsets + st.thermal_offsets(len(pos), 1.0, seed + 1) * np.clip(1.4 - near / 6.0, 0, 1)[:, None]
    return bilayer(svg, cx, cy, scale, key=key, half_x=6.2, z_range=(17.5, 30.5), labels=False,
                   offsets=offsets, shadow=False)


def local_env(svg: Svg, cx, cy, scale, annotate=True):
    scene, r_cr, dists = st.local_environment()
    view = View(azimuth=-22, elevation=16, scale=scale)
    render(svg, scene, (cx, cy), view, r_cr, fog=0.2, size=1.0)
    if annotate:
        a = _screen(view, r_cr, (cx, cy), r_cr)
        ends = [_screen(view, r_cr, (cx, cy), at.pos) for at in scene.atoms[1:]]
        b = max(ends, key=lambda e: e[0] - abs(e[1] - a[1]))  # right-most, near-horizontal bond
        svg.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 1.4, "*r*_{ij}", size="small", anchor="middle", color=P.INK)
    return dists


def force_cluster(svg: Svg, cx, cy, scale, seed=6):
    """Cr first shell with per-atom force arrows (directions SCHEMATIC)."""
    scene, r_cr, _ = st.local_environment()
    scene.bonds.clear()
    view = View(azimuth=-22, elevation=16, scale=scale)
    render(svg, scene, (cx, cy), view, r_cr, fog=0.2, size=1.0)
    rng = np.random.default_rng(seed)
    for at in scene.atoms:
        a = _screen(view, r_cr, (cx, cy), at.pos)
        ang = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(0.9, 1.6) * scale
        arrow(svg, a, (a[0] + length * np.cos(ang), a[1] + length * np.sin(ang)), color=P.INK, sw=0.22, head=0.7)
