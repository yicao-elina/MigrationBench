"""Fig. 3 | MD: how long does each model run stably?"""

from figkit import components as C
from figkit import icons as I
from figkit import layout as L
from figkit import palette as P
from figkit.svgdoc import Svg

from .common import WIDTH, footnote, save, slot_row

TICKS = ("0", "0.25", "0.5", "1 ns")


def _timeline(svg: Svg, x0, x1, y, fail_at=None):
    end = fail_at if fail_at is not None else x1
    svg.line(x0, y, end, y, stroke=P.LINE, sw=0.35)
    if fail_at is not None:
        svg.line(fail_at, y, x1, y, stroke=P.FAINT, sw=0.35, dash="0.8 0.6")
        I.cross(svg, fail_at, y, 1.3)
        svg.text(fail_at + 2.2, y - 1.8, "failure step", size="note", color=P.STATUS["failed"])
    else:
        I.check(svg, x1, y, 1.3)
        svg.text(x1 - 2.0, y - 1.8, "completes 1 ns", size="note", anchor="end", color=P.STATUS["complete"])


def panel_schematic(svg: Svg):
    L.panel(svg, 3, 6, "a", "Same protocol, different fates")
    xs = [32, 58, 84, 110]
    rows = (("Stable rollout", 22, [0.08, 0.1, 0.1, 0.11], None),
            ("Unstable rollout", 48, [0.08, 0.14, 0.22, None], 98))
    for label, y, amps, fail in rows:
        svg.text(4, y + 1, label, size="small", weight="bold",
                 color=P.STATUS["complete"] if fail is None else P.STATUS["failed"])
        for k, (x, amp) in enumerate(zip(xs, amps)):
            if amp is None:
                C.md_frame(svg, x, y, 0.82, 0.3, seed=40 + k, failed=True)
            else:
                C.md_frame(svg, x, y, 0.82, amp, seed=30 + k)
        _timeline(svg, xs[0] - 5, xs[-1] + 7, y + 10.5, fail_at=fail)
    for x, t in zip(xs, TICKS):
        svg.line(x, 69.5, x, 70.5, stroke=P.MUTED, sw=0.2)
        svg.text(x, 73, t, size="note", anchor="middle", color=P.MUTED)
    svg.text(71, 76.5, "simulated time", size="note", anchor="middle", color=P.MUTED)
    L.container(svg, 128, 9, 52, 64, rx=2)
    svg.text(131, 14, "Tracked per frame", size="small", weight="bold")
    C.local_env(svg, 154, 38, 3.2)
    svg.text(131, 62, "*T*, *P*, MSD, Cr coordination *n*_{Cr}", size="small")
    svg.text(131, 66, "RDF and VACF in the SI", size="note", color=P.MUTED)


def build():
    svg = Svg(WIDTH, 130)
    panel_schematic(svg)
    slot_row(svg, 84, 40, [
        ("b", "Run length", "model", "time (ps)", "completed time"),
        ("c", "Temperature", "time (ps)", "T (K)", "mean ± sd"),
        ("d", "Pressure", "time (ps)", "P (GPa)", "mean ± sd"),
        ("e", "MSD", "time (ps)", "MSD (Å²)", "mean ± sd"),
        ("f", "Coordination", "time (ps)", "*n*_{Cr}", "local environment"),
    ])
    footnote(svg, "MD snapshots in a are schematic (thermal offsets on the real lattice); failure position is illustrative.")
    save(svg, "fig3_md")


if __name__ == "__main__":
    build()
