"""Fig. 4 | Local migration: how do three environments change the path response?"""

import numpy as np

from figkit import components as C
from figkit import icons as I
from figkit import layout as L
from figkit import palette as P
from figkit import structures as st
from figkit.svgdoc import Svg, darken

from .common import WIDTH, footnote, save

COLUMNS = ("within_gap", "gap_to_ql", "within_ql")
NOTE = {
    "within_gap": "Cr glides through the vdW gap",
    "gap_to_ql": "Cr at the QL edge; host relaxes (×10)",
    "within_ql": "Cr crosses into the quintuple layer",
}


def build():
    svg = Svg(WIDTH, 150)
    L.panel(svg, 3, 6, "a", "Three local environments")
    col_w = (WIDTH - 12) / 3
    for i, key in enumerate(COLUMNS):
        x = 8 + i * col_w
        cx = x + col_w / 2 - 2
        fill, label = P.TASK[key]
        L.chip(svg, x, 9, col_w - 6, 4.6, fill, label,
               color=P.WHITE if key == "within_ql" else darken(P.TASK_INK[key], 0.15))
        C.vignette(svg, cx, 33, 1.75, key, labels=(i == 0))
        traj = st.paths()[key]["images"][:, st.cr_index(key)]
        disp = np.linalg.norm(traj[-1] - traj[0])
        svg.text(cx, 55, NOTE[key], size="small", anchor="middle", color=darken(P.TASK_INK[key], 0.1))
        svg.text(cx, 58.6, f"Cr displacement {disp:.2f} Å · ‡ DFT peak image" if st.dft_peak_image().get(key)
                 else f"Cr displacement {disp:.2f} Å · downhill in DFT", size="note", anchor="middle", color=P.MUTED)

    for r, (kind, letter) in enumerate((("fixed", "b"), ("self", "c"))):
        y = 64 + r * 41
        color, name = P.OPERATOR[kind]
        L.panel(svg, 3, y + 3, letter, "")
        svg.rect(3, y + 5, 1.2, 33, fill=color)
        svg.text(6, y + 3, name, size="label", weight="bold")
        I.path_operator(svg, 6, y + 7, 14, 8, kind)
        svg.text(13, y + 20, "*E*_{θ}(*q*_{s}^{ref})" if kind == "fixed" else "*E*_{θ}(*q*_{s}^{θ})",
                 size=6.5, anchor="middle")
        for i in range(3):
            x = 26 + i * (WIDTH - 29) / 3
            I.plot_slot(svg, x, y + 1, (WIDTH - 29) / 3 - 3, 37, "reaction coordinate", "energy (eV)",
                        "profiles + seed replicates")
    footnote(svg, "Vignettes use the Cr81 fixed-path NEB images; Cr images light → dark along the path.")
    save(svg, "fig4_local_migration")


if __name__ == "__main__":
    build()
