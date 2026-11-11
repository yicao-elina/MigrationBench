#!/usr/bin/env python3
"""Render the R04 release with a short-range Cr--Cr CN panel at D.

Panels A--C, E, F, the 15-model matrix, source data, and palette are inherited
unchanged from the R03 release.  Only panel D and release filenames change.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_md15_figure_R02 as base

base.RELEASE_TAG = "R04"
base.STEM = base.RESULTS / "fig_md_1ns_updated_A_to_F_15models_R04"
base.D_MODE = "cn_short"
base.F_MODE = "vacf"


if __name__ == "__main__":
    base.render()
