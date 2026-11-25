#!/usr/bin/env python3
"""Render R06S: display-smoothed Cr--Cr RDF replacement."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_md15_figure_R02 as base

base.RELEASE_TAG = "R06S"
base.STEM = base.RESULTS / "fig_md_1ns_updated_A_to_F_15models_R06S"
base.D_MODE = "cn_bars"
base.E_MODE = "smooth_crcr"
base.F_MODE = "vacf"


if __name__ == "__main__":
    base.render()
