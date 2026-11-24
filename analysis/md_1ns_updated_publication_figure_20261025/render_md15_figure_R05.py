#!/usr/bin/env python3
"""Render the R05 release with four-group pair-resolved CN bars at D.

Panels A--C, E, F, the 15-model source matrix, and the established palette are
inherited from the R04 renderer.  Panel D aggregates the requested local CN
metrics as mean +/- sample standard deviation within four model classes.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import render_md15_figure_R02 as base

base.RELEASE_TAG = "R05"
base.STEM = base.RESULTS / "fig_md_1ns_updated_A_to_F_15models_R05"
base.D_MODE = "cn_bars"
base.F_MODE = "vacf"


if __name__ == "__main__":
    base.render()
