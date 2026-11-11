#!/usr/bin/env python3
"""Render the R03 release with the original VACF panel restored at F.

Panels A--E, the 15-model matrix, source data, and palette are inherited
unchanged from the R02 renderer.  Only panel F and the release filenames are
changed.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import render_md15_figure_R02 as base  # noqa: E402


base.RELEASE_TAG = "R03"
base.STEM = base.RESULTS / "fig_md_1ns_updated_A_to_F_15models_R03"
base.F_MODE = "vacf"


if __name__ == "__main__":
    base.render()
