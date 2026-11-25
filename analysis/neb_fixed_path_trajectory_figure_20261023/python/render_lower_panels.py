#!/usr/bin/env python3
"""Re-render lower panels from prepared fixed-path data."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_fixed_path_figure import main


if __name__ == "__main__":
    raise SystemExit(main(["--render-only", *sys.argv[1:]]))
