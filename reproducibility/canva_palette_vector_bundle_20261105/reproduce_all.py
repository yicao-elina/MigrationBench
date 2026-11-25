#!/usr/bin/env python3
"""Reproduce all locally packaged Canva P03–P10 palette figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
RENDERERS = (
    PACKAGE / "scripts/render_rmse_palette_20261102.py",
    PACKAGE / "scripts/render_md_six_panel_palette_20261102.py",
    PACKAGE / "scripts/render_neb_palette_20261102.py",
    PACKAGE / "scripts/render_foundation_class_palette_20261102.py",
)


def main() -> None:
    for renderer in RENDERERS:
        print(f"[reproduce] {renderer.relative_to(PACKAGE)}", flush=True)
        subprocess.run([sys.executable, str(renderer)], cwd=PACKAGE, check=True)


if __name__ == "__main__":
    main()
