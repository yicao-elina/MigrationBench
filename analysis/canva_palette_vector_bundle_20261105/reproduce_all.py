#!/usr/bin/env python3
"""Reproduce the Canva P03-P10 palette bundle from the local source campaigns."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
RENDERERS = (
    SOURCE / "render_rmse_palette_20261102.py",
    SOURCE / "render_md_six_panel_palette_20261102.py",
    SOURCE / "render_neb_palette_20261102.py",
    SOURCE / "render_foundation_class_palette_20261102.py",
)


def main() -> None:
    for renderer in RENDERERS:
        subprocess.run([sys.executable, str(renderer)], cwd=ROOT, check=True)

    release = ROOT / "figures"
    generated = SOURCE / "figures"
    release.mkdir(parents=True, exist_ok=True)
    for path in generated.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".pdf", ".svg", ".png"}:
            relative = path.relative_to(generated)
            target = release / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


if __name__ == "__main__":
    main()
