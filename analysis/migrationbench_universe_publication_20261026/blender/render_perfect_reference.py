#!/usr/bin/env python3
"""Render the fixed Cr-host coordination reference with the same Blender style."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
RENDERER = ROOT / "blender" / "render_cluster_environment.py"
REFERENCE = Path("/Users/alina/Project/migrationbench-paper-repro/experiments/foundation_model_md/data/raw/sb2te3_cr_81atom.xyz")
OUTPUT = ROOT / "blender" / "cluster_renders" / "PERFECT_REFERENCE"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(BLENDER), "-b", "--python", str(RENDERER), "--",
        "--input", str(REFERENCE),
        "--output-dir", str(OUTPUT),
        "--frame", "0",
        "--width", "2800", "--height", "2800",
        "--samples", "96", "--engine", "cycles",
        "--bond-mode", "coordination",
        "--bond-cutoff", "3.75",
        "--bond-radius", "0.105",
        "--accent-rgb", "#202020",
        "--save-blend",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
