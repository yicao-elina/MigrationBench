#!/usr/bin/env python3
"""Compose the three rendered 600 K AIMD snapshots into the A07 asset."""

from pathlib import Path
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
frame_dir = ROOT / "renders" / "aimd_frames"
frames = [
    Image.open(frame_dir / "A07_thermal_sampling_frame0000.png").convert("RGBA"),
    Image.open(frame_dir / "A07_thermal_sampling_frame0020.png").convert("RGBA"),
    Image.open(frame_dir / "A07_thermal_sampling_frame0040.png").convert("RGBA"),
]
canvas = Image.new("RGBA", (1800, 620), (0, 0, 0, 0))
for i, frame in enumerate(frames):
    canvas.alpha_composite(frame, (i * 600, 50))
out = ROOT / "assets" / "production" / "A07_thermal_sampling.png"
out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out, format="PNG", optimize=True)
print(out)
