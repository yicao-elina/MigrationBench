#!/usr/bin/env python3
"""R04 additive Image #1 matte 7.0-in proof wrapper."""

from pathlib import Path
import os
import runpy


HERE = Path(__file__).resolve().parent
os.environ["BLENDER_RENDER_REVISION"] = "R04_matte_body7in"
os.environ["BLENDER_RENDER_STYLE"] = "matte"
os.environ["BLENDER_TRIPTYCH_WIDTH"] = "2100"
os.environ["BLENDER_TRIPTYCH_HEIGHT"] = "875"
os.environ["BLENDER_TARGET_DPI"] = "300"
os.environ["BLENDER_TARGET_BODY_WIDTH_IN"] = "7.0"
os.environ["BLENDER_HEADLESS_COMMAND"] = (
    "/Applications/Blender.app/Contents/MacOS/Blender "
    "--background --python render_scene_R04_matte_body7in.py"
)

runpy.run_path(str(HERE / "render_scene.py"), run_name="__main__")
