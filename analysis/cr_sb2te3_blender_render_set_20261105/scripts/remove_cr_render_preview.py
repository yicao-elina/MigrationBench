"""Create an approved-preview candidate by removing only the Cr mesh.

The source .blend is never overwritten. The Cr-colored key light remains
unchanged; only the mesh object named Cr_image_00 is hidden from render.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import bpy


OUT = Path(__file__).resolve().parents[1]
SOURCE_BLEND = OUT / "source" / "original_interlayer_sliding__final_state__window_1800.blend"
PREVIEW = OUT / "preview" / "interlayer_sliding__final_state__no_cr__window_1800.png"
NO_CR_BLEND = OUT / "interlayer_sliding__final_state__no_cr__window_1800.blend"
MANIFEST = OUT / "render_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    scene = bpy.context.scene
    cr_objects = [o for o in bpy.data.objects if o.name == "Cr_image_00"]
    if len(cr_objects) != 1:
        raise RuntimeError(f"Expected exactly one Cr_image_00 mesh, found {len(cr_objects)}")
    cr = cr_objects[0]
    if cr.type != "MESH":
        raise RuntimeError(f"Cr_image_00 is not a mesh: {cr.type}")
    cr.hide_render = True
    cr.hide_viewport = True

    scene.render.resolution_x = 1800
    scene.render.resolution_y = 1800
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(PREVIEW)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True

    bpy.ops.wm.save_as_mainfile(filepath=str(NO_CR_BLEND))
    bpy.ops.render.render(write_still=True)
    manifest = {
        "status": "PREVIEW_ONLY — user approval required before 3600px export",
        "source_blend": str(SOURCE_BLEND),
        "source_blend_sha256": sha256(SOURCE_BLEND),
        "source_scene_preview": "/Users/alina/Downloads/interlayer_sliding__final_state__window_1800.png",
        "removed_render_objects": [cr.name],
        "preserved_lights": [o.name for o in bpy.data.objects if o.type == "LIGHT"],
        "engine": scene.render.engine,
        "resolution_px": [scene.render.resolution_x, scene.render.resolution_y],
        "color_mode": scene.render.image_settings.color_mode,
        "color_depth": scene.render.image_settings.color_depth,
        "output_preview": str(PREVIEW),
        "output_preview_sha256": sha256(PREVIEW),
        "output_blend": str(NO_CR_BLEND),
        "scientific_status": "Geometry-only visual schematic; identical to the source scene except the Cr mesh is removed.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
