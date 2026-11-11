#!/usr/bin/env python3
"""Render Figure 1 replacement assets from supplied atomistic coordinates.

This script is executed by Blender 4.0.2.  It intentionally keeps the package's
vector/text layers separate: the output is transparent RGBA artwork only.
Coordinates are read from the supplied EXTXYZ frames; the only non-source
geometry is the documented rigid upper-layer shift used for the collective-
sliding schematic (A06).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
from mathutils import Vector


STYLE_SOURCE = "/Users/alina/.codex/skills/molecular-blender-render/scripts/cases/cr_migration_sb2te3/build_cr_migration_preview.py"

# Frozen scene-linear values from the approved Cr/Sb2Te3 mother scene.
PALETTE = {
    "Te": (0.78, 0.28, 0.55, 1.0),
    "Sb": (0.18, 0.62, 0.68, 1.0),
    "Cr": (0.82, 0.48, 0.08, 1.0),
    "bond": (0.37, 0.42, 0.45, 1.0),
    "ink": (0.12, 0.16, 0.18, 1.0),
    "halo": (0.62, 0.90, 0.70, 1.0),
}
RADII = {"Te": 1.04, "Sb": 1.02, "Cr": 0.88}
HOST = {"Te", "Sb"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def header_fields(header: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in shlex.split(header):
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def read_extxyz(path: Path) -> list[dict]:
    frames: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            count = int(line.strip())
            header = handle.readline().strip()
            fields = header_fields(header)
            lattice = fields.get("Lattice")
            if lattice is None:
                raise ValueError(f"Missing Lattice in {path}")
            cell = [list(map(float, lattice.split()))[i : i + 3] for i in range(0, 9, 3)]
            atoms = []
            for _ in range(count):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Truncated atom row in {path}")
                atoms.append({"symbol": parts[0], "position": Vector(map(float, parts[1:4]))})
            frames.append({"header": header, "fields": fields, "cell": cell, "atoms": atoms})
    if not frames:
        raise ValueError(f"No frames found in {path}")
    n = len(frames[0]["atoms"])
    if any(len(frame["atoms"]) != n for frame in frames):
        raise ValueError(f"Frame atom counts differ in {path}")
    return frames


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def make_material(name: str, color: tuple[float, float, float, float], roughness: float = 0.28) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    assert bsdf is not None
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.34
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.0
    return mat


def make_halo_material() -> bpy.types.Material:
    mat = make_material("MAT_selected_Cr_shell_cue", PALETTE["halo"], roughness=0.48)
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    assert bsdf is not None
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 0.32
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    elif hasattr(mat, "blend_method"):
        mat.blend_method = "BLEND"
    return mat


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def configure_world(scene: bpy.types.Scene) -> None:
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 0.46
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def add_light(name: str, location: Vector, energy: float, size: float, color: tuple[float, float, float]) -> None:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


def add_sphere(name: str, position: Vector, radius: float, material: bpy.types.Material) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, ring_count=32, radius=radius, location=position
    )
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    for poly in obj.data.polygons:
        poly.use_smooth = True


def add_cylinder(name: str, start: Vector, end: Vector, radius: float, material: bpy.types.Material) -> None:
    delta = end - start
    if delta.length < 1e-6:
        return
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=24, radius=radius, depth=delta.length, location=(start + end) * 0.5
    )
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = delta.to_track_quat("Z", "Y")
    obj.data.materials.append(material)
    for poly in obj.data.polygons:
        poly.use_smooth = True


def ql_id(position: Vector) -> int:
    return 0 if position.z < 24.0 else 1


def add_visual_neighbour_rods(atoms: list[dict], materials: dict[str, bpy.types.Material]) -> int:
    """Add muted host-host neighbour cues within each QL only.

    The source EXTXYZ does not contain a bond table.  These rods are therefore
    explicitly visual neighbour cues, never Cr coordination bonds and never
    cross-vdW-gap bonds.
    """
    host_indices = [i for i, atom in enumerate(atoms) if atom["symbol"] in HOST]
    count = 0
    for offset, i in enumerate(host_indices):
        ai = atoms[i]
        for j in host_indices[offset + 1 :]:
            aj = atoms[j]
            if ql_id(ai["position"]) != ql_id(aj["position"]):
                continue
            delta = aj["position"] - ai["position"]
            d = delta.length
            if not 2.55 < d < 3.52:
                continue
            if {ai["symbol"], aj["symbol"]} != {"Sb", "Te"}:
                continue
            midpoint = (ai["position"] + aj["position"]) * 0.5
            unit = delta.normalized()
            p0 = ai["position"] + unit * (RADII[ai["symbol"]] * 0.64)
            p1 = midpoint
            p2 = aj["position"] - unit * (RADII[aj["symbol"]] * 0.64)
            add_cylinder(f"visual_neighbour_{i:03d}_{j:03d}_a", p0, p1, 0.070, materials["bond"])
            add_cylinder(f"visual_neighbour_{i:03d}_{j:03d}_b", p1, p2, 0.070, materials["bond"])
            count += 1
    return count


def add_shell_halo(center: Vector, radius: float, camera: bpy.types.Object) -> None:
    direction = (camera.location - center).normalized()
    location = center + direction * 0.16
    bpy.ops.mesh.primitive_torus_add(
        major_segments=96,
        minor_segments=10,
        location=location,
        major_radius=radius * 1.55,
        minor_radius=0.055,
        rotation=direction.to_track_quat("Z", "Y").to_euler(),
    )
    obj = bpy.context.object
    obj.name = "semantic_local_environment_shell"
    obj.data.materials.append(make_halo_material())


def build_camera(scene: bpy.types.Scene, positions: list[Vector], width: int, height: int, local: bool = False) -> tuple[bpy.types.Object, Vector, float]:
    lo = Vector((min(p.x for p in positions), min(p.y for p in positions), min(p.z for p in positions)))
    hi = Vector((max(p.x for p in positions), max(p.y for p in positions), max(p.z for p in positions)))
    target = (lo + hi) * 0.5
    if local:
        view = target + Vector((4.2, -20.0, 5.7))
        scale = max((hi - lo).length * 1.25, 7.5)
    else:
        # Mother-scene camera direction: low-elevation oblique orthographic,
        # zero roll, no DOF. Only ortho scale is fitted to each source crop.
        view = target + Vector((4.2, -53.0, 5.7))
        span_x = hi.x - lo.x
        span_z = hi.z - lo.z
        # The finite EXTXYZ crops are smaller than the mother-scene patch and
        # have different projected depth.  Keep the mother camera direction
        # but use a conservative 1.80 safety factor so no atom is clipped.
        scale = max(span_z * 1.80, span_x / (width / height) * 1.80)
    data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = scale
    camera.location = view
    look_at(camera, target)
    return camera, target, scale


def configure_render(scene: bpy.types.Scene, width: int, height: int, path: Path) -> None:
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = True
    scene.render.use_file_extension = True
    scene.render.filepath = str(path)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.52
    scene.view_settings.gamma = 1.0


def render_asset(
    asset_id: str,
    frame: dict,
    out_path: Path,
    width: int,
    height: int,
    *,
    local_indices: list[int] | None = None,
    layer_shift: float = 0.0,
    include_cr: bool = True,
    add_halo: bool = False,
) -> dict:
    clear_scene()
    scene = bpy.context.scene
    scene.name = f"Figure1_{asset_id}_coordinate_render"
    configure_world(scene)
    atoms = frame["atoms"]
    if local_indices is None:
        indices = list(range(len(atoms)))
    else:
        indices = list(local_indices)
    display_atoms: list[dict] = []
    for index in indices:
        atom = atoms[index]
        if not include_cr and atom["symbol"] == "Cr":
            continue
        pos = atom["position"].copy()
        if layer_shift and atom["symbol"] in HOST and ql_id(pos) == 1:
            pos.x += layer_shift
        display_atoms.append({"index": index, "symbol": atom["symbol"], "position": pos})
    positions = [a["position"] for a in display_atoms]
    local = local_indices is not None
    camera, target, _ = build_camera(scene, positions, width, height, local=local)
    key = add_light("Key_softbox", target + Vector((-16.0, -24.0, 26.0)), 3000.0, 24.0, (1.0, 0.97, 0.93))
    fill = add_light("Cool_fill", target + Vector((20.0, -8.0, 17.0)), 1600.0, 26.0, (0.90, 0.97, 1.0))
    front = add_light("Front_soft_fill", target + Vector((0.0, -38.0, 11.0)), 2200.0, 32.0, (1.0, 1.0, 1.0))
    rim = add_light("Top_rim", target + Vector((0.0, 10.0, 24.0)), 1100.0, 22.0, (1.0, 1.0, 1.0))
    for light in (key, fill, front, rim):
        look_at(light, target)
    materials = {
        element: make_material(f"MAT_{asset_id}_{element}", PALETTE[element], roughness=0.28)
        for element in ("Te", "Sb", "Cr")
    }
    materials["bond"] = make_material(f"MAT_{asset_id}_visual_neighbour", PALETTE["bond"], roughness=0.42)
    rod_count = add_visual_neighbour_rods(display_atoms, materials) if not local else 0
    for atom in display_atoms:
        radius = RADII[atom["symbol"]]
        add_sphere(f"{asset_id}_atom_{atom['index']:03d}_{atom['symbol']}", atom["position"], radius, materials[atom["symbol"]])
    if add_halo:
        focus = next((a for a in display_atoms if a["symbol"] == "Cr"), None)
        if focus:
            add_shell_halo(focus["position"], RADII["Cr"], camera)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    configure_render(scene, width, height, out_path)
    bpy.ops.render.render(write_still=True)
    return {
        "asset_id": asset_id,
        "output": str(out_path),
        "sha256": sha256(out_path),
        "resolution_px": [width, height],
        "source_atom_count": len(atoms),
        "display_atom_count": len(display_atoms),
        "visual_neighbour_rod_count": rod_count,
        "camera": {"type": "ORTHO", "target_A": list(target), "ortho_scale_A": scene.camera.data.ortho_scale},
        "rigid_upper_ql_shift_A": layer_shift,
        "scientific_note": "Actual coordinates preserved; visual neighbour rods are host-host cues only." if not layer_shift else "RECONSTRUCTION: actual host coordinates plus documented rigid upper-QL display shift; not a sampled trajectory.",
    }


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--aimd-source", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    package = Path(args.package_root).resolve()
    inputs = package / "inputs"
    production = package / "assets" / "production"
    render_dir = package / "renders" / "aimd_frames"
    aimd_path = Path(args.aimd_source).resolve()
    in_gap = read_extxyz(inputs / "within_gap__ideal2_cr_canonical_in_gap.extxyz")
    entry = read_extxyz(inputs / "gap_to_ql__ideal1_cr_canonical_gap_to_ql.extxyz")
    within_ql = read_extxyz(inputs / "within_ql__neb_3_sb2te3.extxyz")
    aimd = read_extxyz(aimd_path)
    records = []
    records.append(render_asset("A01_material_hero", in_gap[0], production / "A01_material_hero.png", 1800, 1400))
    focus_index = next(i for i, atom in enumerate(aimd[0]["atoms"]) if atom["symbol"] == "Cr")
    focus_position = aimd[0]["atoms"][focus_index]["position"]
    local_indices = [focus_index]
    local_indices.extend(
        i for i, atom in enumerate(aimd[0]["atoms"])
        if atom["symbol"] in HOST and (atom["position"] - focus_position).length <= 4.6
    )
    records.append(render_asset("A02_local_environment", aimd[0], production / "A02_local_environment.png", 1200, 1200, local_indices=local_indices, add_halo=True))
    records.append(render_asset("A03_in_gap", in_gap[4], production / "A03_in_gap.png", 1500, 1050))
    records.append(render_asset("A04_gap_to_ql", entry[0], production / "A04_gap_to_ql.png", 1500, 1050))
    records.append(render_asset("A05_within_ql", within_ql[4], production / "A05_within_ql.png", 1500, 1050))
    records.append(render_asset("A06_collective_sliding", in_gap[0], production / "A06_collective_sliding.png", 1500, 1050, layer_shift=1.20))
    aimd_frame_indices = [0, min(20, len(aimd) - 1), min(40, len(aimd) - 1)]
    aimd_records = []
    for frame_index in aimd_frame_indices:
        out = render_dir / f"A07_thermal_sampling_frame{frame_index:04d}.png"
        aimd_records.append(render_asset(f"A07_thermal_sampling_frame{frame_index:04d}", aimd[frame_index], out, 720, 520))
    records.extend(aimd_records)
    manifest = {
        "schema_version": "1.0",
        "status": "PREVIEW_ONLY — explicit user approval required before high-resolution export",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "blender_version": bpy.app.version_string,
        "style_source": STYLE_SOURCE,
        "style_contract": {
            "palette_scene_linear": {key: list(value) for key, value in PALETTE.items()},
            "radii_A": RADII,
            "camera": "low-elevation oblique orthographic, zero roll, no DOF; mother-scene direction",
            "lighting": "four broad area softboxes, white world fill, Standard view, exposure 0.52",
            "sphere_mesh": "64 segments x 32 rings, smooth, non-metallic Principled",
            "background": "transparent RGBA output; black viewer background is not baked into the asset",
        },
        "package_root": str(package),
        "sources": {
            "in_gap_extxyz": {"path": str(inputs / "within_gap__ideal2_cr_canonical_in_gap.extxyz"), "sha256": sha256(inputs / "within_gap__ideal2_cr_canonical_in_gap.extxyz"), "frame_count": len(in_gap)},
            "gap_to_ql_extxyz": {"path": str(inputs / "gap_to_ql__ideal1_cr_canonical_gap_to_ql.extxyz"), "sha256": sha256(inputs / "gap_to_ql__ideal1_cr_canonical_gap_to_ql.extxyz"), "frame_count": len(entry)},
            "within_ql_extxyz": {"path": str(inputs / "within_ql__neb_3_sb2te3.extxyz"), "sha256": sha256(inputs / "within_ql__neb_3_sb2te3.extxyz"), "frame_count": len(within_ql)},
            "target_aimd_600K_grouped_valid": {"path": str(aimd_path), "sha256": sha256(aimd_path), "frame_count": len(aimd), "frames_used": aimd_frame_indices, "source_note": "two-Cr 600 K grouped validation configuration frames; visual asset only"},
        },
        "asset_records": records,
        "boundaries": {
            "A01_A03_A04_A05": "coordinate-based renders from supplied one-Cr EXTXYZ paths",
            "A02": "coordinate-based render from a supplied two-Cr 600 K grouped validation frame; nearest-neighbour shell cue is semantic, not a Cr bond table",
            "A06": "RECONSTRUCTION: supplied host coordinates plus rigid upper-QL display shift of +1.20 A; no sampled sliding trajectory was found in the package/workspace source set",
            "A07": "coordinate-based renders from three supplied 600 K AIMD validation frames; not a new MD run and not an independent replica claim",
            "panel_e": "real 30-point coordinate calculation retained in the package build script",
        },
    }
    out = package / "provenance" / "actual_render_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"assets": len(records), "manifest": str(out)}, indent=2))


if __name__ == "__main__":
    main()
