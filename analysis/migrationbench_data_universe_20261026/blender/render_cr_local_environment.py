#!/usr/bin/env python3
"""Render a Cr-centered local environment from a real EXTXYZ frame.

Run with Blender:
  blender -b --python render_cr_local_environment.py -- \
    --input path/to/mlff_neb_images.extxyz --output-dir rendered --frame 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path

import bpy
from mathutils import Vector


ELEMENTS = {
    "Cr": {"rgb": (0.77, 0.32, 0.08), "radius": 0.96, "covalent": 1.39},
    "Sb": {"rgb": (0.09, 0.46, 0.48), "radius": 0.82, "covalent": 1.39},
    "Te": {"rgb": (0.48, 0.16, 0.35), "radius": 0.84, "covalent": 1.38},
}
DEFAULT = {"rgb": (0.48, 0.48, 0.50), "radius": 0.76, "covalent": 1.20}


def args_from_blender() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--frame", type=int, default=6)
    p.add_argument("--focus-index", type=int, default=-1)
    p.add_argument("--cutoff", type=float, default=5.5)
    p.add_argument("--width", type=int, default=3600)
    p.add_argument("--height", type=int, default=3600)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    p.add_argument("--bond-scale", type=float, default=1.16)
    p.add_argument("--bond-radius", type=float, default=0.085)
    p.add_argument("--save-blend", action="store_true")
    return p.parse_args(argv)


def header_fields(header: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in shlex.split(header):
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def read_extxyz(path: Path) -> list[dict]:
    frames = []
    with path.open("r", encoding="utf-8") as handle:
        while True:
            first = handle.readline()
            if not first:
                break
            if not first.strip():
                continue
            n = int(first.strip())
            header = handle.readline().strip()
            atoms = []
            for index in range(n):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError(f"Truncated atom line at frame {len(frames)}")
                atoms.append({"index": index, "symbol": parts[0],
                              "position": Vector(tuple(float(v) for v in parts[1:4]))})
            frames.append({"metadata": header_fields(header), "atoms": atoms})
    if not frames:
        raise ValueError(f"No frames found in {path}")
    n = len(frames[0]["atoms"])
    if any(len(frame["atoms"]) != n for frame in frames):
        raise ValueError("All frames must contain the same number of atoms")
    return frames


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def style(symbol: str) -> dict:
    return ELEMENTS.get(symbol, DEFAULT)


def linear_rgb(rgb: tuple[float, float, float]) -> tuple[float, float, float, float]:
    # Keep the authored palette in the direct Blender color input so the matte
    # atoms remain legible under AgX and the soft three-point light rig.
    return (*rgb, 1.0)


def make_material(name: str, rgb: tuple[float, float, float], roughness: float) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = linear_rgb(rgb)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.28
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.0
    return mat


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def configure_scene(args: argparse.Namespace, transparent: bool) -> None:
    scene = bpy.context.scene
    if args.engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        if hasattr(scene.cycles, "use_adaptive_sampling"):
            scene.cycles.use_adaptive_sampling = True
    else:
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except TypeError:
            scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.film_transparent = transparent
    scene.render.image_settings.compression = 15
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.55
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    env = nodes.new("ShaderNodeBackground")
    env.inputs["Color"].default_value = (0.16, 0.18, 0.20, 1.0)
    env.inputs["Strength"].default_value = 0.32
    white = nodes.new("ShaderNodeBackground")
    white.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    white.inputs["Strength"].default_value = 1.0
    path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(env.outputs["Background"], mix.inputs[1])
    links.new(white.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])


def add_sphere(pos: Vector, radius: float, mat: bpy.types.Material, name: str) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=radius, location=pos)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.use_smooth = True


def add_cylinder(start: Vector, end: Vector, radius: float, mat: bpy.types.Material, name: str) -> None:
    delta = end - start
    if delta.length < 1e-6:
        return
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius,
                                       depth=delta.length, location=(start + end) * 0.5)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = delta.to_track_quat("Z", "Y")
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.use_smooth = True


def mixed_rgb(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(0.5 * (x + y) for x, y in zip(a, b))


def choose_focus(frame: dict, args: argparse.Namespace) -> int:
    if args.focus_index >= 0:
        return args.focus_index
    candidates = [a["index"] for a in frame["atoms"] if a["symbol"] == "Cr"]
    if not candidates:
        raise ValueError("No Cr atom is present in the selected EXTXYZ frame")
    return candidates[0]


def local_selection(frame: dict, focus_index: int, cutoff: float) -> list[dict]:
    focus = frame["atoms"][focus_index]["position"]
    selected = []
    for atom in frame["atoms"]:
        if (atom["position"] - focus).length <= cutoff:
            selected.append(atom)
    return selected


def add_local_geometry(selected: list[dict], focus_index: int, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    focus_atom = next(a for a in selected if a["index"] == focus_index)
    shift = focus_atom["position"]
    atoms = []
    for atom in selected:
        pos = atom["position"] - shift
        atoms.append({**atom, "position_local": pos})
    materials = {symbol: make_material(f"Matte_{symbol}", style(symbol)["rgb"], 0.48)
                 for symbol in {a["symbol"] for a in atoms}}
    bond_materials = {}
    bonds = []
    for i, atom_i in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            atom_j = atoms[j]
            dist = (atom_j["position_local"] - atom_i["position_local"]).length
            cutoff = args.bond_scale * (style(atom_i["symbol"])["covalent"] + style(atom_j["symbol"])["covalent"])
            if 0.55 < dist <= cutoff:
                bonds.append({"atoms": [atom_i["index"], atom_j["index"]], "distance_A": dist,
                              "cutoff_A": cutoff, "type": "geometry_heuristic"})
                key = tuple(sorted((atom_i["symbol"], atom_j["symbol"])))
                if key not in bond_materials:
                    bond_materials[key] = make_material(
                        "Bond_" + "_".join(key), mixed_rgb(style(atom_i["symbol"])["rgb"], style(atom_j["symbol"])["rgb"]), 0.52)
                midpoint = (atom_i["position_local"] + atom_j["position_local"]) * 0.5
                ui = (atom_j["position_local"] - atom_i["position_local"]).normalized()
                ri = style(atom_i["symbol"])["radius"] * (1.10 if atom_i["index"] == focus_index else 0.86)
                rj = style(atom_j["symbol"])["radius"] * (1.10 if atom_j["index"] == focus_index else 0.86)
                add_cylinder(atom_i["position_local"] + ui * (0.55 * ri), midpoint,
                             args.bond_radius, bond_materials[key], f"bond_{atom_i['index']}_{atom_j['index']}_a")
                add_cylinder(atom_j["position_local"] - ui * (0.55 * rj), midpoint,
                             args.bond_radius, bond_materials[key], f"bond_{atom_i['index']}_{atom_j['index']}_b")
    for atom in atoms:
        r = style(atom["symbol"])["radius"] * (1.10 if atom["index"] == focus_index else 0.86)
        add_sphere(atom["position_local"], r, materials[atom["symbol"]],
                   f"atom_{atom['index']:03d}_{atom['symbol']}")
    return atoms, bonds


def add_camera_lights(atoms: list[dict]) -> None:
    coords = [a["position_local"] for a in atoms]
    low = Vector(tuple(min(p[k] for p in coords) for k in range(3)))
    high = Vector(tuple(max(p[k] for p in coords) for k in range(3)))
    center = (low + high) * 0.5
    span = high - low
    view = Vector((0.72, -1.0, 0.44)).normalized()
    distance = max(span) * 4.0 + 18.0
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center - view * distance
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(span.x, span.y, span.z) * 1.75
    bpy.context.scene.camera = camera
    light_specs = [
        ("Key", 1450.0, 10.0, Vector((-0.9, -1.0, 1.3))),
        ("Fill", 520.0, 14.0, Vector((1.0, -0.4, 0.5))),
        ("Rim", 390.0, 11.0, Vector((0.3, 0.9, 0.8))),
    ]
    for name, energy, size, direction in light_specs:
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = center + direction.normalized() * (max(span) * 2.2 + 6.0)
        obj.rotation_euler = (center - obj.location).to_track_quat("-Z", "Y").to_euler()


def render(args: argparse.Namespace, frames: list[dict]) -> dict:
    frame_index = min(max(args.frame, 0), len(frames) - 1)
    frame = frames[frame_index]
    focus_index = choose_focus(frame, args)
    selected = local_selection(frame, focus_index, args.cutoff)
    if len(selected) < 2:
        raise ValueError("Cr local selection contains fewer than two atoms")
    clear_scene()
    configure_scene(args, transparent=False)
    atoms, bonds = add_local_geometry(selected, focus_index, args)
    add_camera_lights(atoms)
    out_dir = Path(args.output_dir).resolve()
    opaque_dir = out_dir / "opaque_rgb"
    alpha_dir = out_dir / "transparent_rgba"
    opaque_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(opaque_dir / "cr_local_environment.png")
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    bpy.ops.render.render(write_still=True)
    configure_scene(args, transparent=True)
    scene.render.filepath = str(alpha_dir / "cr_local_environment.png")
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    bpy.ops.render.render(write_still=True)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "cr_local_environment.blend"))
    output_paths = {
        "opaque_rgb": opaque_dir / "cr_local_environment.png",
        "transparent_rgba": alpha_dir / "cr_local_environment.png",
    }
    if args.save_blend:
        output_paths["blend"] = out_dir / "cr_local_environment.blend"
    manifest = {
        "schema_version": "1.0",
        "source": {"trajectory": str(Path(args.input).resolve()), "trajectory_sha256": sha256(Path(args.input).resolve()),
                    "frame_index": frame_index, "frame_metadata": frame["metadata"]},
        "selection": {"focus_atom_id": focus_index, "focus_element": frame["atoms"][focus_index]["symbol"],
                      "cutoff_A": args.cutoff, "criterion": "direct Cartesian distance <= cutoff; whole coordinates translated by focus position",
                      "atom_count": len(atoms)},
        "atoms": [{"id": a["index"], "element": a["symbol"], "role": "focal" if a["index"] == focus_index else "local_context",
                   "pos_source": [round(v, 8) for v in a["position"].to_tuple()],
                   "pos_render": [round(v, 8) for v in a["position_local"].to_tuple()]} for a in atoms],
        "bonds": bonds,
        "scientific_geometry_note": "No explicit topology is present in the EXTXYZ source. Bonds are geometry-based heuristics and no individual atom was moved; only a rigid global translation centers the Cr atom.",
        "style": {"roughness_atoms": 0.48, "roughness_bonds": 0.52, "metallic": 0.0, "projection": "orthographic", "depth_of_field": False},
        "outputs": {key: str(path) for key, path in output_paths.items()},
        "output_sha256": {key: sha256(path) for key, path in output_paths.items() if path.exists()},
    }
    (out_dir / "cr_local_environment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    args = args_from_blender()
    manifest = render(args, read_extxyz(Path(args.input).resolve()))
    print(json.dumps({"frame": manifest["source"]["frame_index"], "focus": manifest["selection"], "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()

