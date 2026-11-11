#!/usr/bin/env python3
"""Bright, publication-oriented Cr-local coordination-sphere renderer.

The input is a real EXTXYZ frame exported by the structure-universe analysis.
Coordinates are translated so the focal Cr is at the origin, with minimum-image
translations used when an input cell/PBC is available. The default output is a
matte ball-and-stick coordination sphere: Cr-to-neighbour cylinders are drawn
only inside an explicit distance cutoff and are labelled as distance-qualified
coordination guides, not as a recovered chemical topology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


ELEMENTS = {
    "Cr": {"rgb": (0.84, 0.39, 0.08), "radius": 0.96, "covalent": 1.39},
    "Sb": {"rgb": (0.08, 0.55, 0.54), "radius": 0.82, "covalent": 1.39},
    "Te": {"rgb": (0.56, 0.22, 0.48), "radius": 0.84, "covalent": 1.38},
}
DEFAULT = {"rgb": (0.48, 0.48, 0.50), "radius": 0.76, "covalent": 1.20}


def args_from_blender() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--focus-index", type=int, default=-1)
    p.add_argument("--cutoff", type=float, default=5.5)
    p.add_argument("--width", type=int, default=2800)
    p.add_argument("--height", type=int, default=2800)
    p.add_argument("--samples", type=int, default=96)
    p.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    p.add_argument("--accent-rgb", default="#7B2CBF")
    p.add_argument("--bond-mode", choices=("coordination", "spheres", "guides"), default="coordination")
    p.add_argument("--bond-cutoff", type=float, default=3.75)
    p.add_argument("--bond-radius", type=float, default=0.105)
    p.add_argument("--show-geometry-guides", action="store_true")
    p.add_argument("--save-blend", action="store_true")
    return p.parse_args(argv)


def header_fields(header: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in shlex.split(header):
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
    return out


def parse_lattice(metadata: dict[str, str]) -> tuple[tuple[float, float, float], ...] | None:
    value = metadata.get("Lattice") or metadata.get("lattice")
    if not value:
        return None
    try:
        numbers = [float(x) for x in value.replace(",", " ").split()]
        if len(numbers) != 9:
            return None
        return tuple(tuple(numbers[3 * i : 3 * i + 3]) for i in range(3))
    except ValueError:
        return None


def parse_pbc(metadata: dict[str, str]) -> tuple[bool, bool, bool]:
    value = metadata.get("pbc", "F F F")
    tokens = value.replace(",", " ").split()
    if len(tokens) == 1:
        tokens = [tokens[0]] * 3
    tokens = (tokens + ["F", "F", "F"])[:3]
    return tuple(token.upper() in {"T", "TRUE", "1"} for token in tokens)  # type: ignore[return-value]


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
                atoms.append({"index": index, "symbol": parts[0], "position": Vector(tuple(float(v) for v in parts[1:4]))})
            metadata = header_fields(header)
            frames.append({"metadata": metadata, "atoms": atoms, "cell": parse_lattice(metadata), "pbc": parse_pbc(metadata)})
    if not frames:
        raise ValueError(f"No frames found in {path}")
    return frames


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def style(symbol: str) -> dict:
    return ELEMENTS.get(symbol, DEFAULT)


def hex_rgb(text: str) -> tuple[float, float, float]:
    value = text.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit RGB hex colour, got {text}")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def make_material(name: str, rgb: tuple[float, float, float], roughness: float, metallic: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.30
    if "Coat Weight" in bsdf.inputs:
        bsdf.inputs["Coat Weight"].default_value = 0.03
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
    # Standard keeps the pure-white camera-ray background genuinely white;
    # AgX would tone-map white to a grey-ish paper field.
    scene.view_settings.view_transform = "Standard"
    try:
        scene.view_settings.look = "None"
    except TypeError:
        pass
    scene.view_settings.exposure = 0.20
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    env = nodes.new("ShaderNodeBackground")
    env.inputs["Color"].default_value = (0.28, 0.30, 0.34, 1.0)
    env.inputs["Strength"].default_value = 0.42
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
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=delta.length, location=(start + end) * 0.5)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = delta.to_track_quat("Z", "Y")
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.use_smooth = True


def add_cluster_ring(pos: Vector, accent: tuple[float, float, float], radius: float) -> None:
    mat = make_material("Cluster_accent", accent, 0.40, 0.0)
    bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=0.055, major_segments=96, minor_segments=16, location=pos)
    obj = bpy.context.object
    obj.name = "cluster_accent_ring"
    obj.data.materials.append(mat)
    for poly in obj.data.polygons:
        poly.use_smooth = True


def choose_focus(frame: dict, args: argparse.Namespace) -> int:
    if args.focus_index >= 0:
        return args.focus_index
    candidates = [a["index"] for a in frame["atoms"] if a["symbol"] == "Cr"]
    if not candidates:
        raise ValueError("No Cr atom is present in the selected frame")
    return candidates[0]


def minimum_image_delta(delta: Vector, cell, pbc: tuple[bool, bool, bool]) -> Vector:
    if cell is None or not any(pbc):
        return delta
    matrix = Matrix(tuple(tuple(row) for row in cell)).transposed()
    if abs(matrix.determinant()) < 1e-10:
        return delta
    fractional = matrix.inverted() @ delta
    for i, active in enumerate(pbc):
        if active:
            fractional[i] -= round(fractional[i])
    return matrix @ fractional


def local_selection(frame: dict, focus_index: int, cutoff: float) -> list[dict]:
    focus = frame["atoms"][focus_index]["position"]
    selected = []
    for atom in frame["atoms"]:
        delta = minimum_image_delta(atom["position"] - focus, frame.get("cell"), frame.get("pbc", (False, False, False)))
        if delta.length <= cutoff:
            item = dict(atom)
            item["position_relative"] = delta
            item["distance_to_focus_A"] = float(delta.length)
            selected.append(item)
    return selected


def add_local_geometry(selected: list[dict], focus_index: int, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    focus_atom = next(a for a in selected if a["index"] == focus_index)
    shift = focus_atom["position"]
    atoms = [{**atom, "position_local": atom.get("position_relative", atom["position"] - shift)} for atom in selected]
    materials = {symbol: make_material(f"Matte_{symbol}", style(symbol)["rgb"], 0.40) for symbol in {a["symbol"] for a in atoms}}
    accent = hex_rgb(args.accent_rgb)
    focal_radius = style("Cr")["radius"] * 1.10
    add_cluster_ring(Vector((0.0, 0.0, 0.0)), accent, focal_radius * 1.35)
    bonds = []
    if args.bond_mode == "coordination":
        for atom in sorted(atoms, key=lambda item: item["position_local"].length):
            if atom["index"] == focus_index or atom["position_local"].length > args.bond_cutoff:
                continue
            direction = atom["position_local"].normalized()
            start = direction * (style("Cr")["radius"] * 0.86)
            end = atom["position_local"] - direction * (style(atom["symbol"])["radius"] * 0.72)
            rgb = style(atom["symbol"])["rgb"]
            bond_rgb = tuple(0.16 + 0.34 * channel for channel in rgb)
            bond_material = make_material(f"Coordination_{atom['symbol']}", bond_rgb, 0.46)
            add_cylinder(start, end, args.bond_radius, bond_material, f"coordination_{focus_index}_{atom['index']}")
            bonds.append({"atoms": [focus_index, atom["index"]], "distance_A": float(atom["position_local"].length), "cutoff_A": args.bond_cutoff, "type": "coordination_distance_qualified"})
    if args.bond_mode == "guides" or args.show_geometry_guides:
        guide_materials = {}
        for i, atom_i in enumerate(atoms):
            for j in range(i + 1, len(atoms)):
                atom_j = atoms[j]
                dist = (atom_j["position_local"] - atom_i["position_local"]).length
                cutoff = 1.16 * (style(atom_i["symbol"])["covalent"] + style(atom_j["symbol"])["covalent"])
                if 0.55 < dist <= cutoff:
                    key = tuple(sorted((atom_i["symbol"], atom_j["symbol"])))
                    guide_materials.setdefault(key, make_material("Guide_" + "_".join(key), (0.62, 0.64, 0.68), 0.52))
                    ui = (atom_j["position_local"] - atom_i["position_local"]).normalized()
                    add_cylinder(atom_i["position_local"] + ui * 0.48, atom_j["position_local"] - ui * 0.48, 0.055, guide_materials[key], f"guide_{atom_i['index']}_{atom_j['index']}")
                    bonds.append({"atoms": [atom_i["index"], atom_j["index"]], "distance_A": dist, "type": "geometry_guide_not_bond"})
    for atom in atoms:
        radius = style(atom["symbol"])["radius"] * (1.10 if atom["index"] == focus_index else 0.86)
        add_sphere(atom["position_local"], radius, materials[atom["symbol"]], f"atom_{atom['index']:03d}_{atom['symbol']}")
    return atoms, bonds


def add_camera_lights(atoms: list[dict]) -> None:
    coords = [a["position_local"] for a in atoms]
    low = Vector(tuple(min(p[k] for p in coords) for k in range(3)))
    high = Vector(tuple(max(p[k] for p in coords) for k in range(3)))
    center = (low + high) * 0.5
    span = high - low
    view = Vector((0.72, -1.0, 0.44)).normalized()
    extent = max(span.x, span.y, span.z)
    distance = extent * 4.0 + 18.0
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center - view * distance
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = extent * 1.65 + 2.0
    bpy.context.scene.camera = camera
    light_specs = [
        ("Key", 2100.0, 12.0, Vector((-0.9, -1.0, 1.3))),
        ("Fill", 850.0, 15.0, Vector((1.0, -0.4, 0.5))),
        ("Rim", 650.0, 12.0, Vector((0.3, 0.9, 0.8))),
    ]
    for name, energy, size, direction in light_specs:
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
        obj.location = center + direction.normalized() * (extent * 2.2 + 6.0)
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
    output_paths = {"opaque_rgb": opaque_dir / "cr_local_environment.png", "transparent_rgba": alpha_dir / "cr_local_environment.png"}
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / "cr_local_environment.blend"))
        output_paths["blend"] = out_dir / "cr_local_environment.blend"
    manifest = {
        "schema_version": "2.0",
        "source": {"trajectory": str(Path(args.input).resolve()), "trajectory_sha256": sha256(Path(args.input).resolve()), "frame_index": frame_index, "frame_metadata": frame["metadata"]},
        "selection": {"focus_atom_id": focus_index, "focus_element": frame["atoms"][focus_index]["symbol"], "cutoff_A": args.cutoff, "criterion": "direct Cartesian distance <= cutoff; rigid translation only", "atom_count": len(atoms)},
        "atoms": [{"id": a["index"], "element": a["symbol"], "role": "focal" if a["index"] == focus_index else "local_context", "pos_source": [round(v, 8) for v in a["position"].to_tuple()], "pos_render": [round(v, 8) for v in a["position_local"].to_tuple()]} for a in atoms],
        "bonds": bonds,
        "scientific_geometry_note": "The source EXTXYZ contains coordinates but no explicit topology. Coordination cylinders are distance-qualified Cr-neighbour guides inside the recorded cutoff; they are not claimed as recovered covalent bond orders.",
        "style": {"background": "pure white opaque + transparent RGBA", "renderer": args.engine, "roughness_atoms": 0.40, "metallic": 0.0, "projection": "orthographic", "depth_of_field": False, "accent_rgb": args.accent_rgb, "bond_mode": args.bond_mode, "coordination_cutoff_A": args.bond_cutoff, "bond_radius_A": args.bond_radius, "show_geometry_guides": bool(args.show_geometry_guides)},
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
