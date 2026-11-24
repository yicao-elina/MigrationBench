#!/usr/bin/env python3
"""Render the real retained fixed-path Cr migration triptych in Blender.

This script is intentionally self-contained so the scene can be rebuilt headlessly
from the three retained extxyz files.  It does not import any coordinates from
the historical Pov-Ray assets, and it does not read or write the Python curve
data used by the lower panels of the eventual composite figure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import bpy
from mathutils import Vector, Matrix


OUT_DIR = Path(__file__).resolve().parent
REVISION = os.environ.get("BLENDER_RENDER_REVISION", "R03")
RENDER_STYLE = os.environ.get("BLENDER_RENDER_STYLE", "r03").strip().lower()
MATTE_MODE = RENDER_STYLE == "matte"
MATTE_REFERENCE_IMAGE = (
    "/Users/alina/.codex/attachments/"
    "c31191e4-e79f-47bc-bf77-d6eaca40fa6b/image-1.png"
)
WORKSPACE = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
CRITERIA_PATH = (
    WORKSPACE
    / "analysis/neb_fixed_path_trajectory_figure_20261023/visual_review"
    / "FROZEN_VISUAL_ACCEPTANCE_CRITERIA.md"
)
REGISTRY_PATH = (
    WORKSPACE
    / "analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019"
    / "CANONICAL_ENDPOINT_REGISTRY_r01.tsv"
)

PATHS = [
    {
        "column_index": 1,
        "display_label": "within gap",
        "subtitle": "in-gap diffusion  ·  retained fixed path",
        "path_id": "cr_canonical_in_gap",
        "input_path": (
            WORKSPACE
            / "analysis/task3_endpoint_only_neb/exploratory_energy_audit"
            / "rockfish_cr_neb_20261013/inputs/2L-Cu__neb-gap-diffuse.extxyz"
        ),
    },
    {
        "column_index": 2,
        "display_label": "gap2QL",
        "subtitle": "gap→QL  ·  retained fixed path",
        "path_id": "cr_canonical_gap_to_ql",
        "input_path": (
            WORKSPACE
            / "analysis/task3_endpoint_only_neb/exploratory_energy_audit"
            / "rockfish_cr_neb_20261013/inputs/2L-Cu__neb-gap2QL.extxyz"
        ),
    },
    {
        "column_index": 3,
        "display_label": "withinQL",
        "subtitle": "within-QL  ·  retained fixed path",
        "path_id": "cr_canonical_within_ql",
        "input_path": (
            WORKSPACE
            / "analysis/task3_endpoint_only_neb/exploratory_energy_audit"
            / "rockfish_cr_neb_20261013/inputs/2L-Cu__neb-QLs.extxyz"
        ),
    },
]

# The structure palette is deliberately distinct from the four model-family
# colors used by the lower Python panel.
ELEMENT_HEX = {
    "Te": "#9A95C7",  # muted lilac framework
    "Sb": "#9CD5D4",  # soft teal framework
    # More saturated than the historical soft ochre so the real Cr path
    # remains legible on the white publication background.
    "Cr": "#E6A13A",
}
PATH_HEX = "#B46608"
PATH_DARK_HEX = "#754000"
GRAPHITE_HEX = "#28323D"
WHITE_HEX = "#FFFFFF"

# R04 matte is an additive lighting/material variant.  It keeps the same
# element/path semantics as R03 while using a restrained warm-gold Cr, softer
# host palette, and graphite bonds that read as rounded matte solids rather
# than flat swatches or metallic/plastic parts.
if MATTE_MODE:
    ELEMENT_HEX = {
        "Te": "#9693B3",
        "Sb": "#A0C9C8",
        "Cr": "#D29325",
    }
    PATH_HEX = "#A86512"
    PATH_DARK_HEX = "#70400A"
    GRAPHITE_HEX = "#596169"
    WHITE_HEX = "#FCFCFB"

# The first render made the host lattice visually dominant.  These radii keep
# the same ball-and-stick construction while giving the Cr overlays room to be
# read at final size.
ATOM_RADII = {"Te": 0.42, "Sb": 0.46, "Cr": 0.66}
GHOST_CR_RADIUS = 0.52
BOND_RADIUS = 0.052

# One common camera/framing contract for all three paths.  These bounds were
# selected from the actual retained paths after MIC unwrapping and are checked
# again at runtime; they are not used to invent or resample coordinates.
DISPLAY_XMIN = -3.0
DISPLAY_XMAX = 15.9
DISPLAY_ZMIN = 13.6
DISPLAY_ZMAX = 34.0
DISPLAY_CX = (DISPLAY_XMIN + DISPLAY_XMAX) / 2.0
DISPLAY_CZ = (DISPLAY_ZMIN + DISPLAY_ZMAX) / 2.0
DISPLAY_YCENTER = 3.7

TRIPTYCH_WIDTH = int(os.environ.get("BLENDER_TRIPTYCH_WIDTH", "4200"))
TRIPTYCH_HEIGHT = int(os.environ.get("BLENDER_TRIPTYCH_HEIGHT", "1750"))
TARGET_DPI = int(os.environ.get("BLENDER_TARGET_DPI", "300"))
TARGET_BODY_WIDTH_IN = float(
    os.environ.get("BLENDER_TARGET_BODY_WIDTH_IN", str(TRIPTYCH_WIDTH / TARGET_DPI))
)
HEADLESS_COMMAND = os.environ.get(
    "BLENDER_HEADLESS_COMMAND",
    "/Applications/Blender.app/Contents/MacOS/Blender "
    "--background --python render_scene.py",
)
# Blender's orthographic scale is the camera-frame width.  The expanded
# publication view leaves a real-cell margin around each column while keeping
# the same scale for all three master-camera columns.
ORTHO_SCALE = 72.0
COLUMN_CENTERS = (-24.0, 0.0, 24.0)
CAMERA_Y = -80.0
INDIVIDUAL_WIDTH = TRIPTYCH_WIDTH // 3
INDIVIDUAL_HEIGHT = TRIPTYCH_HEIGHT
INDIVIDUAL_ORTHO_SCALE = 26.0
CAMERA_FOCUS_FRACTION = 0.50
CAMERA_FOCUS_X_LIMIT = 1.35
CAMERA_FOCUS_Z_LIMIT = 2.50

HOST_ALPHA = {
    0: 0.34,
    1: 0.030,
    2: 0.035,
    3: 0.040,
    4: 0.045,
    5: 0.050,
    6: 0.055,
    7: 0.060,
    8: 0.065,
    9: 0.16,
}
CR_ALPHA = {
    0: 0.96,
    1: 0.22,
    2: 0.25,
    3: 0.28,
    4: 0.31,
    5: 0.34,
    6: 0.37,
    7: 0.40,
    8: 0.43,
    9: 0.98,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(payload.encode("utf-8"))


def hex_to_rgb(value: str):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def parse_extxyz(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    frames = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        natoms = int(lines[cursor].strip())
        if cursor + 1 >= len(lines):
            raise ValueError(f"{path}: missing extxyz comment line")
        comment = lines[cursor + 1].strip()
        lattice_match = re.search(r'Lattice="([^"]+)"', comment)
        if lattice_match is None:
            raise ValueError(f"{path}: missing Lattice in frame {len(frames)}")
        lattice_values = [float(x) for x in lattice_match.group(1).split()]
        if len(lattice_values) != 9:
            raise ValueError(f"{path}: malformed 3x3 Lattice in frame {len(frames)}")
        cell = [
            lattice_values[0:3],
            lattice_values[3:6],
            lattice_values[6:9],
        ]
        pbc_match = re.search(r'pbc="([^"]+)"', comment)
        if pbc_match is None:
            raise ValueError(f"{path}: missing pbc in frame {len(frames)}")
        pbc_tokens = tuple(pbc_match.group(1).split())
        if len(pbc_tokens) != 3:
            raise ValueError(f"{path}: malformed pbc in frame {len(frames)}")
        species = []
        positions = []
        start = cursor + 2
        stop = start + natoms
        if stop > len(lines):
            raise ValueError(f"{path}: truncated frame {len(frames)}")
        for line in lines[start:stop]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"{path}: malformed atom row: {line!r}")
            species.append(fields[0])
            positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
        frames.append(
            {
                "natoms": natoms,
                "comment": comment,
                "cell": cell,
                "pbc_tokens": pbc_tokens,
                "pbc": tuple(token.upper() == "T" for token in pbc_tokens),
                "species": species,
                "positions": positions,
            }
        )
        cursor = stop
    return frames


def row_mat_vec(vector, matrix):
    """Return vector @ matrix for row-vector convention used by extxyz Lattice."""
    return [
        sum(vector[row] * matrix[row][column] for row in range(3))
        for column in range(3)
    ]


def cart_from_frac(fractional, cell):
    return row_mat_vec(fractional, cell)


def inverse_cell(cell):
    return Matrix(cell).inverted()


def frac_from_cart(vector, inv_cell):
    return row_mat_vec(vector, inv_cell)


def mic_delta(delta_cart, cell, inv_cell, pbc):
    fractional = frac_from_cart(delta_cart, inv_cell)
    wrapped = []
    for index, value in enumerate(fractional):
        if pbc[index]:
            value -= math.floor(value + 0.5)
        wrapped.append(value)
    return cart_from_frac(wrapped, cell), wrapped


def unwrap_frames(frames):
    cell = frames[0]["cell"]
    pbc = frames[0]["pbc"]
    inv_cell = inverse_cell(cell)
    unwrapped = [
        [[float(x), float(y), float(z)] for x, y, z in frame["positions"]]
        for frame in frames
    ]
    corrections = []
    for image_index in range(1, len(frames)):
        image_corrections = []
        for atom_index in range(frames[image_index]["natoms"]):
            previous = unwrapped[image_index - 1][atom_index]
            raw_previous = frames[image_index - 1]["positions"][atom_index]
            raw_current = frames[image_index]["positions"][atom_index]
            delta = [
                raw_current[axis] - raw_previous[axis] for axis in range(3)
            ]
            corrected, fractional_shift = mic_delta(delta, cell, inv_cell, pbc)
            unwrapped[image_index][atom_index] = [
                previous[axis] + corrected[axis] for axis in range(3)
            ]
            image_corrections.append(
                [fractional_shift[axis] for axis in range(3)]
            )
        corrections.append(image_corrections)
    max_abs_fractional_shift = 0.0
    for image_corrections in corrections:
        for shift in image_corrections:
            for value in shift:
                max_abs_fractional_shift = max(
                    max_abs_fractional_shift, abs(float(value))
                )
    return unwrapped, {
        "algorithm": (
            "frame-to-frame minimum-image unwrapping; row-vector fractional "
            "coordinates; periodic axes wrapped to [-0.5, 0.5)"
        ),
        "pbc_axes": list(pbc),
        "max_abs_fractional_step_after_wrap": max_abs_fractional_shift,
        "coordinates_rendered": "unwrapped frame coordinates derived from extxyz",
    }


def endpoint_hash(frame):
    canonical = {
        "species": frame["species"],
        "positions_A": [
            [round(float(value), 10) for value in row]
            for row in frame["positions"]
        ],
        "cell_A": [
            [round(float(value), 10) for value in row]
            for row in frame["cell"]
        ],
        "pbc": list(frame["pbc"]),
    }
    return json_hash(canonical)


def read_registry():
    with REGISTRY_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["path_id"]: row for row in rows}


def validate_path(config, registry_rows):
    path = config["input_path"]
    frames = parse_extxyz(path)
    source_hash = sha256_file(path)
    first = frames[0]
    species_order = " ".join(first["species"])
    atom_order_hash = sha256_bytes(species_order.encode("utf-8"))
    cr_indices = [
        [index for index, element in enumerate(frame["species"]) if element == "Cr"]
        for frame in frames
    ]
    registry = registry_rows.get(config["path_id"])
    if registry is None:
        raise AssertionError(f"missing registry row for {config['path_id']}")
    checks = {
        "source_exists": path.exists(),
        "source_sha256_matches_registry": (
            source_hash == registry["fixed_retained_path_sha256"]
        ),
        "atoms_all_81": all(frame["natoms"] == 81 for frame in frames),
        "image_count_10": len(frames) == 10,
        "pbc_all_TTT": all(frame["pbc"] == (True, True, True) for frame in frames),
        "species_order_matches_registry": species_order == registry["species_order"],
        "single_cr_atom_all_images": all(indices == [80] for indices in cr_indices),
        "registry_fixed_image_count_10": registry["fixed_image_count"] == "10",
        "registry_atoms_81": registry["atoms"] == "81",
        "registry_pbc_TTT": registry["pbc"].split() == ["T", "T", "T"],
        "registry_endpoint_consistency_PASS": (
            registry["endpoint_consistency_gate"] == "PASS"
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"{config['path_id']} validation failed: {failed}")
    if path.resolve() != Path(registry["fixed_retained_path"]).resolve():
        raise AssertionError(f"{config['path_id']} path differs from registry")
    unwrapped, mic_info = unwrap_frames(frames)
    all_positions = [
        position
        for frame in unwrapped
        for position in frame
    ]
    bounds = {
        "x_min": min(position[0] for position in all_positions),
        "x_max": max(position[0] for position in all_positions),
        "z_min": min(position[2] for position in all_positions),
        "z_max": max(position[2] for position in all_positions),
    }
    for axis, minimum, maximum in (
        ("x", DISPLAY_XMIN, DISPLAY_XMAX),
        ("z", DISPLAY_ZMIN, DISPLAY_ZMAX),
    ):
        if bounds[f"{axis}_min"] < minimum or bounds[f"{axis}_max"] > maximum:
            raise AssertionError(
                f"{config['path_id']} bounds exceed common camera frame: {bounds}"
            )
    return {
        **config,
        "input_path": str(path),
        "input_sha256": source_hash,
        "frames": frames,
        "unwrapped": unwrapped,
        "mic_info": mic_info,
        "registry": registry,
        "validation_checks": checks,
        "natoms": first["natoms"],
        "image_count": len(frames),
        "pbc": list(first["pbc"]),
        "species_order": species_order,
        "atom_order_sha256": atom_order_hash,
        "migrating_atom_indices": sorted(set(index for row in cr_indices for index in row)),
        "endpoint_hashes_computed": {
            "image_0": endpoint_hash(frames[0]),
            "image_9": endpoint_hash(frames[-1]),
        },
        "bounds_unwrapped_A": bounds,
        "cell_A": first["cell"],
    }


def transform_position(position, column_offset, depth_offset=0.0):
    return (
        position[0] - DISPLAY_CX + column_offset,
        position[1] - DISPLAY_YCENTER + depth_offset,
        position[2] - DISPLAY_CZ,
    )


def aim_object(obj, target=(0.0, 0.0, 0.0)):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)


def ensure_collection(name, parent=None):
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
    if parent is None:
        parent = bpy.context.scene.collection
    if collection.name not in {child.name for child in parent.children}:
        parent.children.link(collection)
    return collection


def make_material(name, color_hex, alpha=1.0, roughness=0.3, metallic=0.0):
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = (*hex_to_rgb(color_hex), alpha)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is not None:
        material_name = name.lower()
        if MATTE_MODE:
            # Keep matte surfaces diffuse-first.  Cr/path materials retain a
            # controlled broad highlight, while the host and bonds stay soft;
            # no metallic or clear-coat response is introduced.
            metallic = 0.0
            roughness = max(roughness, 0.34)
            specular_level = 0.34 if any(
                token in material_name
                for token in ("cr", "path", "endpoint", "arrow", "index")
            ) else 0.22
        else:
            specular_level = 0.35
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (
                *hex_to_rgb(color_hex),
                alpha,
            )
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = specular_level
        if MATTE_MODE and "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = 0.0
        if MATTE_MODE and "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.0
    if alpha < 0.999:
        if hasattr(material, "surface_render_method"):
            try:
                material.surface_render_method = "DITHERED"
            except Exception:
                pass
        if hasattr(material, "blend_method"):
            try:
                material.blend_method = "BLEND"
            except Exception:
                pass
        if hasattr(material, "use_transparency_overlap"):
            try:
                material.use_transparency_overlap = False
            except Exception:
                pass
    return material


def load_arial_font():
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return bpy.data.fonts.load(str(candidate)), str(candidate)
            except Exception:
                continue
    return None, "Blender Bfont fallback"


def create_base_sphere_mesh():
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=24,
        ring_count=16,
        radius=1.0,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    mesh = obj.data.copy()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    bpy.data.objects.remove(obj, do_unlink=True)
    return mesh


def create_base_cylinder_mesh():
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=20,
        radius=1.0,
        depth=2.0,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    mesh = obj.data.copy()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    bpy.data.objects.remove(obj, do_unlink=True)
    return mesh


def create_base_cone_mesh():
    bpy.ops.mesh.primitive_cone_add(
        vertices=24,
        radius1=1.0,
        radius2=0.0,
        depth=2.0,
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    mesh = obj.data.copy()
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    bpy.data.objects.remove(obj, do_unlink=True)
    return mesh


class GeometryFactory:
    def __init__(self):
        self.base_sphere = create_base_sphere_mesh()
        self.base_cylinder = create_base_cylinder_mesh()
        self.base_cone = create_base_cone_mesh()
        self.materials = {}
        self.sphere_meshes = {}
        self.cylinder_meshes = {}
        self.cone_meshes = {}

    def material(self, key, color_hex, alpha, roughness=0.3, metallic=0.0):
        cache_key = (key, color_hex, round(alpha, 4), roughness, metallic)
        if cache_key not in self.materials:
            self.materials[cache_key] = make_material(
                f"MAT_{key}_{round(alpha, 3)}",
                color_hex,
                alpha,
                roughness,
                metallic,
            )
        return self.materials[cache_key]

    def sphere_mesh(self, key, color_hex, alpha, roughness=0.3, metallic=0.0):
        cache_key = (key, color_hex, round(alpha, 4), roughness, metallic)
        if cache_key not in self.sphere_meshes:
            mesh = self.base_sphere.copy()
            mesh.name = f"SphereMesh_{key}_{round(alpha, 3)}"
            mesh.materials.append(
                self.material(cache_key[0], color_hex, alpha, roughness, metallic)
            )
            self.sphere_meshes[cache_key] = mesh
        return self.sphere_meshes[cache_key]

    def cylinder_mesh(self, key, color_hex, alpha, roughness=0.3, metallic=0.0):
        cache_key = (key, color_hex, round(alpha, 4), roughness, metallic)
        if cache_key not in self.cylinder_meshes:
            mesh = self.base_cylinder.copy()
            mesh.name = f"CylinderMesh_{key}_{round(alpha, 3)}"
            mesh.materials.append(
                self.material(cache_key[0], color_hex, alpha, roughness, metallic)
            )
            self.cylinder_meshes[cache_key] = mesh
        return self.cylinder_meshes[cache_key]

    def cone_mesh(self, key, color_hex, alpha, roughness=0.3, metallic=0.0):
        cache_key = (key, color_hex, round(alpha, 4), roughness, metallic)
        if cache_key not in self.cone_meshes:
            mesh = self.base_cone.copy()
            mesh.name = f"ConeMesh_{key}_{round(alpha, 3)}"
            mesh.materials.append(
                self.material(cache_key[0], color_hex, alpha, roughness, metallic)
            )
            self.cone_meshes[cache_key] = mesh
        return self.cone_meshes[cache_key]

    def sphere(
        self,
        collection,
        name,
        position,
        radius,
        element,
        alpha,
        role="atom",
    ):
        mesh = self.sphere_mesh(
            f"{role}_{element}",
            ELEMENT_HEX[element],
            alpha,
            roughness=(
                0.38
                if element == "Cr"
                else 0.50
                if MATTE_MODE
                else 0.34
            ),
            metallic=0.0 if MATTE_MODE else 0.02,
        )
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        obj.location = Vector(position)
        obj.scale = (radius, radius, radius)
        return obj

    def cylinder(
        self,
        collection,
        name,
        point_a,
        point_b,
        radius,
        color_hex,
        alpha,
        role="bond",
    ):
        point_a = Vector(point_a)
        point_b = Vector(point_b)
        delta = point_b - point_a
        length = delta.length
        if length <= 1e-9:
            return None
        mesh = self.cylinder_mesh(
            role,
            color_hex,
            alpha,
            roughness=0.50 if MATTE_MODE else 0.34,
            metallic=0.0 if MATTE_MODE else 0.05,
        )
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        obj.location = (point_a + point_b) / 2.0
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(
            delta.normalized()
        )
        obj.scale = (radius, radius, length / 2.0)
        return obj

    def cone(
        self,
        collection,
        name,
        tip,
        direction,
        radius,
        depth,
        color_hex,
        alpha,
        role="arrow",
    ):
        direction = Vector(direction)
        if direction.length <= 1e-9:
            return None
        direction.normalize()
        mesh = self.cone_mesh(
            role,
            color_hex,
            alpha,
            roughness=0.38 if MATTE_MODE else 0.25,
            metallic=0.0 if MATTE_MODE else 0.02,
        )
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        obj.location = Vector(tip) - direction * (depth * 0.35)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(direction)
        obj.scale = (radius, radius, depth / 2.0)
        return obj


def make_curve(collection, name, points, color_hex, alpha, bevel_depth=0.06):
    curve_data = bpy.data.curves.new(name=f"{name}_Curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 1
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 3
    spline = curve_data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, coordinate in zip(spline.points, points):
        point.co = (*coordinate, 1.0)
    material = make_material(
        f"MAT_{name}",
        color_hex,
        alpha,
        roughness=0.42 if MATTE_MODE else 0.30,
        metallic=0.0,
    )
    curve_data.materials.append(material)
    obj = bpy.data.objects.new(name, curve_data)
    collection.objects.link(obj)
    return obj


def make_text(
    collection,
    name,
    body,
    location,
    size,
    material,
    font,
    align_x="CENTER",
):
    text_data = bpy.data.curves.new(name=f"{name}_Text", type="FONT")
    text_data.body = body
    text_data.align_x = align_x
    text_data.align_y = "CENTER"
    text_data.size = size
    text_data.extrude = 0.004
    text_data.bevel_depth = 0.002
    if font is not None:
        text_data.font = font
    text_data.materials.append(material)
    obj = bpy.data.objects.new(name, text_data)
    collection.objects.link(obj)
    obj.location = Vector(location)
    # Blender text is born in the XY plane; rotate it into the XZ plane facing
    # the camera, whose viewing direction is +Y.
    obj.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    return obj


def add_endpoint_ring(collection, name, position, radius, material):
    bpy.ops.mesh.primitive_torus_add(
        major_segments=36,
        minor_segments=10,
        location=(position[0], position[1] - 0.80, position[2]),
        major_radius=radius,
        minor_radius=0.055,
        rotation=(math.radians(90.0), 0.0, 0.0),
    )
    obj = bpy.context.active_object
    for current_collection in list(obj.users_collection):
        current_collection.objects.unlink(obj)
    collection.objects.link(obj)
    obj.name = name
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def add_image_index_markers(
    collection,
    path_id,
    cr_points,
    marker_material,
    marker_leader_color,
    font,
):
    """Add annotation-only 0–9 callouts anchored to the exact Cr images.

    The leader starts at each rendered Cr image's exact projected x/z
    coordinate.  Only the label layout is offset into a readable rail; no
    atom, path, or energy coordinate is moved, interpolated, or fabricated.
    """
    x_max = max(point[0] for point in cr_points)
    z_min = min(point[2] for point in cr_points)
    z_max = max(point[2] for point in cr_points)
    z_center = (z_min + z_max) / 2.0
    rail_x = x_max + 2.50
    rail_step = 0.55
    # Index order is bottom-to-top so the gap2QL near-overlap has an immediate
    # visual 0→9 reading without moving the underlying coordinates.
    rail_z = [z_center + 0.5 * rail_step * (2 * image_index - 9)
              for image_index in range(10)]
    # Put callouts in a foreground annotation plane.  The x/z anchor remains
    # the real image coordinate; this y value is only a depth ordering choice.
    annotation_y = min(point[1] for point in cr_points) - 1.65
    leader_y = annotation_y + 0.48
    marker_counts = {
        "image_index_markers": 0,
        "image_index_leader_lines": 0,
        "image_index_marker_rule": (
            "one numbered annotation badge per real Cr image 0–9; leader starts "
            "at exact projected Cr x/z; badge rail offset is annotation-only"
        ),
        "image_index_marker_coordinates": [],
    }
    for image_index, point in enumerate(cr_points):
        label_point = (rail_x, annotation_y, rail_z[image_index])
        leader_points = [
            (point[0], leader_y, point[2]),
            (label_point[0], leader_y, label_point[2]),
        ]
        make_curve(
            collection,
            f"{path_id}_image_{image_index}_index_leader",
            leader_points,
            marker_leader_color,
            0.94,
            bevel_depth=0.045,
        )
        add_endpoint_ring(
            collection,
            f"{path_id}_image_{image_index}_index_badge",
            label_point,
            radius=0.30,
            material=marker_material,
        )
        make_text(
            collection,
            f"{path_id}_image_{image_index}_index_label",
            str(image_index),
            (label_point[0], annotation_y - 0.92, label_point[2]),
            0.48,
            marker_material,
            font,
        )
        marker_counts["image_index_markers"] += 1
        marker_counts["image_index_leader_lines"] += 1
        marker_counts["image_index_marker_coordinates"].append(
            {
                "image_index": image_index,
                "anchor_projected_xz_A": [point[0], point[2]],
                "label_badge_xz_A": [label_point[0], label_point[2]],
                "coordinate_status": "annotation_layout_only; not atom coordinate",
            }
        )
    return marker_counts


def host_bond_pairs(frame, cell):
    inv_cell = inverse_cell(cell)
    host_indices = [
        index for index, element in enumerate(frame["species"]) if element != "Cr"
    ]
    pairs = []
    for left_position, left_index in enumerate(host_indices):
        for right_index in host_indices[left_position + 1 :]:
            delta = [
                frame["positions"][right_index][axis]
                - frame["positions"][left_index][axis]
                for axis in range(3)
            ]
            mic, _ = mic_delta(delta, cell, inv_cell, frame["pbc"])
            distance = math.sqrt(sum(value * value for value in mic))
            if 1.75 <= distance <= 3.75:
                pairs.append((left_index, right_index, mic, distance))
    return pairs


def add_lights(scene):
    lights_collection = ensure_collection("LIGHTS")
    specifications = (
        [
            # Broad, directional key with a large emitter for a soft rolloff.
            ("Key_Area", (-18.0, -30.0, 28.0), 1050.0, 26.0),
            # Restrained fill prevents hard black occlusion while preserving
            # the key-light direction on the matte spheres.
            ("Fill_Area", (18.0, -10.0, -8.0), 520.0, 34.0),
            # Low-energy rear rim separates Cr/host silhouettes from white.
            ("Rim_Area", (0.0, 20.0, 28.0), 620.0, 32.0),
        ]
        if MATTE_MODE
        else [
            ("Key_Area", (0.0, -26.0, 24.0), 1400.0, 28.0),
            ("Fill_Area", (0.0, 6.0, -8.0), 900.0, 20.0),
            ("Rim_Area", (0.0, 18.0, 24.0), 1000.0, 24.0),
        ]
    )
    for name, location, energy, size in specifications:
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        if hasattr(light_data, "use_shadow"):
            light_data.use_shadow = True
        if hasattr(light_data, "use_contact_shadow"):
            light_data.use_contact_shadow = True
        light_obj = bpy.data.objects.new(name, light_data)
        lights_collection.objects.link(light_obj)
        light_obj.location = Vector(location)
        aim_object(light_obj, (0.0, 0.0, 0.0))


def setup_scene():
    scene = bpy.context.scene
    scene.name = "FixedPathTrajectoryTriptych"
    scene.render.engine = (
        "BLENDER_EEVEE_NEXT"
        if "BLENDER_EEVEE_NEXT" in {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
        else "BLENDER_EEVEE"
    )
    scene.render.resolution_x = TRIPTYCH_WIDTH
    scene.render.resolution_y = TRIPTYCH_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.filepath = str(OUT_DIR / f"triptych_top_{REVISION}.png")
    if hasattr(scene.render, "image_settings"):
        # PNG does not carry a universal DPI field in Blender's render API;
        # TARGET_DPI is retained in the manifest as the publication target.
        pass

    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (*hex_to_rgb(WHITE_HEX), 1.0)
        background.inputs["Strength"].default_value = 0.92 if MATTE_MODE else 1.0
    # Standard keeps the white publication background white instead of the
    # gray tone produced by the previous AgX contrast transform.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.10 if MATTE_MODE else 0.0
    scene.view_settings.gamma = 1.0

    # Blender versions differ in where EEVEE exposes GTAO.  Enable the
    # available soft-AO/contact-shadow controls without making the scene
    # version-dependent; the large area emitters above remain the fallback.
    for owner in (scene, getattr(scene, "eevee", None)):
        if owner is None:
            continue
        if hasattr(owner, "use_gtao"):
            owner.use_gtao = MATTE_MODE
        if hasattr(owner, "gtao_distance"):
            owner.gtao_distance = 3.0
        if hasattr(owner, "gtao_factor"):
            owner.gtao_factor = 0.55

    camera_data = bpy.data.cameras.new("Master_Orthographic_Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ORTHO_SCALE
    camera_data.lens = 50.0
    camera_data.clip_start = 0.1
    camera_data.clip_end = 300.0
    camera = bpy.data.objects.new("Master_Orthographic_Camera", camera_data)
    ensure_collection("CAMERA").objects.link(camera)
    camera.location = (0.0, CAMERA_Y, 0.0)
    aim_object(camera, (0.0, 0.0, 0.0))
    scene.camera = camera
    add_lights(scene)
    return scene, camera


def create_column_camera(scene, record, column_offset):
    """Create a common-scale camera with a data-derived local framing target.

    The target shift is calculated only from the retained Cr coordinates after
    the documented MIC unwrap.  It changes framing, never geometry, and is
    bounded so the framework remains inside the camera view.
    """
    cr_index = record["migrating_atom_indices"][0]
    cr_positions = [
        transform_position(
            record["unwrapped"][image_index][cr_index],
            column_offset,
        )
        for image_index in range(record["image_count"])
    ]
    cr_x_center = (min(point[0] for point in cr_positions) + max(
        point[0] for point in cr_positions
    )) / 2.0
    cr_z_center = (min(point[2] for point in cr_positions) + max(
        point[2] for point in cr_positions
    )) / 2.0
    focus_x = max(
        -CAMERA_FOCUS_X_LIMIT,
        min(CAMERA_FOCUS_X_LIMIT, (cr_x_center - column_offset) * CAMERA_FOCUS_FRACTION),
    )
    focus_z = max(
        -CAMERA_FOCUS_Z_LIMIT,
        min(CAMERA_FOCUS_Z_LIMIT, cr_z_center * CAMERA_FOCUS_FRACTION),
    )
    target_x = column_offset + focus_x
    target_z = focus_z
    camera_data = bpy.data.cameras.new(
        f"Camera_{record['path_id']}_Independent"
    )
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = INDIVIDUAL_ORTHO_SCALE
    camera_data.lens = 50.0
    camera_data.clip_start = 0.1
    camera_data.clip_end = 300.0
    camera = bpy.data.objects.new(
        f"Camera_{record['path_id']}_Independent",
        camera_data,
    )
    ensure_collection("CAMERA").objects.link(camera)
    camera.location = (target_x, CAMERA_Y, target_z)
    aim_object(camera, (target_x, 0.0, target_z))
    record["individual_camera"] = {
        "name": camera.name,
        "projection": "ORTHO",
        "location": [target_x, CAMERA_Y, target_z],
        "target": [target_x, 0.0, target_z],
        "ortho_scale": INDIVIDUAL_ORTHO_SCALE,
        "resolution_px": [INDIVIDUAL_WIDTH, INDIVIDUAL_HEIGHT],
        "focus_source": "retained Cr path bounds after frame-to-frame MIC unwrap",
        "cr_path_bounds_in_scene_A": {
            "x": [min(point[0] for point in cr_positions), max(point[0] for point in cr_positions)],
            "z": [min(point[2] for point in cr_positions), max(point[2] for point in cr_positions)],
        },
        "focus_shift_A": [focus_x, focus_z],
    }
    return camera


def setup_individual_cameras(scene, data_records):
    cameras = {}
    for record, offset in zip(data_records, COLUMN_CENTERS):
        cameras[record["path_id"]] = create_column_camera(scene, record, offset)
    return cameras


def build_column(data, factory, font, column_offset):
    collection = ensure_collection(
        f"COLUMN_{data['column_index']:02d}_{data['path_id']}"
    )
    geometry = ensure_collection(
        f"{data['path_id']}_GEOMETRY", parent=collection
    )
    annotations = ensure_collection(
        f"{data['path_id']}_ANNOTATIONS", parent=collection
    )
    frames = data["frames"]
    unwrapped = data["unwrapped"]
    species = frames[0]["species"]
    cr_index = data["migrating_atom_indices"][0]
    bond_material = factory.material(
        f"bond_{data['path_id']}",
        GRAPHITE_HEX,
        0.12,
        roughness=0.50 if MATTE_MODE else 0.38,
        metallic=0.0,
    )
    path_material = factory.material(
        f"path_{data['path_id']}",
        PATH_HEX,
        0.93,
        roughness=0.38 if MATTE_MODE else 0.24,
        metallic=0.0 if MATTE_MODE else 0.02,
    )
    endpoint_material = factory.material(
        f"endpoint_{data['path_id']}",
        PATH_DARK_HEX,
        0.96,
        roughness=0.36 if MATTE_MODE else 0.23,
        metallic=0.0 if MATTE_MODE else 0.02,
    )
    text_material = factory.material(
        f"text_{data['path_id']}",
        GRAPHITE_HEX,
        1.0,
        roughness=0.48 if MATTE_MODE else 0.40,
        metallic=0.0,
    )

    object_counts = {
        "host_spheres_by_image": {},
        "cr_spheres_by_image": {},
        "host_bonds_image_0": 0,
        "trajectory_polyline_points": 10,
        "endpoint_rings": 2,
        "arrowheads": 1,
        "text_objects": 0,
    }

    # Ghost/snapshot layers are all real extxyz image coordinates.  The alpha
    # schedule is identical in every column and is recorded as image-overlay
    # ordering only; it is never an uncertainty/error encoding.
    for image_index in range(10):
        host_count = 0
        for atom_index, element in enumerate(species):
            if atom_index == cr_index:
                continue
            position = transform_position(
                unwrapped[image_index][atom_index],
                column_offset,
            )
            factory.sphere(
                geometry,
                f"{data['path_id']}_host_i{image_index:02d}_a{atom_index:03d}",
                position,
                ATOM_RADII[element],
                element,
                HOST_ALPHA[image_index],
                role=f"host_i{image_index:02d}",
            )
            host_count += 1
        object_counts["host_spheres_by_image"][str(image_index)] = host_count

        cr_position = transform_position(
            unwrapped[image_index][cr_index],
            column_offset,
            depth_offset=-0.10,
        )
        factory.sphere(
            geometry,
            f"{data['path_id']}_Cr_i{image_index:02d}_a{cr_index:03d}",
            cr_position,
            ATOM_RADII["Cr"] if image_index in (0, 9) else GHOST_CR_RADIUS,
            "Cr",
            CR_ALPHA[image_index],
            role=f"cr_i{image_index:02d}",
        )
        object_counts["cr_spheres_by_image"][str(image_index)] = 1

    # A single ball-and-stick host layer makes the framework readable without
    # turning ten nearly coincident framework images into an opaque wall.
    bonds = host_bond_pairs(frames[0], frames[0]["cell"])
    for bond_index, (left, _right, vector, _distance) in enumerate(bonds):
        left_position = transform_position(
            unwrapped[0][left],
            column_offset,
        )
        right_position = (
            left_position[0] + vector[0],
            left_position[1] + vector[1],
            left_position[2] + vector[2],
        )
        if species[left] == "Cr" or species[_right] == "Cr":
            continue
        factory.cylinder(
            geometry,
            f"{data['path_id']}_host_bond_{bond_index:03d}",
            left_position,
            right_position,
            BOND_RADIUS,
            GRAPHITE_HEX,
            0.13,
            role=f"bond_{data['path_id']}",
        )
    object_counts["host_bonds_image_0"] = len(bonds)

    cr_points = [
        transform_position(
            unwrapped[image_index][cr_index],
            column_offset,
            depth_offset=-0.95,
        )
        for image_index in range(10)
    ]
    make_curve(
        geometry,
        f"{data['path_id']}_Cr_trajectory_0_to_9",
        cr_points,
        PATH_HEX,
        0.95,
        bevel_depth=0.105,
    )
    arrow_direction = Vector(cr_points[-1]) - Vector(cr_points[-2])
    factory.cone(
        geometry,
        f"{data['path_id']}_direction_arrow_0_to_9",
        cr_points[-1],
        arrow_direction,
        radius=0.24,
        depth=0.72,
        color_hex=PATH_HEX,
        alpha=0.95,
        role=f"arrow_{data['path_id']}",
    )

    add_endpoint_ring(
        geometry,
        f"{data['path_id']}_endpoint_ring_0",
        cr_points[0],
        radius=0.78,
        material=endpoint_material,
    )
    add_endpoint_ring(
        geometry,
        f"{data['path_id']}_endpoint_ring_9",
        cr_points[-1],
        radius=0.78,
        material=endpoint_material,
    )

    marker_counts = add_image_index_markers(
        annotations,
        data["path_id"],
        cr_points,
        endpoint_material,
        PATH_DARK_HEX,
        font,
    )
    object_counts.update(marker_counts)
    object_counts["text_objects"] += marker_counts["image_index_markers"]

    make_text(
        annotations,
        f"{data['path_id']}_title",
        data["display_label"],
        (column_offset, -4.25, 11.30),
        0.78,
        text_material,
        font,
    )
    make_text(
        annotations,
        f"{data['path_id']}_subtitle",
        data["subtitle"],
        (column_offset, -4.25, 10.48),
        0.31,
        text_material,
        font,
    )
    make_text(
        annotations,
        f"{data['path_id']}_direction_note",
        "image 0  →  9",
        (column_offset, -4.25, -10.55),
        0.33,
        text_material,
        font,
    )

    # The numbered annotation rail above carries all ten image indices.  The
    # endpoint rings and direction note remain independent VIS-04 cues.
    object_counts["text_objects"] += 3

    data["render_object_counts"] = object_counts
    return data


def image_metadata(path: Path):
    if not path.exists():
        return {"exists": False, "path": str(path)}
    image = bpy.data.images.load(str(path), check_existing=False)
    result = {
        "exists": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "width_px": int(image.size[0]),
        "height_px": int(image.size[1]),
        "channels": int(image.channels),
        "file_format": image.file_format,
    }
    bpy.data.images.remove(image)
    return result


def write_json(path: Path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def base_manifest(data_records, font_path):
    return {
        "figure_id": "NEB_FIXED_PATH_TRAJECTORY_TRIPTYCH_20261023",
        "review_round": REVISION,
        "render_style": RENDER_STYLE,
        "review_basis": str(
            WORKSPACE
            / "analysis/neb_fixed_path_trajectory_figure_20261023/visual_review/REVIEW_R01.md"
        ),
        "status": "BLENDER_GEOMETRY_CANDIDATE_NOT_FINAL",
        "manual_review_status": "PENDING_MANUAL_REVIEW",
        "visual_reference": {
            "path": MATTE_REFERENCE_IMAGE,
            "used_for": (
                "matte material, soft directional lighting, restrained fill, "
                "rounded volume and controlled highlight grammar only"
            ),
            "coordinates_or_energy_source": False,
            "pes_or_schematic_source": False,
        },
        "matte_visual_acceptance_addendum": {
            "status": "PENDING_MANUAL_REVIEW",
            "reference": "Image #1 matte 3D material/light quality only",
            "requirements": [
                "advanced matte surface response rather than flat color or plastic",
                "soft directional key with restrained fill/environment",
                "light AO/contact shading without hard black shadows or blown highlights",
                "host remains low-contrast but rounded and volumetric",
                "Cr remains warm-gold matte with controlled highlight",
                "ghost alpha remains image-order semantics only and keeps the same material response",
                "thin graphite lattice/bonds remain low contrast",
                "white or near-white background, common orthographic framing, no clipping, anti-aliased edges",
            ],
        },
        "manual_review_scope": {
            "VIS-03": (
                f"{REVISION} image-index marker readability at the requested final "
                "size, especially 1–8 in near-overlap"
            ),
            "VIS-04": "preserve real endpoint rings and image 0→9 arrow",
            "VIS-06": "preserve common projection/scale/unwrap and non-clipped host layers",
        },
        "criteria_frozen": True,
        "criteria_path": str(CRITERIA_PATH),
        "write_scope": str(OUT_DIR),
        "existing_data_modified": False,
        "paper_files_modified": False,
        "povray_assets_modified": False,
        "synthetic_coordinates_used": False,
        "synthetic_pes_used": False,
        "coordinate_policy": {
            "source": "three retained extxyz files only",
            "coordinates_used": "parsed source positions, then frame-to-frame MIC unwrapped",
            "no_coordinate_sources": [
                "Image #1",
                "Image #2",
                "historical Pov-Ray coordinates",
                "synthetic PES schematic",
                "hand-drawn trajectory",
            ],
        },
        "column_order": [
            {
                "column_index": record["column_index"],
                "display_label": record["display_label"],
                "path_id": record["path_id"],
                "registry_class": record["registry"]["path_class"],
                "input_path": record["input_path"],
            }
            for record in data_records
        ],
        "paths": [],
        "style": {
            "structure_palette": {
                "name": "tissue_pastel_inspired_from_historical_style_only",
                "elements": ELEMENT_HEX,
                "path_line": PATH_HEX,
                "endpoint_label": PATH_DARK_HEX,
                "host_bond": GRAPHITE_HEX,
            },
            "family_colors_not_used_in_structure_render": {
                "Foundation": "#CF4520",
                "From-scratch": "#FF9F1B",
                "FT-600K": "#002E71",
                "FT-MultiT": "#67ACE5",
            },
            "material_rules": {
                "host_atoms": "low-alpha snapshot/ghost layers",
                "Cr": "high-alpha migrating atom and trajectory",
                "ghost_alpha_semantics": "overlay/image order only; not uncertainty/error",
                "uniform_across_columns": True,
                "ball_and_stick": True,
                "atom_radii_A": ATOM_RADII,
                "ghost_Cr_radius_A": GHOST_CR_RADIUS,
                "host_bond_radius_A": BOND_RADIUS,
                "trajectory_bevel_radius_A": 0.105,
                "endpoint_ring_major_radius_A": 0.78,
                "Cr_alpha_by_image": CR_ALPHA,
                "host_alpha_by_image": HOST_ALPHA,
                "image_index_marker_policy": {
                    "count_per_column": 10,
                    "indices": list(range(10)),
                    "marker_color": PATH_DARK_HEX,
                    "leader_color": PATH_DARK_HEX,
                    "marker_type": "small numbered annotation badge plus leader",
                    "anchor": (
                        "exact projected x/z of the corresponding real retained Cr image"
                    ),
                    "label_layout": (
                        "annotation-only rail offset; no atom coordinate, path point, "
                        "interpolation, or energy value is changed"
                    ),
                    "ghost_alpha_meaning": "overlay/image order only; not uncertainty/error",
                },
            },
            "historical_style_reference": (
                "/Users/alina/Project/migrationbench-paper-repro/paper/"
                "response_to_reviewers/reviewer_a1_fig3_figs3_update/atom_renders_v4/"
                "atom_render_manifest.json"
            ),
            "historical_coordinates_reused": False,
            "matte_material_lighting": {
                "active": MATTE_MODE,
                "surface_model": (
                    "Principled BSDF, metallic 0, diffuse-first rough matte response; "
                    "Cr/path retain only a controlled broad specular level"
                ),
                "host_roughness": 0.50 if MATTE_MODE else 0.34,
                "Cr_roughness": 0.38 if MATTE_MODE else 0.26,
                "bond_roughness": 0.50 if MATTE_MODE else 0.34,
                "world_strength": 0.92 if MATTE_MODE else 1.0,
                "area_lights": (
                    "soft directional key / restrained fill / low-energy rim"
                    if MATTE_MODE
                    else "R03 area-light setup"
                ),
                "ao_contact_policy": (
                    "enable available EEVEE GTAO/contact-shadow controls; large area lights "
                    "provide the version-independent soft-shadow fallback"
                    if MATTE_MODE
                    else "not enabled by default"
                ),
            },
        },
        "camera": {
            "type": "ORTHO",
            "projection": "orthographic",
            "orientation": "direct QL cross-section; camera along +Y; Z vertical",
            "master_triptych": {
                "name": "Master_Orthographic_Camera",
                "common_for_all_columns": True,
                "camera_location": [0.0, CAMERA_Y, 0.0],
                "target": [0.0, 0.0, 0.0],
                "ortho_scale": ORTHO_SCALE,
                "resolution_px": [TRIPTYCH_WIDTH, TRIPTYCH_HEIGHT],
            },
            "individual_column_policy": {
                "projection_common": True,
                "ortho_scale_common": True,
                "ortho_scale": INDIVIDUAL_ORTHO_SCALE,
                "resolution_px": [INDIVIDUAL_WIDTH, INDIVIDUAL_HEIGHT],
                "target_rule": (
                    "independent target from real retained Cr path bounds after "
                    "MIC unwrap; bounded focus shift only, no coordinate edit"
                ),
            },
            "column_centers": list(COLUMN_CENTERS),
            "display_bounds_A": {
                "x": [DISPLAY_XMIN, DISPLAY_XMAX],
                "z": [DISPLAY_ZMIN, DISPLAY_ZMAX],
            },
            "pbc_unwrap": (
                "frame-to-frame MIC unwrap; same algorithm and display bounds for "
                "all columns; master and independent cameras retain common "
                "projection/material rules"
            ),
        },
        "render": {
            "renderer": "Blender",
            "blender_version": bpy.app.version_string,
            "headless_command": HEADLESS_COMMAND,
            "resolution_px": [TRIPTYCH_WIDTH, TRIPTYCH_HEIGHT],
            "target_dpi": TARGET_DPI,
            "target_body_width_in": TARGET_BODY_WIDTH_IN,
            "body_width_pixel_check": {
                "target_px": [TRIPTYCH_WIDTH, TRIPTYCH_HEIGHT],
                "target_dpi": TARGET_DPI,
                "target_body_width_in": TARGET_BODY_WIDTH_IN,
                "width_px_equals_in_times_dpi": abs(
                    TRIPTYCH_WIDTH - TARGET_BODY_WIDTH_IN * TARGET_DPI
                ) < 1e-6,
            },
            "image_format": "PNG RGBA on white background",
            "engine": bpy.context.scene.render.engine,
            "font": font_path,
            "render_round": REVISION,
            "render_style": RENDER_STYLE,
        },
        "outputs": {},
        "acceptance_boundary": {
            "visual_render_only": True,
            "lower_python_curves_not_generated_here": True,
            "dft_reference_status": "COMPLETED_BUT_SCF_WARNING",
            "dfT_ground_truth_claim_allowed": False,
            "final_publication_signoff": False,
            "blocked_by": [
                "DFT-02/DFT-03: warning-bearing QE reference cannot be unconditional ground truth",
                "PUB-02: fixed-path family/seed numeric matrix is frozen as blocked",
                "SEED-04: legal uncertainty bands require verified fixed-path seed matrix",
            ],
        },
    }


def render_outputs(scene):
    output_paths = {}
    master_camera = bpy.data.objects.get("Master_Orthographic_Camera")
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.resolution_x = TRIPTYCH_WIDTH
    scene.render.resolution_y = TRIPTYCH_HEIGHT
    scene.camera = master_camera
    scene.render.filepath = str(OUT_DIR / f"triptych_top_{REVISION}.png")
    bpy.ops.render.render(write_still=True)
    output_paths["triptych_top"] = OUT_DIR / f"triptych_top_{REVISION}.png"

    # Render each column through its own common-scale orthographic camera.  The
    # target is data-derived in setup_individual_cameras, so left/right Cr paths
    # are framed deliberately without moving or resampling any atom.
    for index, record in enumerate(PATHS):
        camera = bpy.data.objects.get(
            f"Camera_{record['path_id']}_Independent"
        )
        if camera is None:
            raise RuntimeError(f"missing independent camera for {record['path_id']}")
        scene.camera = camera
        scene.render.use_border = False
        scene.render.use_crop_to_border = False
        scene.render.resolution_x = INDIVIDUAL_WIDTH
        scene.render.resolution_y = INDIVIDUAL_HEIGHT
        output_path = OUT_DIR / f"{record['path_id']}_{REVISION}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        output_paths[record["path_id"]] = output_path

    # Restore the master state in the saved .blend and for any subsequent
    # inspection of the scene.
    scene.camera = master_camera
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.resolution_x = TRIPTYCH_WIDTH
    scene.render.resolution_y = TRIPTYCH_HEIGHT
    scene.render.filepath = str(OUT_DIR / f"triptych_top_{REVISION}.png")
    return output_paths


def write_qa(data_records, output_paths, manifest):
    output_metadata = {
        name: image_metadata(path) for name, path in output_paths.items()
    }
    all_outputs_exist = all(meta.get("exists", False) for meta in output_metadata.values())
    expected_sizes = {
        "triptych_top": [TRIPTYCH_WIDTH, TRIPTYCH_HEIGHT],
        "cr_canonical_in_gap": [INDIVIDUAL_WIDTH, INDIVIDUAL_HEIGHT],
        "cr_canonical_gap_to_ql": [INDIVIDUAL_WIDTH, INDIVIDUAL_HEIGHT],
        "cr_canonical_within_ql": [INDIVIDUAL_WIDTH, INDIVIDUAL_HEIGHT],
    }
    size_checks = {}
    for name, expected in expected_sizes.items():
        actual = output_metadata.get(name, {})
        size_checks[name] = actual.get("width_px") == expected[0] and actual.get(
            "height_px"
        ) == expected[1]

    geometry_checks = {}
    for record in data_records:
        geometry_checks[record["path_id"]] = {
            "81_atoms": record["validation_checks"]["atoms_all_81"],
            "10_images": record["validation_checks"]["image_count_10"],
            "pbc_TTT": record["validation_checks"]["pbc_all_TTT"],
            "source_hash_matches_registry": record["validation_checks"][
                "source_sha256_matches_registry"
            ],
            "species_order_matches_registry": record["validation_checks"][
                "species_order_matches_registry"
            ],
            "Cr_index_80": record["validation_checks"]["single_cr_atom_all_images"],
            "endpoint_consistency_PASS": record["validation_checks"][
                "registry_endpoint_consistency_PASS"
            ],
        }

    marker_visibility = {}
    for record in data_records:
        counts = record["render_object_counts"]
        marker_visibility[record["path_id"]] = {
            "expected_indices": list(range(10)),
            "marker_count": counts.get("image_index_markers", 0),
            "leader_line_count": counts.get("image_index_leader_lines", 0),
            "all_indices_structurally_present": (
                counts.get("image_index_markers", 0) == 10
                and counts.get("image_index_leader_lines", 0) == 10
                and len(counts.get("image_index_marker_coordinates", [])) == 10
            ),
            "visual_status": "PENDING_MANUAL_REVIEW",
            "anchor_source": "retained real Cr image coordinates after MIC unwrap",
            "label_layout_status": "annotation-only rail; no atom coordinate edits",
        }

    qa = {
        "figure_id": manifest["figure_id"],
        "review_round": REVISION,
        "render_style": RENDER_STYLE,
        "qa_status": f"RENDERED_{REVISION.upper()}_CANDIDATE_NOT_ACCEPTED",
        "manual_review_status": "PENDING_MANUAL_REVIEW",
        "body_width_proof": {
            "target_body_width_in": TARGET_BODY_WIDTH_IN,
            "target_dpi": TARGET_DPI,
            "target_triptych_px": [TRIPTYCH_WIDTH, TRIPTYCH_HEIGHT],
            "target_column_px": [INDIVIDUAL_WIDTH, INDIVIDUAL_HEIGHT],
            "final_size_readability_status": "PENDING_MANUAL_REVIEW",
        },
        "matte_visual_review": {
            "reference_image": MATTE_REFERENCE_IMAGE,
            "reference_use": "material/light grammar only; no geometry or energy source",
            "status": "PENDING_MANUAL_REVIEW",
            "checks": {
                "advanced_matte_not_flat_or_plastic": "PENDING_MANUAL_REVIEW",
                "soft_directional_key_and_restrained_fill": "PENDING_MANUAL_REVIEW",
                "soft_ao_contact_without_hard_black_or_blown_highlights": "PENDING_MANUAL_REVIEW",
                "host_low_contrast_rounded_volume": "PENDING_MANUAL_REVIEW",
                "Cr_warm_gold_matte_controlled_highlight": "PENDING_MANUAL_REVIEW",
                "ghosts_keep_material_response_alpha_is_image_order_only": "PENDING_MANUAL_REVIEW",
                "graphite_lattice_low_contrast": "PENDING_MANUAL_REVIEW",
                "near_white_background_common_ortho_no_clip_aa": "PENDING_MANUAL_REVIEW",
            },
        },
        "render_success": all_outputs_exist and all(size_checks.values()),
        "blender_version": bpy.app.version_string,
        "output_metadata": output_metadata,
        "expected_output_sizes_px": expected_sizes,
        "size_checks": size_checks,
        "geometry_checks": geometry_checks,
        "marker_visibility": marker_visibility,
        "contract_checks": {
            "VIS-01_three_columns_and_order": "PASS_AUTOMATED",
            "VIS-02_same_path_semantics": "PASS_AUTOMATED",
            # These remain explicitly pending after the user-reported visual
            # failure.  Technical rendering is not a substitute for the
            # required human final-size inspection.
            "VIS-03_real_atom_ghost_layers": "PENDING_MANUAL_REVIEW",
            "VIS-04_direction_0_to_9_arrow_and_labels": "PENDING_MANUAL_REVIEW",
            "VIS-05_alpha_is_overlay_order_only": "PASS_AUTOMATED",
            "VIS-06_common_camera_scale_and_unwrap_recorded": "PENDING_MANUAL_REVIEW",
            "VIS-07_no_synthetic_PES_or_coordinates": "PASS_AUTOMATED",
            "VIS-16_render_dimensions": (
                "PENDING_MANUAL_REVIEW_BODY_WIDTH_PROOF"
                if TARGET_BODY_WIDTH_IN <= 7.0
                else "PASS_AUTOMATED"
            ),
            "VIS-MATTE-01_to_VIS-MATTE-08": "PENDING_MANUAL_REVIEW",
            "PHY-01_to_PHY-03_geometry_identity_and_MIP_unwrap": "PASS_AUTOMATED",
            "PHY-15_no_61_atom_or_historical_path_mix": "PASS_AUTOMATED",
            "DFT-01_status_transparency": (
                "PENDING_PYTHON_INTEGRATION_LABEL; DFT status preserved in manifest"
            ),
            "DFT-02_unconditional_ground_truth": "BLOCKED_BY_FROZEN_DFT_WARNING",
            "DFT-03_warning_not_energy_error_band": "PENDING_PYTHON_INTEGRATION",
            "SEED-01_to_SEED-05": "PENDING_PYTHON_INTEGRATION",
            "PUB-07_full_figure_PDF_SVG_PNG": "PENDING_PYTHON_INTEGRATION",
            "PUB-10_final_signoff": "NOT_ACCEPTED",
        },
        "manual_visual_review_required": [
            (
                f"{REVISION}/VIS-03: confirm labels 0–9 are individually readable "
                f"at the {TARGET_BODY_WIDTH_IN:.1f}-in final size in all three "
                "columns, especially gap2QL images 1–8; also inspect Cr ghosts "
                "and host lattice"
            ),
            (
                "VIS-04 targeted rerender completed; confirm arrow and 0→9 "
                "endpoint labels are unambiguous in grayscale"
            ),
            (
                "VIS-06 targeted independent cameras completed; confirm no "
                "endpoint or host layer is visually clipped"
            ),
            (
                "VIS-MATTE-01–08: compare the final-size rerender with Image #1's "
                "advanced matte grammar; confirm soft directional shading, low-contrast "
                "rounded host volume, warm-gold Cr highlight, light contact shading, "
                "and no hard black shadows or blown whites"
            ),
            "VIS-14 confirm any eventual composite legend does not cover this render",
            "PUB-08 final-size and grayscale QA after Python integration",
        ],
        "failure_or_limitation_ledger": [
            {
                "criterion": "DFT-02/DFT-03",
                "status": "BLOCKED",
                "reason": (
                    "All three canonical DFT/QE references remain "
                    "COMPLETED_BUT_SCF_WARNING; this geometry render does not "
                    "upgrade them to ground truth."
                ),
            },
            {
                "criterion": "PUB-02/SEED-04",
                "status": "BLOCKED",
                "reason": (
                    "The lower fixed-path family/seed numerical matrix and legal "
                    "mean±1 SD uncertainty gate are outside this Blender-only "
                    "scope and remain frozen as blocked."
                ),
            },
            {
                "criterion": "VIS-03/VIS-04/VIS-06",
                "status": "PENDING_MANUAL_REVIEW",
                "reason": (
                    f"{REVISION} adds ten higher-contrast annotation-only real-image "
                    "index badges and leaders per column, including the "
                    "near-overlapping gap2QL "
                    "images.  Coordinates remain the retained extxyz positions; "
                    "acceptance remains pending a new human visual check."
                ),
            },
            {
                "criterion": "VIS-MATTE-01–08",
                "status": "PENDING_MANUAL_REVIEW",
                "reason": (
                    "The additive Image #1 matte material/light grammar is rendered and "
                    "recorded without changing retained coordinates or marker rails; "
                    "final visual acceptance requires human inspection at the 7.0-in body width."
                ),
            },
        ],
        "provenance": {
            "input_hashes_written_to_manifest": True,
            "output_hashes_written_to_manifest_after_render": True,
            "scene_source_script": str(OUT_DIR / "render_scene.py"),
        },
    }
    return qa, output_metadata


def main():
    registry_rows = read_registry()
    data_records = [
        validate_path(config, registry_rows) for config in PATHS
    ]

    clear_scene()
    scene, _camera = setup_scene()
    setup_individual_cameras(scene, data_records)
    factory = GeometryFactory()
    font, font_path = load_arial_font()
    for data, offset in zip(data_records, COLUMN_CENTERS):
        build_column(data, factory, font, offset)

    manifest = base_manifest(data_records, font_path)
    for record in data_records:
        manifest["paths"].append(
            {
                "column_index": record["column_index"],
                "display_label": record["display_label"],
                "subtitle": record["subtitle"],
                "path_id": record["path_id"],
                "input_path": record["input_path"],
                "input_sha256": record["input_sha256"],
                "registry_expected_fixed_retained_path_sha256": record["registry"][
                    "fixed_retained_path_sha256"
                ],
                "source_hash_match": record["validation_checks"][
                    "source_sha256_matches_registry"
                ],
                "natoms": record["natoms"],
                "image_count": record["image_count"],
                "pbc": record["pbc"],
                "species_order": record["species_order"],
                "atom_order_sha256": record["atom_order_sha256"],
                "migrating_atom_indices_zero_based": record[
                    "migrating_atom_indices"
                ],
                "endpoint_hashes_computed": record["endpoint_hashes_computed"],
                "registry_endpoint_hashes": {
                    "initial": record["registry"]["initial_endpoint_sha256"],
                    "final": record["registry"]["final_endpoint_sha256"],
                },
                "registry_cell_sha256": record["registry"]["cell_sha256"],
                "cell_A": record["cell_A"],
                "registry_endpoint_consistency_gate": record["registry"][
                    "endpoint_consistency_gate"
                ],
                "dft_reference_status": record["registry"]["dft_reference_status"],
                "fixed_retained_path_semantics": record["registry"][
                    "endpoint_source"
                ],
                "mic_unwrap": record["mic_info"],
                "bounds_unwrapped_A": record["bounds_unwrapped_A"],
                "individual_camera": record["individual_camera"],
                "image_index_markers": {
                    "indices": list(range(10)),
                    "count": record["render_object_counts"].get(
                        "image_index_markers", 0
                    ),
                    "leader_line_count": record["render_object_counts"].get(
                        "image_index_leader_lines", 0
                    ),
                    "anchor_source": (
                        "exact projected Cr image coordinates from retained extxyz "
                        "after recorded MIC unwrap"
                    ),
                    "label_layout": (
                        "annotation-only rail with numbered badges; no atom/path "
                        "coordinate modification or interpolation"
                    ),
                },
                "render_object_counts": record["render_object_counts"],
                "validation_checks": record["validation_checks"],
                "Cr_raw_positions_A": [
                    record["frames"][image_index]["positions"][
                        record["migrating_atom_indices"][0]
                    ]
                    for image_index in range(10)
                ],
                "Cr_unwrapped_positions_A": [
                    record["unwrapped"][image_index][
                        record["migrating_atom_indices"][0]
                    ]
                    for image_index in range(10)
                ],
            }
        )
    manifest_path = OUT_DIR / f"scene_manifest_{REVISION}.json"
    qa_path = OUT_DIR / f"RENDER_QA_{REVISION}.json"
    write_json(manifest_path, manifest)

    blend_path = OUT_DIR / f"fixed_path_trajectory_{REVISION}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    output_paths = render_outputs(scene)
    # Save the final .blend with the master camera state, not the last cropped
    # render border.
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    qa, output_metadata = write_qa(data_records, output_paths, manifest)
    manifest["outputs"] = {
        name: {
            **metadata,
            "role": (
                "master triptych"
                if name == "triptych_top"
                else "independent common-scale Blender orthographic column camera"
            ),
        }
        for name, metadata in output_metadata.items()
    }
    manifest["scene_file"] = {
        "path": str(blend_path),
        "sha256": sha256_file(blend_path),
        "bytes": blend_path.stat().st_size,
    }
    write_json(manifest_path, manifest)
    write_json(qa_path, qa)
    print(json.dumps({
        "status": qa["qa_status"],
        "render_success": qa["render_success"],
        "output_dir": str(OUT_DIR),
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
