"""Render clean, coordinate-driven five-image QL cross-sections.

This renderer is deliberately separate from the main operator schematic.  It
uses the same 61-atom DFT and MLIP paths as the earlier structural anchors,
but emits transparent POV-Ray renders for the two standalone five-panel
figures.  The element colours come from the registered Plot Atlas
``tissue_pastel`` palette and are recorded in the manifest.

The trajectory is shown as translucent Cr spheres connected by plain
cylinders.  There are no arrowheads or labels in the atom renders; panel
ordering and branch identity are carried by the surrounding composition and
manifest instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ATLAS = Path(
    __import__("os").environ.get(
        "MIGRATIONBENCH_PLOT_ATLAS", "/Users/alina/Documents/Codex/plot_library"
    )
)
sys.path.insert(0, str(ATLAS / "src"))
from plot_atlas.palettes import get_palette  # noqa: E402


DFT_PATH = REPO / "tests/fixtures/historical_neb/legacy_61/1-4/sb2te3.xyz"
MLIP_PATH = HERE / "data/self_relaxed_foundation_path_61_1-4.extxyz"

ATOM_PALETTE_NAME = "tissue_pastel"
ATOM_PALETTE = get_palette(ATOM_PALETTE_NAME)


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hex colour, got {value!r}")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


# Fixed element mapping for this paper.  Keep this mapping unchanged across
# every atom panel so the visual identity is not reinterpreted per branch.
ELEMENT_HEX = {
    "Te": ATOM_PALETTE["categorical"][5],  # muted lilac
    "Sb": ATOM_PALETTE["categorical"][1],  # soft teal
    "Cr": ATOM_PALETTE["categorical"][3],  # soft ochre
}
ELEMENT_RGB = {key: hex_to_rgb(value) for key, value in ELEMENT_HEX.items()}
RADII = {"Te": 0.78, "Sb": 0.88, "Cr": 1.10}
GHOST_RADIUS = 0.46


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def povray_version() -> str | None:
    executable = shutil.which("povray")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "--version"], check=False, capture_output=True, text=True
    )
    lines = (result.stdout + "\n" + result.stderr).strip().splitlines()
    return lines[0] if lines else executable


def local_framework(atoms: Atoms, center: np.ndarray) -> Atoms:
    """Keep a compact side-view slab containing several stacked QL layers."""
    scene = atoms.copy()
    delta = scene.positions - center
    # x is the in-plane horizontal direction, y is the viewing direction, and
    # z is the stacked-layer vertical direction in the final camera view.
    mask = np.all(np.abs(delta) <= np.array([5.8, 3.25, 6.9]), axis=1)
    scene = scene[mask]
    if len(scene) < 18:
        raise RuntimeError("QL crop unexpectedly removed most framework atoms")
    scene.set_positions(scene.positions - center)
    scene.set_cell(np.zeros((3, 3)))
    scene.set_pbc(False)
    # The framework is kept, while Cr is represented by the explicit path
    # overlay below so the trajectory is never hidden by the current atom.
    return scene[[symbol != "Cr" for symbol in scene.get_chemical_symbols()]]


def render_one(
    atoms: Atoms,
    path_frames: list[Atoms],
    output_pov: Path,
    center: np.ndarray,
    current_index: int,
) -> None:
    framework = local_framework(atoms, center)
    path_positions = np.array(
        [frame.positions[frame.symbols == "Cr"][0] for frame in path_frames]
    )
    path_positions = path_positions - center
    path_atoms = Atoms(symbols=["Cr"] * len(path_positions), positions=path_positions)
    scene = framework + path_atoms

    colors = [ELEMENT_RGB[symbol] for symbol in framework.get_chemical_symbols()]
    colors.extend([ELEMENT_RGB["Cr"]] * len(path_positions))
    radii = [RADII[symbol] for symbol in framework.get_chemical_symbols()]
    radii.extend([GHOST_RADIUS] * len(path_positions))
    # POV-Ray transmittance is zero for opaque and increases toward transparent.
    transmittances = [0.0] * len(framework) + [0.58] * len(path_positions)
    transmittances[len(framework) + current_index] = 0.0
    colors[len(framework) + current_index] = ELEMENT_RGB["Cr"]
    radii[len(framework) + current_index] = RADII["Cr"]

    ghost_start = len(framework)
    trajectory_bonds = [
        (ghost_start + index, ghost_start + index + 1, (0, 0, 0))
        for index in range(len(path_positions) - 1)
    ]
    settings = {
        "display": False,
        "pause": False,
        "transparent": True,
        "canvas_width": 920,
        "camera_type": "orthographic",
        "camera_dist": 65.0,
        "area_light": [(10.0, -14.0, 48.0), "White", 5.0, 5.0, 5, 5],
        "point_lights": [((-14.0, 10.0, 38.0), "White")],
        "background": "White",
        "depth_cueing": False,
        "celllinewidth": 0.0,
        "bondlinewidth": 0.075,
        "bondatoms": trajectory_bonds,
        "transmittances": transmittances,
    }
    write(
        output_pov,
        scene,
        format="pov",
        # Camera looks along +y with +z vertical: a direct QL cross-section.
        rotation="-90x, 0y, 0z",
        radii=radii,
        colors=colors,
        povray_settings=settings,
    )
    subprocess.run(
        ["povray", output_pov.with_suffix(".ini").name],
        cwd=output_pov.parent,
        check=True,
    )


def render(output_dir: Path) -> None:
    if shutil.which("povray") is None:
        raise RuntimeError("POV-Ray executable was not found on PATH")
    dft_frames = read(DFT_PATH, ":")
    mlip_frames = read(MLIP_PATH, ":")
    if len(dft_frames) != 5 or len(mlip_frames) != 5:
        raise RuntimeError("Expected five frames in both atomistic path sources")
    if len(dft_frames[0]) != 61 or len(mlip_frames[0]) != 61:
        raise RuntimeError("Expected 61 atoms in both atomistic path sources")
    if dft_frames[0].get_chemical_symbols() != mlip_frames[0].get_chemical_symbols():
        raise RuntimeError("Atom ordering differs between DFT and MLIP paths")

    cr_index = dft_frames[0].get_chemical_symbols().index("Cr")
    common_center = 0.5 * (
        dft_frames[0].positions[cr_index] + dft_frames[-1].positions[cr_index]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "STRUCTURAL_VISUAL_ANCHOR_WITH_EXPLICIT_CR_TRAJECTORY",
        "renderer": "ASE POV-Ray writer",
        "povray": povray_version(),
        "frame_count": 5,
        "atom_count": 61,
        "view": "direct QL cross-section; camera along +y, z vertical",
        "rotation": "-90x, 0y, 0z",
        "trajectory": "five translucent Cr ghosts connected by plain POV-Ray cylinders; current image opaque",
        "transparent_background": True,
        "palette": {
            "name": ATOM_PALETTE_NAME,
            "element_hex": ELEMENT_HEX,
            "element_roles": {
                "Te": "framework / muted lilac",
                "Sb": "framework / soft teal",
                "Cr": "migrating atom / soft ochre",
            },
            "high_saturation_red_used": False,
        },
        "common_center_A": [float(x) for x in common_center],
        "inputs": {
            "dft_path": {"path": str(DFT_PATH.relative_to(REPO)), "sha256": sha256(DFT_PATH)},
            "mlip_path": {"path": str(MLIP_PATH.relative_to(REPO)), "sha256": sha256(MLIP_PATH)},
        },
        "frames": [],
        "notes": [
            "The framework is cropped to a compact side/front view showing stacked QL layers.",
            "DFT frames are the five-image fixed-path structural sequence.",
            "MLIP frames are the current Foundation self-relaxed path and are provisional.",
            "No atom render contains text, arrows, or arrowheads.",
            "Panel identity is carried by the standalone composition filename and manifest.",
        ],
    }
    for branch, frames in (("dft", dft_frames), ("mlip", mlip_frames)):
        for index, atoms in enumerate(frames):
            pov_path = output_dir / f"{branch}_{index:02d}.pov"
            render_one(atoms, frames, pov_path, common_center, index)
            png_path = pov_path.with_suffix(".png")
            if not png_path.exists():
                raise RuntimeError(f"POV-Ray did not produce {png_path}")
            manifest["frames"].append(
                {
                    "branch": branch,
                    "image_index": index,
                    "pov": str(pov_path),
                    "png": str(png_path),
                    "png_sha256": sha256(png_path),
                }
            )
    manifest["script_sha256"] = sha256(Path(__file__))
    manifest_path = output_dir / "atom_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
