"""Render five atomistic path snapshots with ASE's POV-Ray backend.

The rendered PNGs are deliberately transparent and contain no labels.  The
publication figure keeps all scientific labels, paths, and equations in the
Matplotlib/SVG layer so that those elements remain editable.  The two path
sources are the audited 61-atom DFT image sequence and the current Foundation
MLIP self-relaxed 1-4 image sequence.  The latter is a provisional MLIP
preconditioner path, not a DFT-validated result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase.io import read, write


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

DFT_PATH = REPO / "tests/fixtures/historical_neb/legacy_61/1-4/sb2te3.xyz"
MLIP_PATH = HERE / "data/self_relaxed_foundation_path_61_1-4.extxyz"

# Publication-consistent colors; the exact vector labels are kept outside the
# atom renders.  These colors are chosen to match the existing v1 thumbnails.
COLORS = {
    "Te": (0.63, 0.55, 0.68),
    "Sb": (0.29, 0.50, 0.72),
    "Cr": (0.86, 0.55, 0.05),
}
RADII = {
    "Te": 0.68,
    "Sb": 0.78,
    "Cr": 1.02,
}


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
    try:
        result = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return executable
    text = (result.stdout + "\n" + result.stderr).strip().splitlines()
    return text[0] if text else executable


def local_scene(atoms, center: np.ndarray):
    """Return a common-frame, non-periodic copy for stable snapshot framing."""
    scene = atoms.copy()
    # Keep a common local slab around the migration event.  This preserves the
    # surrounding coordination environment while avoiding the large vacuum
    # direction of the full 61-atom supercell in the thumbnail.
    delta = scene.positions - center
    mask = np.all(np.abs(delta) <= np.array([5.8, 5.8, 6.4]), axis=1)
    if int(mask.sum()) < 18:
        raise RuntimeError("Local atom crop unexpectedly removed most atoms")
    scene = scene[mask]
    scene.set_positions(scene.positions - center)
    # The slab is intentionally shown as a finite publication crop.  Removing
    # the periodic cell prevents POV-Ray from drawing a large vacuum cell.
    scene.set_cell(np.zeros((3, 3)))
    scene.set_pbc(False)
    return scene


def render_one(atoms, output_pov: Path, center: np.ndarray) -> None:
    scene = local_scene(atoms, center)
    colors = [COLORS[symbol] for symbol in scene.get_chemical_symbols()]
    radii = [RADII[symbol] for symbol in scene.get_chemical_symbols()]
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
        "bondlinewidth": 0.0,
    }
    write(
        output_pov,
        scene,
        format="pov",
        rotation="-45x, 12z",
        radii=radii,
        colors=colors,
        povray_settings=settings,
    )
    # ASE writes a relative Input_File_Name into the .ini file.  Invoke
    # POV-Ray from the output directory so the sibling .pov file is found on
    # macOS/Homebrew as well as on Linux.
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
        "status": "STRUCTURAL_VISUAL_ANCHOR_ONLY",
        "renderer": "ASE POV-Ray writer",
        "povray": povray_version(),
        "frame_count": 5,
        "atom_count": 61,
        "rotation": "-45x, 12z",
        "transparent_background": True,
        "common_center_A": [float(x) for x in common_center],
        "inputs": {
            "dft_path": {"path": str(DFT_PATH.relative_to(REPO)), "sha256": sha256(DFT_PATH)},
            "mlip_path": {
                "path": str(MLIP_PATH.relative_to(REPO)),
                "sha256": sha256(MLIP_PATH),
            },
        },
        "frames": [],
        "notes": [
            "DFT frames are the five-image fixed-path structural sequence.",
            "MLIP frames are the current Foundation self-relaxed path and are provisional.",
            "No atom render contains text; labels are added by the editable vector figure.",
        ],
    }
    for branch, frames in (("dft", dft_frames), ("mlip", mlip_frames)):
        for index, atoms in enumerate(frames):
            pov_path = output_dir / f"{branch}_{index:02d}.pov"
            render_one(atoms, pov_path, common_center)
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
    (output_dir / "atom_render_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote {output_dir / 'atom_render_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
