"""Real MigrationBench coordinates → Scenes (slabs, paths, local environments).

Sources: Cr81 fixed-path NEB images (public/data/geometry.json, exported from the
NEB r01 inputs) and the DFT interlayer-sliding path (.xyz, same supercell).
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .atoms import Scene

PROJECT = Path(__file__).resolve().parents[3]
GEOMETRY_JSON = PROJECT / "public/data/geometry.json"
PROFILES_JSON = PROJECT / "public/data/cr_neb_profiles.json"
SLIDING_DIR = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/"
                   "sliding_four_model_campaign_20261024/inputs/dft")
GAP_Z = (22.3, 25.1)
BOND_CUTOFF = 3.4


def _read_xyz(path: Path):
    lines = path.read_text().splitlines()
    frames, species, i = [], [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        rows = [ln.split() for ln in lines[i + 2 : i + 2 + n]]
        species = [r[0] for r in rows]
        frames.append([[float(v) for v in r[1:4]] for r in rows])
        i += n + 2
    return species, np.array(frames)


@lru_cache
def paths() -> dict:
    geo = json.loads(GEOMETRY_JSON.read_text())["paths"]
    out = {k: {"species": v["species"], "images": np.array(v["images"]),
               "cell": np.array(v["lattice"]).reshape(3, 3)} for k, v in geo.items()}
    cell = out["within_gap"]["cell"]
    for key, name in (("sliding_cr", "cr_in_gap_sb2te3.xyz"), ("sliding_pristine", "pristine_sb2te3.xyz")):
        species, images = _read_xyz(SLIDING_DIR / name)
        # Sliding .xyz files carry no cell; they are the same Sb2Te3 supercell.
        out[key] = {"species": species, "images": images, "cell": cell}
    return out


@lru_cache
def dft_peak_image() -> dict:
    out = {}
    for p in json.loads(PROFILES_JSON.read_text())["paths"]:
        e = [pt["e"] for pt in next(s for s in p["series"] if s["kind"] == "reference")["points"]]
        out[p["key"]] = int(np.argmax(e)) if max(e) > 0 else None
    return out


def cr_index(key: str) -> int | None:
    sp = paths()[key]["species"]
    return sp.index("Cr") if "Cr" in sp else None


def tiled(key: str, image: int, reps: int = 2):
    p = paths()[key]
    pos = p["images"][image]
    shifts = [i * p["cell"][0] + j * p["cell"][1] for i in range(-reps, reps + 1) for j in range(-reps, reps + 1)]
    return np.concatenate([pos + s for s in shifts]), np.tile(np.arange(len(pos)), len(shifts))


def box_mask(pos, center, half_x, y_range, z_range=(14.5, 33.0)):
    d = pos - center
    return ((np.abs(d[:, 0]) <= half_x) & (d[:, 1] >= y_range[0]) & (d[:, 1] <= y_range[1])
            & (pos[:, 2] >= z_range[0]) & (pos[:, 2] <= z_range[1]))


def slab(key: str, image: int = 0, center=None, half_x=9.5, y_range=(-5.5, 1.2), z_range=(14.5, 33.0),
         include_cr=True, host_alpha=1.0, offsets=None):
    """Wide, shallow block of the bilayer around `center` (default: Cr).

    y_range is asymmetric (Å, relative to centre) so Cr sits near the front face.
    """
    p = paths()[key]
    ci = cr_index(key)
    if center is None:
        center = p["images"][image][ci] if ci is not None else p["images"][image].mean(axis=0)
    pos, idx = tiled(key, image)
    keep = box_mask(pos, center, half_x, y_range, z_range)
    scene = Scene()
    lookup = {}
    for k in np.flatnonzero(keep):
        if idx[k] == ci:
            continue
        r = pos[k] + (offsets[k] if offsets is not None else 0)
        lookup[k] = scene.add(r, p["species"][idx[k]], alpha=host_alpha)
    if include_cr and ci is not None:
        scene.add(p["images"][image][ci], "Cr", glow=True, scale=1.08)
    return scene, np.asarray(center), {"pos": pos, "idx": idx, "keep": keep, "lookup": lookup}


def stroboscope(scene: Scene, key: str, ramp, ring_image=None, every=1, trail_color="#c98a12"):
    """Add Cr images 0..N-1 as a light→dark series plus a dotted trail."""
    p = paths()[key]
    ci = cr_index(key)
    traj = p["images"][:, ci]
    n = len(traj)
    for i in range(0, n, every):
        last = i == n - 1
        scene.add(traj[i], "Cr", color=ramp(i / (n - 1)), scale=1.08 if last or i == 0 else 0.8,
                  alpha=1.0, ring=(i == ring_image), glow=last)
    scene.trails.append((traj, trail_color, 0.22))
    return traj


def local_environment(key="within_gap", image=0, cutoff=BOND_CUTOFF):
    """Cr and its first shell as a ball-and-stick Scene (real coordinates)."""
    p = paths()[key]
    ci = cr_index(key)
    r_cr = p["images"][image][ci]
    pos, idx = tiled(key, image)
    d = np.linalg.norm(pos - r_cr, axis=1)
    shell = np.flatnonzero((d <= cutoff) & (idx != ci))
    scene = Scene()
    cr = scene.add(r_cr, "Cr", scale=1.25, glow=True)
    for k in shell:
        j = scene.add(pos[k], p["species"][idx[k]], scale=1.1)
        scene.bonds.append((cr, j))
    return scene, r_cr, d[shell]


def thermal_offsets(n: int, amplitude: float, seed: int) -> np.ndarray:
    """SCHEMATIC thermal displacements (not MD data)."""
    return np.random.default_rng(seed).normal(0, amplitude, size=(n, 3))
