#!/usr/bin/env python3
"""Canonical 81-atom Cr-in-Sb2Te3 DFT-NEB reference profiles, read from QE output.

Replaces the energy arrays that used to be hard-coded in
``make_cr_three_path_2x3_R0*.py::dft_reference()`` (copied from a vector PDF).
The profiles are taken from the last complete NEB iteration of the committed
Quantum ESPRESSO ``neb.out`` files under
``data/raw_compact/cr81_dft_neb_reference_20261013/raw/`` (Rockfish origin:
``/scratch16/pclancy3/yi/MigrationBench_rockfish_cr_neb_20261013``).

The QE directory names say "2L-Cu" for historical reasons; the structures are
the 81-atom Cr-doped system. None of the three runs passes the default
convergence gate (max image force <= 0.03 eV/A), which is reported alongside
the profiles rather than hidden.

Run as a script to write the reference table:
    python3 src/migrationbench/dft_neb_reference.py --output data/processed/cr81_dft_neb_reference_profiles.tsv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_qe_neb_output import parse_neb_out  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "data/raw_compact/cr81_dft_neb_reference_20261013/raw"
PATH_DIRS = {
    "in_gap": "2L-Cu__neb-gap-diffuse",
    "gap_to_ql": "2L-Cu__neb-gap2QL",
    "within_ql": "2L-Cu__neb-QLs",
}
FIELDS = [
    "path_key", "image_index", "reaction_coordinate", "energy_eV", "relative_energy_eV",
    "image_force_error_eV_A", "neb_iteration", "converged_by_default_gate", "source_neb_out",
]


def load_path(path_key: str, root: Path = REFERENCE_ROOT) -> dict:
    """Return the last complete NEB iteration for one path, energies relative to image 1."""
    if path_key not in PATH_DIRS:
        raise KeyError(f"unknown path_key {path_key!r}; expected one of {sorted(PATH_DIRS)}")
    neb_out = root / PATH_DIRS[path_key] / "neb.out"
    if not neb_out.is_file():
        raise FileNotFoundError(f"DFT-NEB reference missing: {neb_out}")
    parsed = parse_neb_out(neb_out)
    if not parsed["complete_iterations"]:
        raise ValueError(f"no complete NEB iteration in {neb_out}")
    last = parsed["complete_iterations"][-1]
    images = sorted(last["images"], key=lambda image: image["image_index"])
    if any(image["energy_overflow"] for image in images):
        raise ValueError(f"overflowed image energy in final iteration of {neb_out}")
    first = images[0]["energy_eV"]
    n_steps = len(images) - 1
    return {
        "path_key": path_key,
        "neb_out": neb_out,
        "iteration": last["iteration"],
        "converged": parsed["converged_by_default_gate"],
        "x": [index / n_steps for index in range(len(images))],
        "energy_eV": [image["energy_eV"] for image in images],
        "relative_energy_eV": [image["energy_eV"] - first for image in images],
        "force_error_eV_A": [image["error_eV_A"] for image in images],
    }


def profiles_for_plot(root: Path = REFERENCE_ROOT):
    """Same shape as the former hard-coded ``dft_reference()``: {path: (x, relative_E)}."""
    import numpy as np

    return {
        key: (np.asarray(profile["x"]), np.asarray(profile["relative_energy_eV"]))
        for key, profile in ((key, load_path(key, root)) for key in PATH_DIRS)
    }


def reference_rows(root: Path = REFERENCE_ROOT) -> list[dict]:
    rows = []
    for key in PATH_DIRS:
        profile = load_path(key, root)
        for index, x in enumerate(profile["x"]):
            rows.append({
                "path_key": key,
                "image_index": index + 1,
                "reaction_coordinate": f"{x:.6f}",
                "energy_eV": f"{profile['energy_eV'][index]:.7f}",
                "relative_energy_eV": f"{profile['relative_energy_eV'][index]:.7f}",
                "image_force_error_eV_A": f"{profile['force_error_eV_A'][index]:.6f}",
                "neb_iteration": profile["iteration"],
                "converged_by_default_gate": profile["converged"],
                "source_neb_out": str(profile["neb_out"].relative_to(REPO_ROOT)),
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(reference_rows())
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
