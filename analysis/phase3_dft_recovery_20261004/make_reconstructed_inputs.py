#!/usr/bin/env python3
"""Build provenance-preserving 81-atom local QE inputs from full trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_frames"
TEMPLATE_ROOT = ROOT.parent / "phase3_dft_validation_20261002" / "qe_inputs_81"
INPUT_ROOT = ROOT / "qe_inputs_reconstructed"

CASES = {
    "normal_461800_initial": {
        "source": SOURCE / "normal_461800_initial.extxyz",
        "reference": SOURCE / "normal_461800_initial.extxyz",
        "template": TEMPLATE_ROOT / "step_461800" / "scf.in",
        "class": "normal",
        "source_model": "Foundation-OMAT canonical trajectory initial frame",
    },
    "bounded_multiT_462000_step464000": {
        "source": SOURCE / "bounded_multiT_462000_step464000.extxyz",
        "reference": SOURCE / "foundation_462000_initial.extxyz",
        "template": TEMPLATE_ROOT / "step_462000" / "scf.in",
        "class": "bounded_with_structural_shift",
        "source_model": "Multi-T canonical trajectory endpoint",
    },
    "runaway_foundation_461800_step462120": {
        "source": SOURCE / "runaway_foundation_461800_step462120.extxyz",
        "reference": SOURCE / "normal_461800_initial.extxyz",
        "template": TEMPLATE_ROOT / "step_461800" / "scf.in",
        "class": "runaway",
        "source_model": "Foundation-OMAT canonical first force-gate frame",
    },
    "runaway_scratch_461800_step462060": {
        "source": SOURCE / "runaway_scratch_461800_step462060.extxyz",
        "reference": SOURCE / "normal_461800_initial.extxyz",
        "template": TEMPLATE_ROOT / "step_461800" / "scf.in",
        "class": "runaway",
        "source_model": "from-scratch canonical first force-gate frame",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def map_local_to_full(template, reference):
    """Return full-trajectory indices and per-atom local-cell translations."""
    local_cell = np.asarray(template.cell.array)
    translations = np.array(
        [i * local_cell[0] + j * local_cell[1] for i in range(-5, 6) for j in range(-5, 6)]
    )
    indices = []
    shifts = []
    distances = []
    for symbol, target in zip(template.get_chemical_symbols(), template.positions):
        candidates = np.flatnonzero(np.array(reference.get_chemical_symbols()) == symbol)
        delta = reference.positions[candidates, None, :] + translations[None, :, :] - target
        norms = np.linalg.norm(delta, axis=2)
        flat = int(np.argmin(norms))
        atom_i, shift_i = np.unravel_index(flat, norms.shape)
        indices.append(int(candidates[atom_i]))
        shifts.append(translations[shift_i].tolist())
        distances.append(float(norms[atom_i, shift_i]))
    if max(distances) > 1e-6:
        raise RuntimeError(f"local-to-full mapping mismatch: max distance {max(distances):.6g} A")
    return indices, shifts, distances


def replace_positions(template_text: str, symbols, positions) -> str:
    lines = template_text.splitlines()
    marker = next(i for i, line in enumerate(lines) if line.strip().upper() == "ATOMIC_POSITIONS ANGSTROM")
    end = marker + 1
    while end < len(lines) and lines[end].strip() and not lines[end].lstrip().upper().startswith(("K_POINTS", "CELL_PARAMETERS", "ATOMIC_SPECIES")):
        end += 1
    atom_lines = [f"{s:<3s} {p[0]: .12f} {p[1]: .12f} {p[2]: .12f}" for s, p in zip(symbols, positions)]
    return "\n".join(lines[: marker + 1] + atom_lines + lines[end:]) + "\n"


def main() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ledger = {"package": "phase3_dft_recovery_20261004", "cases": {}}
    for name, cfg in CASES.items():
        source = read(cfg["source"])
        reference = read(cfg["reference"])
        template = read(cfg["template"], format="espresso-in")
        indices, shifts, distances = map_local_to_full(template, reference)
        positions = source.positions[indices] + np.asarray(shifts)
        out_dir = INPUT_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_input = out_dir / "scf.in"
        out_input.write_text(replace_positions(cfg["template"].read_text(), template.get_chemical_symbols(), positions))
        metadata = {
            "case": name,
            "class": cfg["class"],
            "source_model": cfg["source_model"],
            "source_frame": str(cfg["source"].relative_to(ROOT)),
            "reference_frame": str(cfg["reference"].relative_to(ROOT)),
            "template": str(cfg["template"].relative_to(ROOT.parent)),
            "source_sha256": sha256(cfg["source"]),
            "reference_sha256": sha256(cfg["reference"]),
            "template_sha256": sha256(cfg["template"]),
            "input_sha256": sha256(out_input),
            "nat": len(template),
            "symbols": template.get_chemical_symbols(),
            "full_atom_indices_zero_based": indices,
            "local_translation_vectors_angstrom": shifts,
            "mapping_max_error_angstrom": max(distances),
            "formal_dft_gate": False,
            "scope": "RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED",
        }
        (out_dir / "mapping.json").write_text(json.dumps(metadata, indent=2) + "\n")
        ledger["cases"][name] = metadata
    (INPUT_ROOT / "reconstruction_manifest.json").write_text(json.dumps(ledger, indent=2) + "\n")


if __name__ == "__main__":
    main()
