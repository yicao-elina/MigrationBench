#!/usr/bin/env python3
"""Extract exact MD frames and record identity/provenance diagnostics.

The source trajectory is read frame-by-frame. Exact step metadata, not the
filename or frame number, selects the handoff snapshots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
TARGET_STEPS = (460000, 461800, 462000, 462200)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_array(x) -> bool:
    try:
        return bool(np.isfinite(np.asarray(x, dtype=float)).all())
    except Exception:
        return False


def geometry_diagnostics(positions, cell, symbols, momenta, forces):
    frac = np.asarray(positions) @ np.linalg.inv(np.asarray(cell))
    df = frac[:, None, :] - frac[None, :, :]
    df -= np.rint(df)
    d = np.linalg.norm(df @ np.asarray(cell), axis=2)
    np.fill_diagonal(d, np.inf)
    symbols = np.asarray(symbols)
    cr = np.flatnonzero(symbols == "Cr")
    cr_te = d[np.ix_(cr, np.flatnonzero(symbols == "Te"))]
    return {
        "min_pair_distance_A": float(np.nanmin(d)),
        "min_Cr_Te_distance_A": float(np.nanmin(cr_te)) if cr_te.size else None,
        "cell_volume_A3": float(abs(np.linalg.det(cell))),
        "position_finite": finite_array(positions),
        "momenta_present": momenta is not None,
        "momenta_finite": finite_array(momenta) if momenta is not None else False,
        "forces_present_in_source": forces is not None,
        "forces_finite_in_source": finite_array(forces) if forces is not None else False,
    }


def parse_lattice(header: str):
    m = re.search(r'Lattice="([^"]+)"', header)
    if not m:
        raise ValueError("missing Lattice metadata")
    values = np.fromstring(m.group(1), sep=" ")
    if values.size != 9:
        raise ValueError(f"expected 9 lattice values, got {values.size}")
    return values.reshape(3, 3)


def parse_atom_line(line: str):
    p = line.split()
    if len(p) < 4:
        raise ValueError(f"short atom line: {line!r}")
    return p[0], np.asarray([float(p[1]), float(p[2]), float(p[3])], dtype=float)


def property_columns(header: str):
    match = re.search(r"Properties=([^ ]+)", header)
    if not match:
        return {}
    tokens = match.group(1).split(":")
    cols = {}
    i = 0
    offset = 0
    while i + 2 < len(tokens):
        name, kind, count = tokens[i], tokens[i + 1], int(tokens[i + 2])
        cols[name] = (offset, count)
        offset += count
        i += 3
    return cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--label", default="from-scratch-reference")
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    wanted = set(TARGET_STEPS)
    found = {}
    species_ref = None
    cell_ref = None
    pbc_ref = None
    frame_count = 0
    bad_step_metadata = 0
    source_sha = sha256(source)

    with source.open("r", errors="strict") as stream:
      frame_index = 0
      while True:
        first = stream.readline()
        if not first:
            break
        if not first.strip():
            continue
        natoms = int(first.strip())
        header = stream.readline().rstrip("\n")
        frame_count += 1
        step_match = re.search(r"(?:^| )step=([-+]?\d+)(?: |$)", header)
        step_raw = step_match.group(1) if step_match else None
        step = int(step_raw) if step_raw is not None else None
        if step not in wanted:
            for _ in range(natoms):
                stream.readline()
            if step is None:
                bad_step_metadata += 1
            frame_index += 1
            continue
        lines = [stream.readline() for _ in range(natoms)]
        parsed = [parse_atom_line(line) for line in lines]
        symbols = tuple(x[0] for x in parsed)
        positions = np.asarray([x[1] for x in parsed], dtype=float)
        columns = property_columns(header)
        momenta = None
        forces = None
        if "momenta" in columns:
            off, count = columns["momenta"]
            momenta = np.asarray([[float(x) for x in line.split()[off:off + count]] for line in lines], dtype=float)
        if "forces" in columns:
            off, count = columns["forces"]
            forces = np.asarray([[float(x) for x in line.split()[off:off + count]] for line in lines], dtype=float)
        cell = parse_lattice(header)
        pbc_match = re.search(r'pbc="([^"]+)"', header)
        pbc = tuple(x.upper() in {"T", "TRUE", "1"} for x in pbc_match.group(1).split()) if pbc_match else (True, True, True)
        species_ref = symbols if species_ref is None else species_ref
        cell_ref = cell if cell_ref is None else cell_ref
        pbc_ref = pbc if pbc_ref is None else pbc_ref
        if symbols != species_ref:
            raise ValueError(f"species identity changed at frame {frame_index}")
        if step in found:
            frame_index += 1
            continue
        if natoms != 2050:
            raise ValueError(f"unexpected atom count at step {step}: {natoms}")
        if not all(pbc):
            raise ValueError(f"non-periodic frame at step {step}: {pbc}")
        if not finite_array(positions):
            raise ValueError(f"non-finite positions at step {step}")
        target = out / f"snapshot_step_{step:06d}.extxyz"
        target_header = header + f' phase1_source_path="{source}" phase1_source_sha256={source_sha} phase1_source_frame_index={frame_index} phase1_selected_step={step}'
        target.write_text(f"{natoms}\n{target_header}\n" + "".join(lines))
        found[step] = {
            "step": step,
            "source_frame_index": frame_index,
            "output_path": str(target),
            "output_sha256": sha256(target),
            "natoms": natoms,
            "species_counts": {s: int(symbols.count(s)) for s in sorted(set(symbols))},
            "species_sha256": hashlib.sha256("\n".join(symbols).encode()).hexdigest(),
            "cell_A": cell.tolist(),
            "pbc": list(pbc),
            "geometry": geometry_diagnostics(positions, cell, symbols, momenta, forces),
            "energy_eV_in_source": (re.search(r"(?:^| )energy=([^ ]+)", header).group(1) if re.search(r"(?:^| )energy=([^ ]+)", header) else None),
            "stress_in_source": (re.search(r'stress="([^"]+)"', header).group(1) if re.search(r'stress="([^"]+)"', header) else None),
        }
        frame_index += 1

    if set(found) != wanted:
        missing = sorted(wanted - set(found))
        raise RuntimeError(f"missing exact target steps: {missing}; scanned {frame_count} frames")

    species = list(species_ref)
    key_indices = [i + 1 for i, s in enumerate(species) if s == "Cr"]
    manifest = {
        "schema_version": "phase1-static-force-snapshot-1.0",
        "label": args.label,
        "source_path": str(source),
        "source_sha256": source_sha,
        "source_size_bytes": source.stat().st_size,
        "source_mtime_epoch": source.stat().st_mtime,
        "scan": {"frames_scanned": frame_count, "frames_without_integer_step": bad_step_metadata},
        "target_steps": list(TARGET_STEPS),
        "selected_steps": {str(k): found[k] for k in sorted(found)},
        "identity": {
            "natoms": len(species),
            "species_counts": {s: int(species.count(s)) for s in sorted(set(species))},
            "species_sha256": hashlib.sha256("\n".join(species).encode()).hexdigest(),
            "key_species": "Cr",
            "key_atom_indices_one_based": key_indices,
            "key_atom_count": len(key_indices),
            "cell_A_reference": np.asarray(cell_ref).tolist(),
            "cell_equal_across_selected": all(np.allclose(found[k]["cell_A"], cell_ref, atol=1e-10, rtol=0) for k in found),
            "pbc_reference": list(pbc_ref),
            "pbc_equal_across_selected": all(found[k]["pbc"] == list(pbc_ref) for k in found),
        },
        "unit_convention": {
            "positions": "Angstrom (extxyz/ASE)",
            "momenta": "ASE atomic units as stored by source trajectory; retained for provenance only",
            "source_forces": "eV/Angstrom if present; not used as the static evaluation result",
            "model_energy": "eV",
            "model_forces": "eV/Angstrom",
            "model_stress": "eV/Angstrom^3, with GPa conversion in analysis",
        },
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    (out / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"source": str(source), "frames_scanned": frame_count, "found": sorted(found), "manifest": str(out / 'snapshot_manifest.json')}, indent=2))


if __name__ == "__main__":
    main()
