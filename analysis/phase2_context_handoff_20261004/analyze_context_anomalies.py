#!/usr/bin/env python3
"""Extract local anomaly windows for the additive context snapshots."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
from ase.io import iread
from ase.neighborlist import neighbor_list


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    (460000, "foundation-omat"): {"first_gate_step": 460180, "source_boundary": False},
    (462200, "foundation-omat"): {"first_gate_step": 462200, "source_boundary": True},
    (462200, "from-scratch"): {"first_gate_step": 462200, "source_boundary": True},
    (462200, "naive-fine-tuning"): {"first_gate_step": 462200, "source_boundary": True},
    (462200, "multi_T-fine-tuning"): {"first_gate_step": 462200, "source_boundary": True},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_record(atoms, step: int) -> dict:
    forces = np.asarray(atoms.get_forces(), dtype=float)
    force_norm = np.linalg.norm(forces, axis=1)
    velocities = atoms.get_momenta() / atoms.get_masses()[:, None] * 1000.0
    velocity_norm = np.linalg.norm(velocities, axis=1)
    force_idx = int(np.argmax(force_norm))
    velocity_idx = int(np.argmax(velocity_norm))
    i, j, d = neighbor_list("ijd", atoms, 6.0, self_interaction=False)
    neighbors = []
    for jj, dd in zip(j[i == force_idx], d[i == force_idx]):
        neighbors.append((float(dd), int(jj), atoms[int(jj)].symbol))
    for jj, dd in zip(i[j == force_idx], d[j == force_idx]):
        neighbors.append((float(dd), int(jj), atoms[int(jj)].symbol))
    local = [item for item in neighbors if item[0] <= 3.6]
    return {
        "step": step,
        "max_force_eV_A": float(force_norm.max()),
        "max_force_atom_index": force_idx,
        "max_force_atom_symbol": atoms[force_idx].symbol,
        "max_force_atom_force_vector_eV_A": forces[force_idx].tolist(),
        "max_velocity_A_ps": float(velocity_norm.max()),
        "max_velocity_atom_index": velocity_idx,
        "max_velocity_atom_symbol": atoms[velocity_idx].symbol,
        "min_distance_A": float(np.min(d)) if len(d) else None,
        "local_neighbor_count_within_3p6A": len(local),
        "local_neighbor_species_within_3p6A": {
            symbol: sum(item[2] == symbol for item in local) for symbol in ("Cr", "Sb", "Te")
        },
        "neighbors_of_max_force_atom": sorted(neighbors)[:12],
    }


def main() -> None:
    out = []
    for (snapshot, model), metadata in EXPECTED.items():
        path = ROOT / "anomaly_frames" / f"snapshot_{snapshot}" / f"model_{model}" / "window.extxyz"
        if not path.exists():
            out.append({"snapshot": snapshot, "model": model, "status": "MISSING_WINDOW", **metadata})
            continue
        frames = []
        seen = set()
        for atoms in iread(path, index=":"):
            step = int(atoms.info.get("absolute_step", atoms.info.get("step", -1)))
            if step in seen:
                continue
            seen.add(step)
            frames.append(frame_record(atoms, step))
        out.append({"snapshot": snapshot, "model": model, "status": "EXTRACTED", **metadata,
                    "window_sha256": sha256(path), "frames": frames})
    (ROOT / "context_anomaly_local_neighborhood.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for row in out:
        if row["status"] != "EXTRACTED":
            print(row["snapshot"], row["model"], row["status"])
            continue
        peak = max(row["frames"], key=lambda item: item["max_force_eV_A"])
        print(row["snapshot"], row["model"], row["status"],
              "steps=" + ",".join(str(frame["step"]) for frame in row["frames"]),
              "peak_atom=" + str(peak["max_force_atom_index"]),
              "peak_symbol=" + peak["max_force_atom_symbol"],
              "peak_force=" + f"{peak['max_force_eV_A']:.6f}")


if __name__ == "__main__":
    main()
