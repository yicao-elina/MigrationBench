#!/usr/bin/env python3
"""Export only final endpoint coordinates from remote QE outputs.

Raw inputs/outputs remain on Rockfish. The generated XYZ is a compact processed
artifact with explicit provenance and is suitable for a later MACE run.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "collaborator_dft_neb"
OUT = DATA / "processed_endpoints.extxyz"


def fetch(path: str) -> str:
    remote = "awk '/Begin final coordinates/{p=1;next} /End final coordinates/{p=0} p && NF>=4{print}' " + path
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "rockfish", remote],
        text=True, capture_output=True, check=True,
    ).stdout


def fetch_cell(path: str) -> list[str]:
    remote = "awk '/CELL_PARAMETERS/{p=3;next} p>0{print; p--}' " + path
    text = subprocess.run(["ssh", "-o", "BatchMode=yes", "rockfish", remote],
                          text=True, capture_output=True, check=True).stdout
    rows = [line.split() for line in text.splitlines() if len(line.split()) >= 3]
    if len(rows) != 3:
        raise RuntimeError(f"{path}: expected 3 cell rows, got {len(rows)}")
    return [" ".join(r[:3]) for r in rows]


def main() -> None:
    rows = list(csv.DictReader((DATA / "endpoint_provenance.csv").open(newline="")))
    identities = {r["endpoint_id"]: r for r in csv.DictReader((DATA / "system_identity.csv").open(newline=""))}
    forces = {r["endpoint_id"]: r for r in csv.DictReader((DATA / "force_vector_audit.csv").open(newline=""))}
    blocks: list[str] = []
    for row in rows:
        eid = row["endpoint_id"]
        coords = [line.split() for line in fetch(row["raw_output_path"]).splitlines()]
        identity = identities[eid]
        if len(coords) != int(identity["natoms"]):
            raise RuntimeError(f"{eid}: expected {identity['natoms']} coordinates, got {len(coords)}")
        force = forces[eid]
        cell = fetch_cell(row["raw_input_path"])
        comment = (
            f"endpoint_id={eid} system={identity['system_label']} formula={identity['formula']} "
            f"site={eid.split('-')[-1]} energy_ry={row['final_energy_ry']} "
            f"max_force_ev_ang={force['approx_max_force_ev_ang']} "
            f"input_sha256={row['input_sha256']} output_sha256={row['output_sha256']} "
            f"Lattice=\"{' '.join(' '.join(c.split()) for c in cell)}\" "
            "pbc=\"T T T\" source=rockfish_raw_remote"
        )
        blocks.append(str(len(coords)) + "\n" + comment + "\n" +
                      "\n".join(" ".join(c[:4]) for c in coords))
    OUT.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    print(f"wrote {OUT} with {len(rows)} endpoint structures")


if __name__ == "__main__":
    main()
