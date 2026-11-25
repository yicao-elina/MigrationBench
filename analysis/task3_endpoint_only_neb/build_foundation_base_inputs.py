#!/usr/bin/env python3
"""Build a fresh endpoint-only linear base-path matrix for all local models."""
from pathlib import Path
import hashlib
import json
import numpy as np
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent
INPUTS = Path("/tmp/migrationbench_remote_endpoints_20261002")
OUT = Path("/tmp/migrationbench_foundation_base_linear_remote_20261002")
PATHS = [
    "2L-Cu__neb-QLs", "2L-Cu__neb-gap-diffuse", "2L-Cu__neb-gap2QL",
    "2L-Li__neb-QLs", "2L-Li__neb-gap-diffuse", "2L-Li__neb-gap2Ql",
    "2L-Na__neb-gap-diffuse", "2L-Na__neb-gap2QL",
]

def sha256(p):
    h = hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def read_qe_endpoint(path):
    lines = path.read_text().splitlines()
    nat = next(int(x.split('=')[1].split()[0]) for x in lines if x.strip().startswith('nat'))
    pos_i = next(i for i, x in enumerate(lines) if x.strip().startswith('ATOMIC_POSITIONS'))
    cell_i = next(i for i, x in enumerate(lines) if x.strip().startswith('CELL_PARAMETERS'))
    symbols=[]; positions=[]
    for line in lines[pos_i+1:pos_i+1+nat]:
        toks=line.replace('!',' ').split()
        symbols.append(toks[0]); positions.append([float(toks[1]),float(toks[2]),float(toks[3])])
    cell=np.array([[float(v) for v in lines[cell_i+j].split()[:3]] for j in range(1,4)])
    return Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for path_id in PATHS:
        initial = read_qe_endpoint(INPUTS / path_id / "initial.in")
        final = read_qe_endpoint(INPUTS / path_id / "final.in")
        delta = final.positions - initial.positions
        if initial.pbc.any():
            frac_delta = delta @ np.linalg.inv(initial.cell.array)
            frac_delta -= np.rint(frac_delta)
            delta = frac_delta @ initial.cell.array
        frames=[]
        for i, alpha in enumerate(np.linspace(0, 1, 10)):
            atoms = initial.copy()
            atoms.positions = initial.positions + alpha * delta
            atoms.info.update({"path_id": path_id, "variant": "base", "image_index": i+1, "reaction_coordinate": float(alpha), "source": "endpoint_only_linear"})
            frames.append(atoms)
        out = OUT / f"{path_id}__base.extxyz"
        write(out, frames, format="extxyz")
        rows.append({"path_id": path_id, "n_images": 10, "input_sha256": sha256(out), "initial_sha256": sha256(INPUTS / path_id / "initial.in"), "final_sha256": sha256(INPUTS / path_id / "final.in")})
    (OUT / "manifest.json").write_text(json.dumps({"status":"endpoint_only_linear_base","n_paths":len(rows),"n_frames":len(rows)*10,"paths":rows}, indent=2)+"\n")
    print(json.dumps({"output":str(OUT),"n_paths":len(rows),"n_frames":len(rows)*10}, indent=2))

if __name__ == "__main__": main()
