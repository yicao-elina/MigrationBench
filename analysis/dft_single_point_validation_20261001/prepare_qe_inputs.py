"""Prepare isolated non-magnetic QE single-point inputs for selected candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "candidate_manifest.csv"
OUT = ROOT / "qe_inputs"
PSEUDO_DIR = "/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/"
PSEUDOS = {
    "Sb": (121.76000, "Sb_PBE_FR.SG15v1.2.UPF"),
    "Te": (127.60000, "Te_PBE_FR.SG15v1.2.UPF"),
    "Cr": (51.99610, "Cr_PBE_FR.SG15v1.2.UPF"),
}


def qe_float(value: float) -> str:
    return f"{float(value): .10f}"


def make_input(row: pd.Series) -> str:
    atoms = read(row.input_path, format="extxyz")
    symbols = list(dict.fromkeys(atoms.get_chemical_symbols()))
    if set(symbols) != set(PSEUDOS):
        raise ValueError(f"{row.candidate_id}: unexpected species {symbols}")
    if len(atoms) != int(row.n_atoms):
        raise ValueError(f"{row.candidate_id}: nat mismatch")
    if not np.all(atoms.pbc):
        raise ValueError(f"{row.candidate_id}: non-periodic candidate")
    cell = np.asarray(atoms.cell.array, float)
    if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-6:
        raise ValueError(f"{row.candidate_id}: invalid cell")

    lines = [
        f"! Isolated DFT single-point validation; campaign=20261001; candidate={row.candidate_id}",
        f"! category={row.category}; endpoint_id={row.endpoint_id}; cov_id={row.cov_id}",
        f"! source_identity={row.source_identity}",
        f"! input_sha256={row.input_sha256}",
        "",
        "&CONTROL",
        "  calculation = 'scf'",
        "  restart_mode = 'from_scratch'",
        "  outdir = './out/'",
        f"  prefix = '{row.candidate_id}'",
        f"  pseudo_dir = '{PSEUDO_DIR}'",
        "  verbosity = 'high'",
        "  tprnfor = .true.",
        "  tstress = .true.",
        "  max_seconds = 171000",
        "/",
        "",
        "&SYSTEM",
        "  ibrav = 0",
        f"  nat = {len(atoms)}",
        "  ntyp = 3",
        "  ecutwfc = 70",
        "  ecutrho = 280",
        "  occupations = 'smearing'",
        "  smearing = 'cold'",
        "  degauss = 1.4699723600d-02",
        "  vdw_corr = 'DFT-D3'",
        "/",
        "",
        "&ELECTRONS",
        "  conv_thr = 1.0000000000d-06",
        "  electron_maxstep = 400",
        "  mixing_beta = 0.4",
        "/",
        "",
        "ATOMIC_SPECIES",
    ]
    for symbol in ["Sb", "Te", "Cr"]:
        mass, pseudo = PSEUDOS[symbol]
        lines.append(f"  {symbol:2s} {mass:10.5f}  {pseudo}")
    lines += [
        "",
        "K_POINTS AUTOMATIC",
        "2 2 2 1 1 1",
        "",
        "CELL_PARAMETERS angstrom",
    ]
    lines += ["  " + "  ".join(qe_float(v) for v in vec) for vec in cell]
    lines += ["", "ATOMIC_POSITIONS angstrom"]
    for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
        lines.append(f"{symbol:2s}  " + "  ".join(qe_float(v) for v in pos))
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST).fillna("")
    selected = df[df.dft_status.eq("selected_wave1") & df.input_path.ne("")].copy()
    if len(selected) != 20:
        raise RuntimeError(f"expected 20 selected inputs, found {len(selected)}")
    records = []
    for _, row in selected.iterrows():
        path = OUT / f"{row.candidate_id}.scf.in"
        path.write_text(make_input(row))
        records.append({"candidate_id": row.candidate_id, "qe_input": str(path), "category": row.category})
    (OUT / "selected_input_index.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {len(records)} QE inputs to {OUT}")


if __name__ == "__main__":
    main()
