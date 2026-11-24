#!/usr/bin/env python3
"""Prepare isolated QE single-point inputs for one frozen DFT wave."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WAVE = ROOT / "wave_01_manifest.csv"
OUT = ROOT / "dft_wave_campaign_20261013"
INPUTS = OUT / "qe_inputs"
PSEUDO_DIR = "/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/"
PSEUDOS = {
    "Cr": (51.99610, "Cr_PBE_FR.SG15v1.2.UPF"),
    "Sb": (121.76000, "Sb_PBE_FR.SG15v1.2.UPF"),
    "Te": (127.60000, "Te_PBE_FR.SG15v1.2.UPF"),
}
EXPECTED_COUNTS = {"Cr": 50, "Sb": 800, "Te": 1200}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def qe_float(value: float) -> str:
    return f"{float(value): .10f}"


def read_extxyz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, list[bool]]:
    """Read the minimal extxyz fields needed for a periodic QE input."""
    with path.open(encoding="utf-8") as handle:
        nat = int(handle.readline().strip())
        header = handle.readline().rstrip("\n")
        match = re.search(r'Lattice="([^"]+)"', header)
        if not match:
            raise ValueError(f"{path}: missing Lattice field")
        cell = np.asarray([float(value) for value in match.group(1).split()], dtype=float).reshape(3, 3)
        symbols = []
        positions = []
        for _ in range(nat):
            fields = handle.readline().split()
            if len(fields) < 4:
                raise ValueError(f"{path}: malformed atom row")
            symbols.append(fields[0])
            positions.append([float(value) for value in fields[1:4]])
    pbc_match = re.search(r'pbc="([^"]+)"', header)
    pbc = [value.upper() in {"T", "TRUE", ".TRUE."} for value in pbc_match.group(1).split()] if pbc_match else [True, True, True]
    return symbols, np.asarray(positions, dtype=float), cell, pbc


def make_input(row: pd.Series) -> str:
    source = Path(row.local_extxyz)
    symbols, positions, cell, pbc = read_extxyz(source)
    counts = {symbol: symbols.count(symbol) for symbol in sorted(set(symbols))}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"{row.candidate_id}: unexpected species counts {counts}")
    if len(symbols) != 2050:
        raise ValueError(f"{row.candidate_id}: natoms={len(symbols)} != 2050")
    if not np.all(np.asarray(pbc, dtype=bool)):
        raise ValueError(f"{row.candidate_id}: candidate is not fully periodic")
    if not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-6:
        raise ValueError(f"{row.candidate_id}: invalid cell")
    if not np.isfinite(positions).all():
        raise ValueError(f"{row.candidate_id}: nonfinite positions")

    lines = [
        f"! Isolated DFT single-point validation; campaign=dft_wave_campaign_20261013; candidate={row.candidate_id}",
        f"! source={row.source}; training_seed={int(row.training_seed)}; run_key={row.run_key}",
        f"! window={row.window}; selection_role={row.selection_role}; time_ps={row.time_ps}",
        f"! model_sha256={row.model_sha256}",
        f"! source_snapshot={row.source_snapshot}",
        f"! prepared_extxyz={row.remote_extxyz}",
        f"! prepared_extxyz_sha256={row.local_extxyz_sha256}",
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
        "  nat = 2050",
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
    for symbol in ["Cr", "Sb", "Te"]:
        mass, pseudo = PSEUDOS[symbol]
        lines.append(f"  {symbol:2s} {mass:10.5f}  {pseudo}")
    lines += [
        "",
        "K_POINTS AUTOMATIC",
        "2 2 2 1 1 1",
        "",
        "CELL_PARAMETERS angstrom",
    ]
    lines += ["  " + "  ".join(qe_float(value) for value in vector) for vector in cell]
    lines += ["", "ATOMIC_POSITIONS angstrom"]
    for symbol, position in zip(symbols, positions):
        lines.append(f"{symbol:2s}  " + "  ".join(qe_float(value) for value in position))
    return "\n".join(lines) + "\n"


def main() -> None:
    wave = pd.read_csv(WAVE).fillna("")
    if len(wave) != 20:
        raise RuntimeError(f"expected 20 wave-1 candidates, found {len(wave)}")
    if wave.candidate_id.duplicated().any():
        raise RuntimeError("duplicate candidate_id in wave manifest")
    INPUTS.mkdir(parents=True, exist_ok=True)
    records = []
    for _, row in wave.iterrows():
        path = INPUTS / f"{row.candidate_id}.scf.in"
        path.write_text(make_input(row), encoding="utf-8")
        records.append({
            "candidate_id": row.candidate_id,
            "qe_input": str(path),
            "input_sha256": sha256(path),
            "source": row.source,
            "training_seed": int(row.training_seed),
            "run_key": row.run_key,
            "window": row.window,
            "selection_role": row.selection_role,
            "time_ps": float(row.time_ps),
            "remote_extxyz": row.remote_extxyz,
            "prepared_extxyz_sha256": row.local_extxyz_sha256,
            "model_sha256": row.model_sha256,
        })
    input_hashes = {record["candidate_id"]: record["input_sha256"] for record in records}
    wave["input_sha256"] = wave["candidate_id"].map(input_hashes)
    (OUT / "selected_input_index.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    wave.to_csv(OUT / "wave_01_manifest.csv", index=False)
    metadata = {
        "campaign_id": "dft_wave_campaign_20261013",
        "local_root": str(OUT),
        "remote_root": "/scratch16/pclancy3/yi/dft_wave_campaign_20261013",
        "scope": "20 independent QE 7.2 non-magnetic SCF single points for failure-attribution mechanism testing",
        "candidate_inventory": 639,
        "wave_size": 20,
        "selection_manifest": str(WAVE),
        "method": {
            "qe_version": "7.2",
            "calculation": "scf",
            "spin": "QE default nspin=1; magnetic keywords absent",
            "pseudo_dir": PSEUDO_DIR,
            "pseudopotentials": {name: "present_on_rockfish" for _, name in PSEUDOS.values()},
            "ecutwfc_Ry": 70,
            "ecutrho_Ry": 280,
            "kpoints": "2 2 2 1 1 1",
            "smearing": "cold",
            "degauss_Ry": 0.0146997236,
            "vdw_corr": "DFT-D3",
        },
        "quality_gate": [
            "input hash matches selected manifest",
            "explicit QE JOB DONE",
            "finite total energy",
            "finite total force",
            "QE-reported nat=2050",
            "no QE error markers",
            "same non-magnetic SCF contract retained",
        ],
        "submission_status": "prepared_not_submitted",
    }
    (OUT / "campaign_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"campaign": str(OUT), "inputs": len(records), "sha256": [r["input_sha256"] for r in records]}, indent=2))


if __name__ == "__main__":
    main()
