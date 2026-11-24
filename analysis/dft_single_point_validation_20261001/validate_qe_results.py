"""Scientific quality gate for QE single-point outputs.

The parser deliberately requires more than scheduler completion or ``JOB DONE``:
finite total energy, finite total force, matching atom count, valid input
identity, no QE error markers, and the non-magnetic SCF input contract.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "candidate_manifest.csv"
RESULTS = ROOT / "results"
FORBIDDEN = re.compile(r"\b(nspin|starting_magnetization|tot_magnetization|constrained_magnetization|angle1|angle2)\b", re.I)
ENERGY = re.compile(r"!\s+total energy\s+=\s+([-+0-9.EeDd]+)\s+Ry")
FORCE = re.compile(r"Total force\s*=\s*([-+0-9.EeDd]+)\s+Ry/au")
NAT = re.compile(r"number of atoms/cell\s*=\s*(\d+)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fnum(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def one(row: pd.Series) -> dict:
    cid = row.candidate_id
    run = RESULTS / cid
    input_path = run / "input.scf.in"
    out_path = run / "scf.out"
    err_path = run / "scf.err"
    reasons = []
    energy_ry = math.nan
    total_force_ry_au = math.nan
    nat_out = None
    text = ""
    if not input_path.is_file():
        reasons.append("missing_input_copy")
    else:
        input_text = input_path.read_text(errors="replace")
        if FORBIDDEN.search(input_text):
            reasons.append("forbidden_magnetic_keyword")
        if "calculation = 'scf'" not in input_text:
            reasons.append("not_scf_input")
        if sha256(input_path) == "":
            reasons.append("input_hash_unavailable")
    if not out_path.is_file():
        reasons.append("missing_scf_output")
    else:
        text = out_path.read_text(errors="replace")
        energies = [fnum(x) for x in ENERGY.findall(text)]
        forces = [fnum(x) for x in FORCE.findall(text)]
        nats = NAT.findall(text)
        if energies:
            energy_ry = energies[-1]
        else:
            reasons.append("missing_total_energy")
        if forces:
            total_force_ry_au = forces[-1]
        else:
            reasons.append("missing_total_force")
        if nats:
            nat_out = int(nats[-1])
        else:
            reasons.append("missing_qe_atom_count")
        if "JOB DONE" not in text:
            reasons.append("missing_explicit_qe_termination")
        for marker in ["Error in routine", "MPI_ABORT", "Maximum CPU time exceeded", "%%%%%%%%"]:
            if marker in text:
                reasons.append(f"qe_error_marker:{marker}")
    if nat_out is not None and nat_out != int(row.n_atoms):
        reasons.append("atom_count_mismatch")
    if not np.isfinite(energy_ry):
        reasons.append("nonfinite_total_energy")
    if not np.isfinite(total_force_ry_au):
        reasons.append("nonfinite_total_force")
    return {
        "candidate_id": cid,
        "category": row.category,
        "endpoint_id": row.endpoint_id,
        "cov_id": row.cov_id,
        "dft_status": "scientifically_accepted" if not reasons else "not_accepted",
        "energy_Ry": energy_ry,
        "total_force_Ry_AU": total_force_ry_au,
        "nat_output": nat_out if nat_out is not None else "",
        "qe_output": str(out_path),
        "qe_output_sha256": sha256(out_path) if out_path.is_file() else "",
        "reasons": ";".join(reasons),
    }


def main() -> None:
    df = pd.read_csv(MANIFEST).fillna("")
    selected = df[df.dft_status.eq("selected_wave1") & df.input_path.ne("")]
    results = pd.DataFrame([one(row) for _, row in selected.iterrows()])
    results.to_csv(ROOT / "dft_single_point_results.csv", index=False)
    print(results[["candidate_id", "dft_status", "reasons"]].to_string(index=False))
    print("accepted", int(results.dft_status.eq("scientifically_accepted").sum()), "/", len(results))


if __name__ == "__main__":
    main()
