#!/usr/bin/env python3
"""Build a within-method DFT-vs-MACE force/energy-change comparison.

The script refuses to write a comparison until all four QE cases pass the
diagnostic gate. Energies are reported only as changes relative to the normal
461800 case within each method; no cross-model absolute energy ranking is
constructed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
GATE = ROOT / "results_reconstructed" / "qe_gate_audit.json"
MACE = ROOT / "mace_results" / "mace_reconstructed.json"
OUT_CSV = ROOT / "results_reconstructed" / "dft_vs_mace_within_method.csv"
OUT_MD = ROOT / "results_reconstructed" / "DFT_VS_MACE.md"
RY_TO_EV = 13.605693009
NORMAL = "normal_461800_initial"


def main() -> None:
    gate = json.loads(GATE.read_text())
    accepted = {row["case"]: row for row in gate["rows"] if row.get("status") == "DIAGNOSTIC_ACCEPTED"}
    expected = {"normal_461800_initial", "bounded_multiT_462000_step464000",
                "runaway_foundation_461800_step462120", "runaway_scratch_461800_step462060"}
    missing = sorted(expected - set(accepted))
    if missing:
        raise SystemExit(f"DFT gate incomplete; refusing comparison for: {', '.join(missing)}")
    mace_rows = json.loads(MACE.read_text())["rows"]
    normal_mace = {row["model"]: row for row in mace_rows if row["case"] == NORMAL}
    normal_dft = accepted[NORMAL]
    dft_normal_force = np.asarray(normal_dft["force_vectors_eV_A"], dtype=float)
    dft_normal_energy_eV = float(normal_dft["total_energy_ry"]) * RY_TO_EV
    out = []
    for row in mace_rows:
        dft = accepted[row["case"]]
        dft_forces = np.asarray(dft["force_vectors_eV_A"], dtype=float)
        mace_forces = np.asarray(row["forces_eV_A"], dtype=float)
        if dft_forces.shape != mace_forces.shape:
            raise SystemExit(f"force shape mismatch for {row['case']}/{row['model']}: "
                             f"DFT={dft_forces.shape} MACE={mace_forces.shape}")
        delta_dft_eV = float(dft["total_energy_ry"]) * RY_TO_EV - dft_normal_energy_eV
        delta_mace_eV = float(row["energy_eV"]) - float(normal_mace[row["model"]]["energy_eV"])
        diff = dft_forces - mace_forces
        out.append({
            "case": row["case"], "model": row["model"],
            "dft_delta_energy_eV_within_dft": delta_dft_eV,
            "mace_delta_energy_eV_within_model": delta_mace_eV,
            "dft_max_force_eV_A": float(np.linalg.norm(dft_forces, axis=1).max()),
            "mace_max_force_eV_A": float(row["max_force_eV_A"]),
            "force_mae_eV_A": float(np.linalg.norm(diff, axis=1).mean()),
            "force_rmse_eV_A": float(np.sqrt(np.mean(diff * diff))),
            "force_max_abs_component_eV_A": float(np.abs(diff).max()),
            "dft_input_sha256": dft["input_sha256"], "mace_input_sha256": row["input_sha256"],
            "model_sha256": row["model_sha256"],
            "scope": "RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED",
        })
    fields = list(out[0])
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(out)
    lines = [
        "# DFT-vs-MACE within-method comparison", "",
        "All four QE cases passed the diagnostic gate. Energy columns are changes"
        " relative to `normal_461800_initial` within DFT or within the same MACE"
        " model; absolute cross-model energy ranking is not interpreted.", "",
        "| Case | Model | ΔE DFT (eV) | ΔE MACE (eV) | DFT max | MACE max | Force MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in out:
        lines.append(f"| {row['case']} | {row['model']} | {row['dft_delta_energy_eV_within_dft']:.4f} | "
                     f"{row['mace_delta_energy_eV_within_model']:.4f} | {row['dft_max_force_eV_A']:.4f} | "
                     f"{row['mace_max_force_eV_A']:.4f} | {row['force_mae_eV_A']:.4f} |")
    lines += ["", "Scope: `RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED`; these 81-atom local cells are not formal full-cell DFT validation."]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_CSV} and {OUT_MD} ({len(out)} model/case rows)")


if __name__ == "__main__":
    main()
