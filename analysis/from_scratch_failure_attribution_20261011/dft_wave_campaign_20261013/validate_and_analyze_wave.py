#!/usr/bin/env python3
"""Validate QE wave-1 outputs and compare DFT forces with MACE forces."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
WAVE = ROOT / "wave_01_manifest.csv"
RESULTS = ROOT / "results"
RY_PER_BOHR_TO_EV_PER_A = 25.71104309541616
RY_TO_EV = 13.605693122994
ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-+0-9.EeDd]+)\s+Ry")
TOTAL_FORCE_RE = re.compile(r"Total force\s*=\s*([-+0-9.EeDd]+)\s+Ry/au")
NAT_RE = re.compile(r"number of atoms/cell\s*=\s*(\d+)")
FORCE_BLOCK_RE = re.compile(r"Forces acting on atoms \(Ry/au\)\s*\n(?P<body>.*?)(?=\n\s*Total force\s*=|\Z)", re.S)
FORCE_LINE_RE = re.compile(
    r"^\s*atom\s+\d+\s+type\s+\d+\s+force\s*=\s*"
    r"([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fnum(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def read_extxyz(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        nat = int(handle.readline().strip())
        header = handle.readline()
        if "Properties=species:S:1:pos:R:3:momenta:R:3:forces:R:3" not in header:
            raise ValueError(f"{path}: unexpected properties contract")
        symbols = []
        forces = []
        for _ in range(nat):
            fields = handle.readline().split()
            if len(fields) < 10:
                raise ValueError(f"{path}: malformed force row")
            symbols.append(fields[0])
            forces.append([float(value) for value in fields[7:10]])
    return symbols, np.asarray(forces, dtype=float)


def parse_qe_output(path: Path) -> dict:
    if not path.is_file():
        return {"qe_status": "not_accepted", "reasons": "missing_scf_output"}
    text = path.read_text(errors="replace")
    reasons = []
    energies = [fnum(value) for value in ENERGY_RE.findall(text)]
    total_forces = [fnum(value) for value in TOTAL_FORCE_RE.findall(text)]
    nats = [int(value) for value in NAT_RE.findall(text)]
    blocks = FORCE_BLOCK_RE.findall(text)
    force_vectors = []
    if blocks:
        for line in blocks[-1].splitlines():
            match = FORCE_LINE_RE.match(line)
            if match:
                force_vectors.append([fnum(match.group(i)) for i in range(1, 4)])
    energy_ry = energies[-1] if energies else math.nan
    total_force_ry_au = total_forces[-1] if total_forces else math.nan
    nat_out = nats[-1] if nats else None
    if not energies:
        reasons.append("missing_total_energy")
    if not total_forces:
        reasons.append("missing_total_force")
    if not nats:
        reasons.append("missing_qe_atom_count")
    if len(force_vectors) != 2050:
        reasons.append(f"missing_or_incomplete_atom_forces:{len(force_vectors)}")
    if "JOB DONE" not in text:
        reasons.append("missing_explicit_qe_termination")
    for marker in ["Error in routine", "MPI_ABORT", "Maximum CPU time exceeded", "%%%%%%%%"]:
        if marker in text:
            reasons.append(f"qe_error_marker:{marker}")
    if nat_out is not None and nat_out != 2050:
        reasons.append("atom_count_mismatch")
    if not np.isfinite(energy_ry):
        reasons.append("nonfinite_total_energy")
    if not np.isfinite(total_force_ry_au):
        reasons.append("nonfinite_total_force")
    return {
        "qe_status": "scientifically_accepted" if not reasons else "not_accepted",
        "reasons": ";".join(reasons),
        "energy_Ry": energy_ry,
        "energy_eV": energy_ry * RY_TO_EV if np.isfinite(energy_ry) else math.nan,
        "total_force_Ry_AU": total_force_ry_au,
        "total_force_eV_A": total_force_ry_au * RY_PER_BOHR_TO_EV_PER_A if np.isfinite(total_force_ry_au) else math.nan,
        "nat_output": nat_out if nat_out is not None else "",
        "qe_force_vectors": np.asarray(force_vectors, dtype=float) * RY_PER_BOHR_TO_EV_PER_A if force_vectors else None,
        "qe_output_sha256": sha256(path),
    }


def compare_forces(mace: np.ndarray, dft: np.ndarray) -> dict:
    mace_norm = np.linalg.norm(mace, axis=1)
    dft_norm = np.linalg.norm(dft, axis=1)
    delta = dft - mace
    delta_norm = np.linalg.norm(delta, axis=1)
    denom = max(float(np.sqrt(np.mean(mace**2))), 1e-12)
    cosine = float(np.sum(mace * dft) / max(np.linalg.norm(mace) * np.linalg.norm(dft), 1e-12))
    top = int(np.argmax(delta_norm))
    return {
        "mace_force_max_eV_A": float(mace_norm.max()),
        "mace_force_rms_eV_A": float(np.sqrt(np.mean(mace_norm**2))),
        "dft_force_max_eV_A": float(dft_norm.max()),
        "dft_force_rms_eV_A": float(np.sqrt(np.mean(dft_norm**2))),
        "force_delta_mae_eV_A": float(np.mean(np.abs(delta))),
        "force_delta_rmse_eV_A": float(np.sqrt(np.mean(delta**2))),
        "force_delta_max_norm_eV_A": float(delta_norm.max()),
        "force_delta_relative_to_mace_rms": float(np.sqrt(np.mean(delta**2)) / denom),
        "force_vector_cosine": cosine,
        "largest_force_delta_atom_index": top,
        "largest_force_delta_norm_eV_A": float(delta_norm[top]),
        "mace_largest_force_atom_index": int(np.argmax(mace_norm)),
        "dft_largest_force_atom_index": int(np.argmax(dft_norm)),
    }


def main() -> None:
    wave = pd.read_csv(WAVE).fillna("")
    records = []
    comparisons = []
    for _, row in wave.iterrows():
        candidate_id = row.candidate_id
        run = RESULTS / candidate_id
        output = parse_qe_output(run / "scf.out")
        base = row.to_dict()
        dft_vectors = output.pop("qe_force_vectors")
        base.update(output)
        base["input_copy_sha256"] = sha256(run / "input.scf.in") if (run / "input.scf.in").is_file() else ""
        base["input_hash_matches"] = base["input_copy_sha256"] == row.input_sha256
        if not base["input_hash_matches"]:
            base["reasons"] = ";".join(filter(None, [base.get("reasons", ""), "input_hash_mismatch"]))
            base["qe_status"] = "not_accepted"
        records.append(base)
        if dft_vectors is None or output.get("qe_status") != "scientifically_accepted":
            continue
        symbols, mace_vectors = read_extxyz(Path(row.local_extxyz))
        if len(symbols) != len(dft_vectors):
            continue
        comparison = {
            "candidate_id": candidate_id,
            "source": row.source,
            "training_seed": int(row.training_seed),
            "run_key": row.run_key,
            "window": row.window,
            "candidate_group": row.candidate_group,
            "selection_role": row.selection_role,
            "time_ps": float(row.time_ps),
            "trigger": row.trigger,
            "model_sha256": row.model_sha256,
            "worst_force_atom_species": symbols[int(output.get("largest_force_delta_atom_index", 0))],
        }
        comparison.update(compare_forces(mace_vectors, dft_vectors))
        comparisons.append(comparison)

    result_df = pd.DataFrame(records)
    comparison_df = pd.DataFrame(comparisons)
    result_df.to_csv(ROOT / "dft_wave_results.csv", index=False)
    comparison_df.to_csv(ROOT / "dft_force_comparison.csv", index=False)

    summary = {
        "campaign_id": "dft_wave_campaign_20261013",
        "wave_size": int(len(wave)),
        "scientifically_accepted": int((result_df.qe_status == "scientifically_accepted").sum()),
        "not_accepted": int((result_df.qe_status != "scientifically_accepted").sum()),
        "force_comparison_rows": int(len(comparison_df)),
        "by_selection_role": result_df.groupby(["selection_role", "qe_status"], dropna=False).size().to_dict() if not result_df.empty else {},
        "by_candidate_group": result_df.groupby(["candidate_group", "qe_status"], dropna=False).size().to_dict() if not result_df.empty else {},
        "quality_gate": {
            "requires_job_done_finite_energy_force_nat_2050_and_complete_atom_forces": True,
            "input_hash_mismatch_is_rejected": True,
        },
        "interpretation_policy": {
            "force_disagreement_is_distributional_evidence_only": True,
            "no_causal_training_data_claim_without_matched_dft_and_mace_force_evidence": True,
            "no_dxy_or_transport_claim_from_this_single_point_wave": True,
        },
    }
    if not comparison_df.empty:
        grouped = comparison_df.groupby("selection_role")[["force_delta_rmse_eV_A", "force_delta_relative_to_mace_rms", "force_vector_cosine"]].median()
        summary["median_force_comparison_by_role"] = grouped.round(6).to_dict(orient="index")
        summary["decision_gate"] = {
            "status": "awaiting_scientific_review",
            "rule": "compare stable controls, precursor rows, and threshold/terminal rows; submit wave 2 only if a reproducible DFT-vs-MACE force pattern is separated from the terminal stable control and is supported by accepted outputs",
        }
    else:
        summary["decision_gate"] = {"status": "hold", "rule": "no accepted DFT force vectors; do not submit wave 2"}
    (ROOT / "dft_wave_analysis.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print("\nquality results:")
    print(result_df[["candidate_id", "selection_role", "qe_status", "reasons"]].to_string(index=False))
    if not comparison_df.empty:
        print("\nforce comparisons:")
        print(comparison_df.groupby("selection_role")[["force_delta_rmse_eV_A", "force_delta_relative_to_mace_rms", "force_vector_cosine"]].median().to_string())


if __name__ == "__main__":
    main()
