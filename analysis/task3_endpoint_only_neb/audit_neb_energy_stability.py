#!/usr/bin/env python3
"""Audit observable NEB iterations and penultimate/final energy stability.

The QE NEB logs expose outer path iterations and one self-consistency block per
image, but not the inner electronic SCF iteration count.  This distinction is
recorded explicitly rather than inferred.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SNAP = ROOT / "remote_snapshot"
OUT = ROOT / "exploratory_energy_audit"


def read_tables(text: str, n_images: int = 10) -> list[dict]:
    lines = text.splitlines()
    tables = []
    for i, line in enumerate(lines):
        if not re.search(r"^\s*image\s+energy \(eV\)", line):
            continue
        rows = []
        for row in lines[i + 2 : i + 2 + n_images]:
            m = re.match(r"^\s*(\d+)\s+([-+0-9.]+|\*+)\s+([-+0-9.]+|\*+)\s+([FT])", row)
            if not m:
                break
            try:
                energy = float(m.group(2)); error = float(m.group(3))
            except ValueError:
                energy = np.nan; error = np.nan
            rows.append({"image": int(m.group(1)), "energy_eV": energy, "error_eV_A": error, "frozen": m.group(4)})
        if len(rows) == n_images:
            tables.append({"rows": rows, "finite": all(np.isfinite(x["energy_eV"]) and np.isfinite(x["error_eV_A"]) for x in rows)})
    return tables


def choose_output(path: Path) -> Path | None:
    candidates = sorted(path.glob("neb*.out"))
    completed = [p for p in candidates if "JOB DONE" in p.read_text(errors="replace")]
    if completed:
        # Prefer the canonical neb.out when it is a completed run; otherwise use
        # the completed restart output with the most complete energy history.
        canonical = [p for p in completed if p.name == "neb.out"]
        return canonical[0] if canonical else max(completed, key=lambda p: len(read_tables(p.read_text(errors="replace"))))
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, image_rows = [], []
    for system_dir in sorted(SNAP.glob("2L-*/NEB")):
        for path_dir in sorted(x for x in system_dir.iterdir() if x.is_dir()):
            path_id = f"{system_dir.parent.name}__{path_dir.name}"
            output = choose_output(path_dir)
            if output is None:
                summary.append({"path_id": path_id, "output": "", "output_sha256": "", "job_done": False, "neb_outer_iterations": 0, "self_consistency_image_calls": 0, "inner_electronic_scf_iterations": "not_recorded", "n_energy_tables": 0, "n_complete_finite_energy_tables": 0, "penultimate_final_available": False, "max_abs_image_energy_delta_eV": "", "rms_image_energy_delta_eV": "", "barrier_penultimate_eV": "", "barrier_final_eV": "", "barrier_delta_eV": "", "energy_delta_lt_0p1_eV": False, "exploratory_energy_stable": False, "failure_reason": "no_completed_neb_output"})
                continue
            text = output.read_text(errors="replace")
            outer = [int(x) for x in re.findall(r"iteration\s+(\d+)", text)]
            scf_calls = re.findall(r"self-consistency for image\s+(\d+)", text)
            tables = read_tables(text)
            finite = [t for t in tables if t["finite"]]
            last_two = finite[-2:] if len(finite) >= 2 else []
            reason = []
            if "SCF convergence NOT achieved" in text:
                reason.append("final_scf_warning_present")
            if not last_two:
                reason.append("fewer_than_two_complete_finite_energy_tables")
            if last_two:
                prev = np.array([x["energy_eV"] for x in last_two[0]["rows"]])
                final = np.array([x["energy_eV"] for x in last_two[1]["rows"]])
                delta = final - prev
                abs_delta = np.abs(delta)
                barrier_prev = float(np.max(prev) - prev[0])
                barrier_final = float(np.max(final) - final[0])
                barrier_delta = barrier_final - barrier_prev
                stable = bool(np.max(abs_delta) < 0.1)
                for r, d in zip(last_two[1]["rows"], delta):
                    image_rows.append({"path_id": path_id, "image": r["image"], "penultimate_energy_eV": float(prev[r["image"] - 1]), "final_energy_eV": float(r["energy_eV"]), "delta_eV": float(d), "abs_delta_eV": float(abs(d)), "energy_delta_lt_0p1_eV": bool(abs(d) < 0.1), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
            else:
                abs_delta = np.array([]); barrier_prev = barrier_final = barrier_delta = np.nan; stable = False
            summary.append({"path_id": path_id, "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "job_done": "JOB DONE" in text, "neb_outer_iterations": max(outer) if outer else 0, "self_consistency_image_calls": len(scf_calls), "self_consistency_calls_by_image": json.dumps({str(i): scf_calls.count(str(i)) for i in range(1, 11)}), "inner_electronic_scf_iterations": "not_recorded_in_neb_output", "n_energy_tables": len(tables), "n_complete_finite_energy_tables": len(finite), "penultimate_final_available": bool(last_two), "max_abs_image_energy_delta_eV": float(np.max(abs_delta)) if len(abs_delta) else "", "rms_image_energy_delta_eV": float(np.sqrt(np.mean(abs_delta**2))) if len(abs_delta) else "", "barrier_penultimate_eV": barrier_prev, "barrier_final_eV": barrier_final, "barrier_delta_eV": barrier_delta, "energy_delta_lt_0p1_eV": stable, "exploratory_energy_stable": stable, "failure_reason": ";".join(reason)})
    write_csv(OUT / "scf_neb_iteration_energy_audit.csv", summary)
    write_csv(OUT / "scf_neb_image_energy_deltas.csv", image_rows)
    audit = {"n_paths": len(summary), "n_energy_stable_under_0p1_eV": sum(x["exploratory_energy_stable"] for x in summary), "inner_electronic_scf_iterations": "not_recorded_in_neb_output", "interpretation": "energy-table stability supports exploratory single-point/path sampling only; it does not establish DFT NEB convergence", "summary_sha256": hashlib.sha256((OUT / "scf_neb_iteration_energy_audit.csv").read_bytes()).hexdigest()}
    (OUT / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
