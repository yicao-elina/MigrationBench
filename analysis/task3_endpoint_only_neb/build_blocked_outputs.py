#!/usr/bin/env python3
"""Emit explicit blocked/empty schemas when the DFT gate prevents formal MACE NEB."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main() -> None:
    rows = list(csv.DictReader((ROOT / "dft_gate_manifest.csv").open()))
    manifest_sha = hashlib.sha256((ROOT / "dft_gate_manifest.csv").read_bytes()).hexdigest()
    env = "not_run_dft_gate_blocked"
    summary_fields = ["path_id", "model_id", "seed", "initialization", "n_images", "optimizer", "path_thr_eV_A", "status", "scientific_success", "failure_reason", "manifest_sha256", "input_sha256", "model_sha256", "environment"]
    summary = []
    for r in rows:
        for init in ("linear", "idpp"):
            summary.append({"path_id": r["path_id"], "model_id": "not_run", "seed": "", "initialization": init, "n_images": r["declared_n_images"], "optimizer": "FIRE", "path_thr_eV_A": r["path_thr_eV_A"], "status": "blocked_dft_gate", "scientific_success": "False", "failure_reason": r["failure_reason"], "manifest_sha256": manifest_sha, "input_sha256": f"{r['initial_endpoint_sha256']};{r['final_endpoint_sha256']}", "model_sha256": "", "environment": env})
    write(ROOT / "mace_neb_summary.csv", summary_fields, summary)
    write(ROOT / "mace_neb_image_metrics.csv", ["path_id", "model_id", "seed", "initialization", "image_index", "reaction_coordinate", "energy_eV_relative_model", "max_force_eV_A", "finite_energy", "finite_force", "scientific_success", "failure_reason"], [])
    failure_fields = ["record_id", "scope", "path_id", "attempt", "initialization", "status", "scientific_success", "failure_reason", "source_path", "source_sha256", "manifest_sha256", "retry_action"]
    failure = [{"record_id": f"dft_gate_{r['path_id']}", "scope": "DFT_NEb_gate", "path_id": r["path_id"], "attempt": "0", "initialization": "linear;idpp", "status": "blocked", "scientific_success": "False", "failure_reason": r["failure_reason"], "source_path": r["dft_output_path"], "source_sha256": r["dft_output_sha256"], "manifest_sha256": manifest_sha, "retry_action": "await explicit DFT path convergence and all-image SCF convergence; do not launch MACE formal stage"} for r in rows]
    write(ROOT / "mace_neb_failure_retry_ledger.csv", failure_fields, failure)
    comparison = [{"path_id": r["path_id"], "system": r["system"], "dopant": r["dopant"], "linear_status": "geometry_smoke_only", "idpp_status": "geometry_smoke_only_representative_only", "dft_gate_status": "blocked", "formal_mace_status": "not_run", "scientific_success": "False", "failure_reason": r["failure_reason"], "manifest_sha256": manifest_sha} for r in rows]
    write(ROOT / "mace_neb_path_comparison.csv", list(comparison[0]), comparison)
    smoke = {"status": "blocked_dft_gate", "representative_path": "2L-Cu__neb-gap-diffuse", "dft_gate_eligible": False, "formal_mace_command_executed": False, "reason": "No audited DFT path has explicit path convergence; 9/10 completed logs retain an SCF-not-converged image warning; the remaining path lacks a completed output.", "planned_protocol": {"n_images": 10, "fixed_cell": True, "optimizer": "FIRE", "path_thr_eV_A": 0.001, "seeds": [123], "models": "registered Rockfish models only after gate"}, "manifest_sha256": manifest_sha}
    (ROOT / "formal_gate_smoke.json").write_text(json.dumps(smoke, indent=2) + "\n")


if __name__ == "__main__":
    main()
