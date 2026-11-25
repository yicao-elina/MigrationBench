#!/usr/bin/env python3
"""Strict terminal audit for the isolated campaign outputs."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main() -> None:
    manifest = json.loads((ROOT / "provenance_manifest.json").read_text())
    expected = manifest["expected"]["runs"]
    rows = []
    for p in sorted((ROOT / "results").glob("*/**/mace_relax_summary.json")):
        d = json.loads(p.read_text())
        rows.append({"path": str(p), "model_label": d.get("model_label"),
            "task_id": d.get("task", {}).get("task_id"),
            "scientific_success": bool(d.get("scientific_success", False)),
            "optimizer_converged": d.get("optimizer_converged"),
            "final_max_force_eV_A": d.get("final_max_force_eV_A"),
            "initial_energy_eV": d.get("initial_energy_eV"),
            "final_energy_eV": d.get("final_energy_eV"),
            "steps": d.get("optimizer_steps_completed"),
            "input_sha256": d.get("input_sha256_verified"),
            "model_sha256": d.get("model_sha256")})
    failures = []
    for p in sorted((ROOT / "results").glob("*/**/failure.json")):
        failures.append({"path": str(p), **json.loads(p.read_text())})
    terminal = len(rows) + len(failures)
    status = {"expected_runs": expected, "success_summaries": sum(r["scientific_success"] for r in rows),
        "summary_count": len(rows), "failure_count": len(failures), "terminal_count": terminal,
        "terminal_complete": terminal == expected,
        "scientific_gate_complete": len(rows) == expected and all(r["scientific_success"] for r in rows)}
    (ROOT / "terminal_audit.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    with (ROOT / "tidy_results.csv").open("w", newline="") as f:
        fields = list(rows[0]) if rows else ["path"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with (ROOT / "failure_retry_ledger.csv").open("w", newline="") as f:
        fields = sorted({k for r in failures for k in r}) or ["path"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(failures)
    print(json.dumps(status, sort_keys=True))

if __name__ == "__main__": main()
