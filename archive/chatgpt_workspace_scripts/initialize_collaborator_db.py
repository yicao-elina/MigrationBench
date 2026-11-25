#!/usr/bin/env python3
"""Initialize the provenance-first collaborator DFT/NEB SQLite catalog.

This only registers observed source roots and manually reviewed catalog rows.
Scientific acceptance is deliberately left to the parser/quality-gate stage.
"""
from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "collaborator_dft_neb"
DB = DATA / "migrationbench_dft_neb.sqlite"


def main() -> None:
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript((DATA / "schema.sql").read_text())
    now = datetime.now(timezone.utc).isoformat()
    with (DATA / "source_inventory.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            con.execute(
                "INSERT INTO source VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (row["source_id"], row["owner"], row["username"], row["host"],
                 row["root_path"], row["access_state"], now, row["notes"]),
            )
    with (DATA / "calculation_catalog.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            con.execute(
                "INSERT INTO calculation VALUES (?, ?, NULL, ?, ?, NULL, ?, ?, ?, ?)",
                (row["calculation_id"], row["source_id"], row["source_path"],
                 row["calculation_type"], row["method_status"], row["result_status"],
                 now, row["notes"]),
            )
    with (DATA / "trajectory_candidates.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            con.execute(
                "INSERT INTO trajectory(trajectory_id,calculation_id,n_images,image_order_status,endpoint_identity_status,barrier_ev,barrier_method,convergence_status,trajectory_uri) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)",
                (row["trajectory_id"], row["calculation_id"], row["quality_status"],
                 "unknown", float(row["reported_forward_barrier_ev"]),
                 "reported_activation_energy_forward", row["completion_evidence"],
                 row["source_path"]),
            )
            con.execute(
                "INSERT INTO result(calculation_id,metric_name,value,unit,uncertainty,extraction_method) VALUES (?, ?, ?, ?, NULL, ?)",
                (row["calculation_id"], f"{row['path_label']}_forward_activation_energy",
                 float(row["reported_forward_barrier_ev"]), "eV",
                 "remote output text audit; not accepted as publication result"),
            )
            con.execute(
                "INSERT INTO quality_gate(gate_id,calculation_id,status,evidence,checked_utc) VALUES (?, ?, ?, ?, ?)",
                (f"Q3:{row['trajectory_id']}", row["calculation_id"], row["quality_status"],
                 row["completion_evidence"], now),
            )
    with (DATA / "endpoint_candidates.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            method = {"label": row["method"], "ecutwfc_ry": row["ecutwfc_ry"],
                      "ecutrho_ry": row["ecutrho_ry"], "scf_conv_thr": row["scf_conv_thr"]}
            con.execute(
                "INSERT INTO endpoint VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row["endpoint_id"], row["source_id"], row["source_path"],
                 row["system_label"], row["site_label"], row["calculation_type"],
                 str(method), int(row["natoms"]),
                 None if not row["reported_energy_ry"] else float(row["reported_energy_ry"]),
                row["optimizer_evidence"], row["quality_status"], row["publication_role"]),
            )
    with (DATA / "force_vector_audit.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            con.execute(
                "INSERT INTO endpoint_quality VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (row["endpoint_id"], float(row["last_local_force_max_ry_au"]),
                 float(row["approx_max_force_ev_ang"]), 0.03,
                 row["force_threshold_030_ev_ang"], row["parser_status"],
                row["notes"], now),
            )
    with (DATA / "endpoint_provenance.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            con.execute(
                "INSERT INTO endpoint_provenance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row["endpoint_id"], row["input_sha256"], row["output_sha256"],
                 int(row["input_size_bytes"]), int(row["output_size_bytes"]),
                 float(row["final_energy_ry"]), row["raw_host"],
                row["raw_input_path"], row["raw_output_path"], row["provenance_status"]),
            )
    with (DATA / "system_identity.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            con.execute(
                "INSERT INTO endpoint_identity VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (row["endpoint_id"], row["system_label"], row["formula"],
                 int(row["natoms"]), row["dopant_species"], int(row["dopant_count"]),
                 row["host_counts"], row["cell_source"], row["pbc_status"],
                 row["identity_status"]),
            )
    con.commit()
    print(f"created {DB} with {con.execute('SELECT count(*) FROM source').fetchone()[0]} sources and "
          f"{con.execute('SELECT count(*) FROM calculation').fetchone()[0]} calculations")
    con.close()


if __name__ == "__main__":
    main()
