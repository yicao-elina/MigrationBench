#!/usr/bin/env python3
"""Compute-node smoke for the endpoint repeat-relax acceptance state machine."""

import argparse
import hashlib
import json
from pathlib import Path

from build_endpoint_pair_acceptance import evaluate_record
from parse_qe_relax_output import last_complete_force_step, parse_relax_out
from prepare_qe_repeat_relaxations import repeat_input


QE_INPUT = """&CONTROL
  calculation = 'relax'
  restart_mode = 'from_scratch'
  prefix = 'endpoint'
  tprnfor = .true.
  max_seconds = 100
/
&SYSTEM
  nat = 2
  ntyp = 2
  ecutwfc = 50
  ecutrho = 200
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.005
/
&ELECTRONS
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
Cr 1 Cr.UPF
Te 1 Te.UPF
ATOMIC_POSITIONS angstrom
Cr 0 0 0
Te 2.1 0 0
K_POINTS gamma
CELL_PARAMETERS angstrom
10 0 0
0 10 0
0 0 10
"""

QE_OUTPUT = """
! total energy = {energy} Ry
atom 1 type 1 force = 0.0001 0.0 0.0
atom 2 type 2 force = -0.0001 0.0 0.0
Total force = 0.0002
ATOMIC_POSITIONS (angstrom)
Cr {crx} 0 0
Te 2.1 0 0
bfgs converged in 1 scf cycles and 1 bfgs steps
End of BFGS Geometry Optimization
JOB DONE.
"""


def make_endpoint(root, image_index, offset):
    parent_dir = root / "parent_{}".format(image_index)
    repeat_dir = root / "repeat_{}".format(image_index)
    accepted_dir = root / "accepted"
    for directory in (parent_dir, repeat_dir, accepted_dir):
        directory.mkdir(exist_ok=True)
    parent_input = QE_INPUT.replace("Cr 0 0 0", "Cr {:.3f} 0 0".format(offset))
    (parent_dir / "relax.in").write_text(parent_input)
    (parent_dir / "relax.out").write_text(
        QE_OUTPUT.format(energy="-10.0000", crx="{:.3f}".format(offset))
    )
    parent_step = last_complete_force_step(parse_relax_out(parent_dir / "relax.out"))
    parent_step["cell_A"] = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    repeat_text = repeat_input(parent_input, "repeat_{}".format(image_index), 500, parent_step)
    (repeat_dir / "relax.in").write_text(repeat_text)
    (repeat_dir / "relax.out").write_text(
        QE_OUTPUT.format(energy="-10.0001", crx="{:.3f}".format(offset))
    )
    repeat_hash = hashlib.sha256((repeat_dir / "relax.in").read_bytes()).hexdigest()
    parent_hash = hashlib.sha256((parent_dir / "relax.in").read_bytes()).hexdigest()
    for directory, job_id, input_hash in (
        (parent_dir, "parent-{}".format(image_index), parent_hash),
        (repeat_dir, "repeat-{}".format(image_index), repeat_hash),
    ):
        (directory / "runtime_provenance.json").write_text(json.dumps({
            "environment": {"SLURM_JOB_ID": job_id},
            "inputs": {"relax_input": {"exists": True, "sha256": input_hash}},
            "binaries": {"pw.x": {"exists": True, "sha256": "fixture-pw"}},
        }))
    (repeat_dir / "endpoint_relax_manifest.json").write_text(json.dumps({
        "parent_job_id": "parent-{}".format(image_index),
        "relax_input_sha256": repeat_hash,
    }))
    parent_job = {
        "job_id": "parent-{}".format(image_index),
        "path_id": "smoke",
        "image_index_qe": image_index,
        "local_dir": str(parent_dir),
        "classification": "accepted_local_minimum",
    }
    repeat_job = {
        "job_id": "repeat-{}".format(image_index),
        "path_id": "smoke",
        "image_index_qe": image_index,
        "local_dir": str(repeat_dir),
        "classification": "accepted_local_minimum",
    }
    return evaluate_record(parent_job, repeat_job, accepted_dir), parent_job, repeat_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.output.resolve().parent / "fixture"
    root.mkdir(parents=True, exist_ok=True)
    fixtures = [make_endpoint(root, 1, 0.0), make_endpoint(root, 2, 4.5)]
    records = [row[0] for row in fixtures]
    first_status = root / "first_status.json"
    repeat_status = root / "repeat_status.json"
    first_status.write_text(json.dumps({"jobs": [row[1] for row in fixtures]}, indent=2) + "\n")
    repeat_status.write_text(json.dumps({"jobs": [row[2] for row in fixtures]}, indent=2) + "\n")
    checks = {
        "both_records_accepted": all(row["status"] == "accepted" for row in records),
        "same_calculator_identity": len({row["calculator_identity"] for row in records}) == 1,
        "distinct_structures": records[0]["repeat_final_structure_sha256"] != records[1]["repeat_final_structure_sha256"],
        "repeat_inputs_unique": records[0]["repeat_manifest_sha256"] != records[1]["repeat_manifest_sha256"],
    }
    payload = {
        "schema_version": "1.0",
        "seed": args.seed,
        "status": "accepted" if all(checks.values()) else "rejected",
        "smoke_status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "records": records,
        "endpoint_records": records,
        "first_status": str(first_status),
        "repeat_status": str(repeat_status),
        "initial_structure": records[0]["accepted_structure"],
        "initial_structure_sha256": records[0]["accepted_structure_sha256"],
        "final_structure": records[1]["accepted_structure"],
        "final_structure_sha256": records[1]["accepted_structure_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "smoke_status": payload["smoke_status"], "checks": checks}, indent=2))
    if payload["smoke_status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
