#!/usr/bin/env python3
"""Build a fail-closed endpoint-pair acceptance artifact after repeat relax."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from analyze_neb_path_topology import displacement, norm, read_qe_image
from audit_qe_calculator_identity import identity as calculator_identity
from parse_qe_relax_output import last_complete_force_step, parse_relax_out, write_extxyz
from prepare_qe_repeat_relaxations import sha256_file, structure_sha256
from validate_runtime_provenance import validate_runtime_provenance


def minimum_pair_distance(step):
    distances = []
    for left in range(len(step["positions_A"])):
        for right in range(left):
            distances.append(norm(displacement(
                step["positions_A"][left], step["positions_A"][right], step["cell_A"]
            )))
    return min(distances)


def repeat_metrics(parent_step, repeat_step):
    vectors = [
        displacement(left, right, parent_step["cell_A"])
        for left, right in zip(parent_step["positions_A"], repeat_step["positions_A"])
    ]
    magnitudes = [norm(vector) for vector in vectors]
    return {
        "maximum_repeat_relax_displacement_A": max(magnitudes),
        "rms_repeat_relax_displacement_A": math.sqrt(sum(value * value for value in magnitudes) / len(magnitudes)),
        "repeat_relax_energy_drop_eV": parent_step["energy_eV"] - repeat_step["energy_eV"],
        "absolute_final_energy_difference_eV": abs(parent_step["energy_eV"] - repeat_step["energy_eV"]),
        "repeat_final_max_force_eV_A": repeat_step["max_atom_force_eV_A"],
        "repeat_final_minimum_pair_distance_A": minimum_pair_distance(repeat_step),
    }


def evaluate_record(parent_job, repeat_job, out_dir, policy=None):
    policy = policy or {
        "endpoint_gate": {
            "maximum_force_eV_A": 0.05,
            "maximum_repeat_relax_displacement_A": 0.10,
            "maximum_repeat_relax_energy_drop_eV": 0.02,
        },
        "path_gate": {"minimum_pair_distance_A": 1.80},
    }
    endpoint_gate = policy["endpoint_gate"]
    path_gate = policy["path_gate"]
    repeat_dir = Path(repeat_job["local_dir"]).resolve()
    repeat_manifest_path = repeat_dir / "endpoint_relax_manifest.json"
    repeat_manifest = json.loads(repeat_manifest_path.read_text())
    parent_dir = Path(parent_job["local_dir"]).resolve()
    parent_parsed = parse_relax_out(parent_dir / "relax.out", parent_dir / "relax.in")
    repeat_parsed = parse_relax_out(repeat_dir / "relax.out", repeat_dir / "relax.in")
    parent_step = last_complete_force_step(parent_parsed)
    repeat_step = last_complete_force_step(repeat_parsed)
    if parent_step is None or repeat_step is None:
        raise ValueError("Missing complete endpoint force step")
    if parent_step["symbols"] != repeat_step["symbols"]:
        raise ValueError("Atom order changed during repeat relaxation")
    if parent_step.get("cell_A") is None:
        parent_step["cell_A"] = read_qe_image(parent_dir / "relax.in")["cell"]
    if repeat_step.get("cell_A") is None:
        repeat_step["cell_A"] = read_qe_image(repeat_dir / "relax.in")["cell"]
    parent_identity = calculator_identity(parent_dir / "relax.in", "parent")
    repeat_identity = calculator_identity(repeat_dir / "relax.in", "repeat")
    parent_runtime = validate_runtime_provenance(
        parent_dir / "runtime_provenance.json",
        {"relax_input": sha256_file(parent_dir / "relax.in")},
        ["pw.x"],
        str(parent_job["job_id"]),
    )
    repeat_runtime = validate_runtime_provenance(
        repeat_dir / "runtime_provenance.json",
        {"relax_input": sha256_file(repeat_dir / "relax.in")},
        ["pw.x"],
        str(repeat_job["job_id"]),
    )
    metrics = repeat_metrics(parent_step, repeat_step)
    checks = {
        "parent_accepted_local_minimum": parent_job.get("classification") == "accepted_local_minimum",
        "repeat_accepted_local_minimum": repeat_job.get("classification") == "accepted_local_minimum",
        "parent_output_bfgs_converged": parent_parsed.get("job_done") and parent_parsed.get("bfgs_converged"),
        "repeat_output_bfgs_converged": repeat_parsed.get("job_done") and repeat_parsed.get("bfgs_converged"),
        "parent_force_matches_latest_geometry": parent_parsed.get("latest_geometry_has_evaluated_forces"),
        "repeat_force_matches_latest_geometry": repeat_parsed.get("latest_geometry_has_evaluated_forces"),
        "repeat_parent_binding": str(repeat_manifest.get("parent_job_id")) == str(parent_job["job_id"]),
        "repeat_input_binding": repeat_manifest.get("relax_input_sha256") == sha256_file(repeat_dir / "relax.in"),
        "parent_runtime_provenance": parent_runtime["passed"],
        "repeat_runtime_provenance": repeat_runtime["passed"],
        "calculator_identity_complete": parent_identity["identity_complete"] and repeat_identity["identity_complete"],
        "calculator_identity_unchanged": parent_identity["calculator_identity"] == repeat_identity["calculator_identity"],
        "repeat_force_gate": metrics["repeat_final_max_force_eV_A"] <= endpoint_gate["maximum_force_eV_A"],
        "repeat_displacement_gate": metrics["maximum_repeat_relax_displacement_A"] <= endpoint_gate["maximum_repeat_relax_displacement_A"],
        "repeat_energy_lowering_gate": metrics["repeat_relax_energy_drop_eV"] <= endpoint_gate["maximum_repeat_relax_energy_drop_eV"],
        "repeat_energy_consistency_gate": metrics["absolute_final_energy_difference_eV"] <= endpoint_gate["maximum_repeat_relax_energy_drop_eV"],
        "geometry_gate": metrics["repeat_final_minimum_pair_distance_A"] >= path_gate["minimum_pair_distance_A"],
    }
    key = "{}_img{}".format(parent_job["path_id"].replace("/", "_"), parent_job["image_index_qe"])
    structure_path = out_dir / (key + "_repeat_final.extxyz")
    write_extxyz(structure_path, [repeat_step])
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "path_id": parent_job["path_id"],
        "image_index_qe": parent_job["image_index_qe"],
        "parent_job_id": str(parent_job["job_id"]),
        "repeat_job_id": str(repeat_job["job_id"]),
        "status": "accepted" if not failed else "rejected",
        "failed_checks": failed,
        "checks": checks,
        "metrics": metrics,
        "calculator_identity": repeat_identity["calculator_identity"],
        "parent_final_structure_sha256": structure_sha256(parent_step),
        "repeat_final_structure_sha256": structure_sha256(repeat_step),
        "accepted_structure": str(structure_path),
        "accepted_structure_sha256": sha256_file(structure_path),
        "parent_relax_output_sha256": sha256_file(parent_dir / "relax.out"),
        "repeat_relax_output_sha256": sha256_file(repeat_dir / "relax.out"),
        "repeat_manifest_sha256": sha256_file(repeat_manifest_path),
        "parent_runtime_provenance": parent_runtime,
        "repeat_runtime_provenance": repeat_runtime,
    }


def status_index(path):
    payload = json.loads(path.read_text())
    return {(row["path_id"], int(row["image_index_qe"])): row for row in payload["jobs"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-status", type=Path, required=True)
    parser.add_argument("--repeat-status", type=Path, required=True)
    parser.add_argument("--initial", required=True, help="PATH_ID:IMAGE_INDEX")
    parser.add_argument("--final", required=True, help="PATH_ID:IMAGE_INDEX")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path, default=Path("configs/representative_path_selection.json")
    )
    args = parser.parse_args()
    policy_path = args.policy.resolve()
    policy = json.loads(policy_path.read_text())
    first = status_index(args.first_status.resolve())
    repeat = status_index(args.repeat_status.resolve())
    keys = []
    for raw in (args.initial, args.final):
        path_id, image = raw.rsplit(":", 1)
        keys.append((path_id, int(image)))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for key in keys:
        if key not in first or key not in repeat:
            raise ValueError("Endpoint missing from first/repeat status: {}:{}".format(*key))
        records.append(evaluate_record(first[key], repeat[key], out_dir, policy))
    identities = {row["calculator_identity"] for row in records}
    pair_checks = {
        "initial_endpoint_accepted": records[0]["status"] == "accepted",
        "final_endpoint_accepted": records[1]["status"] == "accepted",
        "shared_calculator_identity": len(identities) == 1,
        "distinct_endpoint_structures": records[0]["repeat_final_structure_sha256"] != records[1]["repeat_final_structure_sha256"],
    }
    failed = sorted(name for name, passed in pair_checks.items() if not passed)
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "accepted" if not failed else "rejected",
        "failed_checks": failed,
        "checks": pair_checks,
        "calculator_identity": next(iter(identities)) if len(identities) == 1 else None,
        "initial_structure": records[0]["accepted_structure"],
        "initial_structure_sha256": records[0]["accepted_structure_sha256"],
        "final_structure": records[1]["accepted_structure"],
        "final_structure_sha256": records[1]["accepted_structure_sha256"],
        "endpoint_records": records,
        "first_status": str(args.first_status.resolve()),
        "first_status_sha256": sha256_file(args.first_status.resolve()),
        "repeat_status": str(args.repeat_status.resolve()),
        "repeat_status_sha256": sha256_file(args.repeat_status.resolve()),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
    }
    output = out_dir / "endpoint_pair_acceptance.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "failed_checks": failed, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
