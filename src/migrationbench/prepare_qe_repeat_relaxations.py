#!/usr/bin/env python3
"""Prepare repeat QE relaxations from independently converged endpoint jobs."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from analyze_neb_path_topology import read_qe_image
from audit_qe_calculator_identity import identity as calculator_identity
from parse_qe_relax_output import last_complete_force_step, parse_relax_out
from prepare_qe_path_iteration_relax_pairs import replace_positions
from prepare_qe_scf_warmup import set_namelist_value, walltime_seconds
from validate_runtime_provenance import validate_runtime_provenance


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure_sha256(step):
    payload = {
        "symbols": step["symbols"],
        "positions_A": [[round(float(value), 12) for value in row] for row in step["positions_A"]],
        "cell_A": [[round(float(value), 12) for value in row] for row in step["cell_A"]],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def repeat_input(text, prefix, max_seconds, step):
    updates = {
        "calculation": "'relax'",
        "restart_mode": "'from_scratch'",
        "prefix": repr(prefix),
        "tprnfor": ".true.",
        "max_seconds": str(max_seconds),
    }
    for key, value in updates.items():
        text = set_namelist_value(text, "CONTROL", key, value)
    return replace_positions(text, step["symbols"], step["positions_A"])


def repeat_identifiers(path_id, iteration, image_index, seed, branch_tag=""):
    safe_path = re.sub(r"[^A-Za-z0-9]+", "", path_id).lower()
    tag = re.sub(r"[^A-Za-z0-9]+", "", branch_tag or "").lower()
    if branch_tag and not tag:
        raise ValueError("branch-tag must contain at least one alphanumeric character")
    tagged_role = "_{}".format(tag) if tag else ""
    return {
        "branch_id": "{}_iter{}_img{}{}_repeat_relax_s{}".format(
            path_id, iteration, image_index, tagged_role, seed
        ),
        "job_name": "mb_rr{}{}i{}s{}".format(tag[:8], safe_path, image_index, seed),
        "prefix": "rr_{}_{}_i{}_s{}".format(tag or "base", safe_path, image_index, seed),
        "branch_tag": tag or None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-status", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--select", action="append", help="PATH_ID:IMAGE_INDEX; default all accepted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-seconds", type=int, default=84600)
    parser.add_argument("--ntasks", type=int, default=2)
    parser.add_argument("--memory", default="160G")
    parser.add_argument(
        "--branch-tag",
        help="Short lineage tag, for example direct or geomvoid, to prevent paired-repeat collisions.",
    )
    parser.add_argument(
        "--policy", type=Path, default=Path("configs/representative_path_selection.json")
    )
    args = parser.parse_args()
    if args.max_seconds >= walltime_seconds(args.walltime):
        raise ValueError("max_seconds must be below the Slurm walltime")

    policy_path = args.policy.resolve()
    policy = json.loads(policy_path.read_text())
    endpoint_gate = policy["endpoint_gate"]
    path_gate = policy["path_gate"]
    selected = set(args.select or [])
    branch_tag = re.sub(r"[^A-Za-z0-9]+", "", args.branch_tag or "").lower()
    if args.branch_tag and not branch_tag:
        raise ValueError("branch-tag must contain at least one alphanumeric character")
    source_status = args.first_status.resolve()
    status = json.loads(source_status.read_text())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for source in status["jobs"]:
        key = "{}:{}".format(source["path_id"], source["image_index_qe"])
        if selected and key not in selected:
            continue
        if source.get("classification") != "accepted_local_minimum":
            continue
        local_dir = Path(source["local_dir"]).resolve()
        first_input = local_dir / "relax.in"
        first_output = local_dir / "relax.out"
        first_runtime = validate_runtime_provenance(
            local_dir / "runtime_provenance.json",
            {"relax_input": sha256_file(first_input)},
            ["pw.x"],
            str(source["job_id"]),
        )
        if not first_runtime["passed"]:
            raise ValueError(
                "Accepted first relaxation lacks valid runtime provenance for {}: {}".format(
                    key, ", ".join(first_runtime["failed_checks"])
                )
            )
        parsed = parse_relax_out(first_output, first_input)
        step = last_complete_force_step(parsed)
        if not parsed["job_done"] or not parsed["bfgs_converged"] or step is None:
            raise ValueError("Accepted status has incomplete output for {}".format(key))
        if (
            step["max_atom_force_eV_A"] is None
            or step["max_atom_force_eV_A"] > endpoint_gate["maximum_force_eV_A"]
        ):
            raise ValueError("Accepted status fails force gate for {}".format(key))
        if step.get("cell_A") is None:
            step["cell_A"] = read_qe_image(first_input)["cell"]
        identifiers = repeat_identifiers(
            source["path_id"], source.get("source_iteration", "na"),
            source["image_index_qe"], args.seed, branch_tag,
        )
        branch_id = identifiers["branch_id"]
        job_name = identifiers["job_name"]
        branch_dir = out_dir / branch_id
        branch_dir.mkdir(parents=True, exist_ok=True)
        output_input = branch_dir / "relax.in"
        output_input.write_text(repeat_input(
            first_input.read_text(),
            identifiers["prefix"],
            args.max_seconds,
            step,
        ))
        first_identity = calculator_identity(first_input, "first")
        repeat_identity = calculator_identity(output_input, "repeat")
        if not first_identity["identity_complete"] or first_identity["calculator_identity"] != repeat_identity["calculator_identity"]:
            raise ValueError("Calculator identity changed for {}".format(key))
        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "branch_id": branch_id,
            "job_name": job_name,
            "seed": args.seed,
            "scientific_role": "repeat_relax_local_minimum_stability_check",
            "repeat_branch_tag": identifiers["branch_tag"],
            "source_path_id": source["path_id"],
            "source_iteration": source.get("source_iteration"),
            "source_image_index_qe": source["image_index_qe"],
            "parent_job_id": str(source["job_id"]),
            "parent_status_file": str(source_status),
            "parent_status_file_sha256": sha256_file(source_status),
            "parent_relax_input": str(first_input),
            "parent_relax_input_sha256": sha256_file(first_input),
            "parent_relax_output": str(first_output),
            "parent_relax_output_sha256": sha256_file(first_output),
            "parent_runtime_provenance": first_runtime["path"],
            "parent_runtime_provenance_sha256": first_runtime["sha256"],
            "parent_final_structure_sha256": structure_sha256(step),
            "parent_final_energy_eV": step["energy_eV"],
            "parent_final_max_force_eV_A": step["max_atom_force_eV_A"],
            "calculator_identity": first_identity["calculator_identity"],
            "policy": str(policy_path),
            "policy_sha256": sha256_file(policy_path),
            "resources": {
                "walltime": args.walltime,
                "pw_max_seconds": args.max_seconds,
                "ntasks": args.ntasks,
                "memory": args.memory,
            },
            "repeat_gate": {
                "job_done": True,
                "bfgs_converged": True,
                "final_max_force_eV_A_lte": endpoint_gate["maximum_force_eV_A"],
                "maximum_structure_displacement_A": endpoint_gate["maximum_repeat_relax_displacement_A"],
                "maximum_energy_drop_eV": endpoint_gate["maximum_repeat_relax_energy_drop_eV"],
                "minimum_pair_distance_A": path_gate["minimum_pair_distance_A"],
                "calculator_identity_unchanged": True,
            },
            "relax_input": str(output_input),
            "relax_input_sha256": sha256_file(output_input),
        }
        manifest_path = branch_dir / "endpoint_relax_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest["manifest_path"] = str(manifest_path)
        jobs.append(manifest)
    if selected:
        found = {"{}:{}".format(row["source_path_id"], row["source_image_index_qe"]) for row in jobs}
        missing = sorted(selected - found)
        if missing:
            raise ValueError("Selected endpoints are not accepted first relaxations: " + ", ".join(missing))
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "first_status": str(source_status),
        "first_status_sha256": sha256_file(source_status),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "jobs": jobs,
    }
    batch_path = out_dir / "endpoint_relax_batch_manifest.json"
    batch_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prepared": len(jobs), "batch_manifest": str(batch_path)}, indent=2))


if __name__ == "__main__":
    main()
