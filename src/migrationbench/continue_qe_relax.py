#!/usr/bin/env python3
"""Continue a cleanly stopped QE relaxation from its last complete geometry."""

import argparse
import fcntl
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from analyze_neb_path_topology import read_qe_image
from audit_qe_calculator_identity import identity as calculator_identity
from expand_qe_scf_warmup_batch import atomic_json, find_remote_job, run, sha256_file
from parse_qe_relax_output import last_complete_force_step, last_printed_geometry, parse_relax_out
from prepare_qe_path_iteration_relax_pairs import replace_positions
from prepare_qe_scf_warmup import set_namelist_value, walltime_seconds


def select_parent(status, job_id, mode="continuation"):
    rows = [row for row in status["jobs"] if str(row["job_id"]) == str(job_id)]
    if len(rows) != 1:
        raise RuntimeError(f"expected one parent row for {job_id}, found {len(rows)}")
    parent = rows[0]
    allowed = (
        {"relax_done_force_gate_failed"}
        if mode == "force-refinement"
        else {"clean_max_seconds_restartable", "clean_max_ionic_steps_restartable"}
    )
    if parent.get("classification") not in allowed:
        raise RuntimeError(f"parent {job_id} is not cleanly restartable: {parent.get('classification')}")
    return parent


def derive_name(value, attempt, seed):
    suffix = f"r{attempt}s{seed}"
    if re.search(r"r\d+s\d+$", value):
        return re.sub(r"r\d+s\d+$", suffix, value)
    if re.search(r"s\d+$", value):
        return re.sub(r"s\d+$", suffix, value)
    return value + "_" + suffix


def parent_seed(parent, requested_seed=None):
    candidates = set()
    for value in (parent.get("job_name"), parent.get("branch_id")):
        match = re.search(r"s(\d+)$", value or "")
        if match:
            candidates.add(int(match.group(1)))
    if parent.get("seed") is not None:
        candidates.add(int(parent["seed"]))
    if requested_seed is not None:
        candidates.add(int(requested_seed))
    if len(candidates) != 1:
        raise RuntimeError(f"cannot resolve one seed from parent/request: {sorted(candidates)}")
    return candidates.pop()


def structure_sha256(step):
    payload = {
        "symbols": step["symbols"],
        "positions_A": [[round(float(value), 12) for value in row] for row in step["positions_A"]],
        "cell_A": [[round(float(value), 12) for value in row] for row in step["cell_A"]],
    }
    return __import__("hashlib").sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(args, parent):
    parent_dir = Path(parent["local_dir"]).resolve()
    parent_input = parent_dir / "relax.in"
    parent_output = parent_dir / "relax.out"
    parent_manifest_path = parent_dir / "endpoint_relax_manifest.json"
    for required in (parent_input, parent_output, parent_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    parsed = parse_relax_out(parent_output, parent_input)
    force_step = last_complete_force_step(parsed)
    step = last_printed_geometry(parsed)
    if args.mode == "force-refinement":
        clean_stop = (
            parsed.get("job_done")
            and parsed.get("bfgs_converged")
            and parsed.get("latest_geometry_has_evaluated_forces")
            and parsed.get("final_max_atom_force_eV_A") is not None
            and parsed["final_max_atom_force_eV_A"] > args.target_force_eV_A
        )
    else:
        clean_stop = parsed.get("max_seconds_reached") or (
            parsed.get("job_done") and parsed.get("max_ionic_steps_reached")
        )
    if step is None or force_step is None or parsed.get("scf_not_converged_count") or not clean_stop:
        raise RuntimeError("parent output does not contain a clean restartable geometry")
    if step.get("cell_A") is None:
        step["cell_A"] = read_qe_image(parent_input)["cell"]
    if not step["cell_A"] or not all(math.isfinite(value) for row in step["positions_A"] for value in row):
        raise RuntimeError("parent final geometry is incomplete or non-finite")

    seed = parent_seed(parent, args.seed)
    job_name = derive_name(parent["job_name"], args.attempt, seed)
    branch_id = derive_name(parent["branch_id"], args.attempt, seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_input = args.out_dir / "relax.in"
    output_manifest = args.out_dir / "endpoint_relax_manifest.json"
    resources = {
        "walltime": args.walltime,
        "pw_max_seconds": args.max_seconds,
        "nstep": args.nstep,
        "ntasks": args.ntasks,
        "memory": args.memory,
        "disable_symmetry": bool(getattr(args, "disable_symmetry", False)),
    }
    if args.mode == "force-refinement":
        resources.update({
            "etot_conv_thr_Ry": args.etot_conv_thr,
            "forc_conv_thr_Ry_Bohr": args.forc_conv_thr,
            "target_max_atom_force_eV_A": args.target_force_eV_A,
        })
    parent_manifest = json.loads(parent_manifest_path.read_text())
    inherited = parent_manifest.get("lineage_metrics_before_segment", {})
    totals = {
        "ionic_steps": int(inherited.get("ionic_steps", 0)) + int(parsed["n_ionic_steps"]),
        "scf_iterations": int(inherited.get("scf_iterations", 0)) + int(parsed["total_scf_iterations_seen"]),
        "segments": int(inherited.get("segments", 0)) + 1,
    }
    if output_input.exists() and output_manifest.exists():
        manifest = json.loads(output_manifest.read_text())
        checks = {
            "parent_job_id": str(manifest.get("parent", {}).get("job_id")) == str(parent["job_id"]),
            "attempt": manifest.get("attempt") == args.attempt,
            "input_hash": manifest.get("relax_input_sha256") == sha256_file(output_input),
            "parent_output_hash": manifest.get("parent", {}).get("relax_output_sha256") == sha256_file(parent_output),
            "resources": manifest.get("resources") == resources,
        }
        if not all(checks.values()):
            raise RuntimeError(f"existing continuation conflicts with request: {checks}")
        return manifest, output_manifest
    if output_input.exists() or output_manifest.exists():
        raise RuntimeError(f"partial continuation artifacts in {args.out_dir}")

    text = parent_input.read_text()
    for key, value in {
        "calculation": "'relax'",
        "restart_mode": "'from_scratch'",
        "prefix": repr(f"{job_name}_pw"),
        "tprnfor": ".true.",
        "max_seconds": str(args.max_seconds),
        "nstep": str(args.nstep),
    }.items():
        text = set_namelist_value(text, "CONTROL", key, value)
    if args.mode == "force-refinement":
        text = set_namelist_value(text, "CONTROL", "etot_conv_thr", str(args.etot_conv_thr))
        text = set_namelist_value(text, "CONTROL", "forc_conv_thr", str(args.forc_conv_thr))
    if getattr(args, "disable_symmetry", False):
        text = set_namelist_value(text, "SYSTEM", "nosym", ".true.")
        text = set_namelist_value(text, "SYSTEM", "noinv", ".true.")
    text = replace_positions(text, step["symbols"], step["positions_A"])
    output_input.write_text(text)
    parent_identity = calculator_identity(parent_input, "parent")
    continuation_identity = calculator_identity(output_input, "continuation")
    if (
        not parent_identity["identity_complete"]
        or parent_identity["calculator_identity"] != continuation_identity["calculator_identity"]
    ):
        raise RuntimeError("calculator identity changed across relaxation continuation")
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": (
            "validated_qe_relax_force_refinement"
            if args.mode == "force-refinement"
            else "validated_qe_relax_geometry_continuation"
        ),
        "branch_id": branch_id,
        "job_name": job_name,
        "path_id": parent["path_id"],
        "source_path_id": parent["path_id"],
        "source_image_index_qe": parent["image_index_qe"],
        "seed": seed,
        "attempt": args.attempt,
        "parent": {
            "job_id": str(parent["job_id"]),
            "branch_id": parent["branch_id"],
            "job_name": parent["job_name"],
            "classification": parent["classification"],
            "remote_run_dir": parent["remote_run_dir"],
            "status_file": str(args.relax_status.resolve()),
            "status_file_sha256": sha256_file(args.relax_status),
            "relax_input_sha256": sha256_file(parent_input),
            "relax_output_sha256": sha256_file(parent_output),
            "manifest_sha256": sha256_file(parent_manifest_path),
            "latest_printed_geometry_sha256": structure_sha256(step),
            "latest_geometry_has_evaluated_forces": parsed["latest_geometry_has_evaluated_forces"],
            "last_complete_force_geometry_sha256": structure_sha256(force_step),
            "last_complete_ionic_step": force_step["ionic_step"],
            "last_complete_energy_eV": force_step["energy_eV"],
            "last_complete_max_force_eV_A": force_step["max_atom_force_eV_A"],
        },
        "lineage_metrics_before_segment": totals,
        "calculator_identity": parent_identity["calculator_identity"],
        "resources": resources,
        "continuation_policy": {
            "mode": args.mode,
            "geometry_source": "parent_latest_printed_geometry",
            "electronic_restart": "from_scratch",
            "bfgs_hessian_restart": False,
            "parent_mutated": False,
            "symmetry_release": bool(getattr(args, "disable_symmetry", False)),
            "reason": "portable continuation without relying on incomplete electronic restart files",
            "parent_stop_reason": (
                "benchmark_force_gate_failed_after_qe_bfgs"
                if args.mode == "force-refinement"
                else "max_seconds" if parsed.get("max_seconds_reached") else "max_ionic_steps"
            ),
        },
        "relax_input": str(output_input.resolve()),
        "relax_input_sha256": sha256_file(output_input),
    }
    atomic_json(output_manifest, manifest)
    return manifest, output_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relax-status", type=Path, required=True)
    parser.add_argument("--active-jobs", type=Path, required=True)
    parser.add_argument("--parent-job-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--remote-input-dir", required=True)
    parser.add_argument("--remote-code-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    parser.add_argument("--remote-wrapper", default="/scratch16/pclancy3/yi/revision1_migrationbench_pipeline/scripts/migrationbench/submit_slurm_qe_relax.sh")
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("continuation", "force-refinement"), default="continuation")
    parser.add_argument("--target-force-eV-A", type=float, default=0.05)
    parser.add_argument("--etot-conv-thr", type=float, default=1.0e-6)
    parser.add_argument("--forc-conv-thr", type=float, default=5.0e-4)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-seconds", type=int, default=84600)
    parser.add_argument("--nstep", type=int, default=200)
    parser.add_argument("--ntasks", type=int, default=2)
    parser.add_argument("--memory", default="160G")
    parser.add_argument(
        "--disable-symmetry", action="store_true",
        help="Release QE space-group and inversion constraints for a true all-coordinate local-minimum refinement.",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.attempt < 2:
        raise ValueError("continuation attempt must be at least 2")
    if args.max_seconds >= walltime_seconds(args.walltime):
        raise ValueError("max_seconds must be below Slurm walltime")
    if args.nstep < 1:
        raise ValueError("nstep must be positive")
    status = json.loads(args.relax_status.read_text())
    parent = select_parent(status, args.parent_job_id, args.mode)
    lock_path = args.active_jobs.with_name(args.active_jobs.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest, manifest_path = prepare(args, parent)
        if args.submit:
            registry = json.loads(args.active_jobs.read_text())
            registered = next((row for row in registry["jobs"] if row["job_name"] == manifest["job_name"]), None)
            job_id = str(registered["job_id"]) if registered else find_remote_job(args.ssh_host, manifest["job_name"])
            submission_state = "already_registered" if registered else "recovered_from_rockfish" if job_id else "submitted"
            if not job_id:
                run(["ssh", args.ssh_host, "mkdir", "-p", args.remote_input_dir])
                run(["rsync", "-a", str(args.out_dir) + "/", f"{args.ssh_host}:{args.remote_input_dir}/"])
                export = ",".join([
                    "ALL",
                    f"MIGRATIONBENCH_RELAX_INPUT={args.remote_input_dir}/relax.in",
                    f"MIGRATIONBENCH_RELAX_MANIFEST={args.remote_input_dir}/endpoint_relax_manifest.json",
                    f"MIGRATIONBENCH_CODE_ROOT={args.remote_code_root}",
                    f"MIGRATIONBENCH_RUN_ROOT={args.remote_run_root}",
                ])
                submitted = run([
                    "ssh", args.ssh_host, "sbatch", "--parsable", "--job-name", manifest["job_name"],
                    "--time", args.walltime, "--ntasks-per-node", str(args.ntasks), "--mem", args.memory,
                    "--export", export, args.remote_wrapper,
                ])
                job_id = submitted.split(";", 1)[0].strip()
                if not re.fullmatch(r"\d+", job_id):
                    raise RuntimeError(f"could not parse Slurm job id from {submitted!r}")
            if not registered:
                registry["jobs"] = [row for row in registry["jobs"] if str(row["job_id"]) != str(parent["job_id"])]
                registry["jobs"].append({
                    "branch_id": manifest["branch_id"],
                    "path_id": manifest["path_id"],
                    "image_index_qe": manifest["source_image_index_qe"],
                    "job_name": manifest["job_name"],
                    "job_id": job_id,
                    "parent_job_id": str(parent["job_id"]),
                    "attempt": manifest["attempt"],
                    "seed": manifest["seed"],
                    "remote_run_dir": f"{args.remote_run_root.rstrip('/')}/{manifest['job_name']}_{job_id}",
                })
                atomic_json(args.active_jobs, registry)
            manifest["submission"] = {"state": submission_state, "job_id": job_id}
            atomic_json(manifest_path, manifest)
            run(["rsync", "-a", str(manifest_path), f"{args.ssh_host}:{args.remote_input_dir}/"])
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
