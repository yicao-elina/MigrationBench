#!/usr/bin/env python3
"""Sync and classify MLFF NEB jobs with provenance and history gates."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read

from repair_neb_images import choose_migrant, path_stats


ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
PROVENANCE_CHECKS = (
    "input_hash",
    "model_hash",
    "candidate_manifest_hash",
    "runtime_identity_complete",
    "runtime_input_hashes",
    "runtime_code_snapshot",
    "output_images_hash",
    "trajectory_hash",
    "history_output_hashes",
    "history_complete",
    "dual_potential_history",
    "protocol_match",
)


def sha256_file(path):
    if not path or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_artifact(base, recorded):
    if not recorded:
        return None
    path = Path(recorded)
    if path.is_file():
        return path
    candidate = base / path.name
    return candidate if candidate.is_file() else None


def hash_matches(path, expected):
    return bool(path and expected and sha256_file(path) == expected)


def command(argv, check=True):
    result = subprocess.run(argv, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError("command failed: {}\n{}".format(" ".join(argv), result.stderr))
    return result


def accounting(job_id, host):
    result = command([
        "ssh", host, "sacct", "-X", "-j", str(job_id),
        "--format=JobID,JobName%40,State,ExitCode,Elapsed,MaxRSS,ReqMem", "-n", "-P",
    ], check=False)
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 7:
            rows.append(dict(zip(("JobID", "JobName", "State", "ExitCode", "Elapsed", "MaxRSS", "ReqMem"), fields)))
    main = next((row for row in rows if row["JobID"] == str(job_id)), rows[0] if rows else {})
    return main, rows


def first_header_has_pbc(path):
    if not path.exists():
        return None
    with path.open(errors="replace") as handle:
        handle.readline()
        header = handle.readline()
    return 'pbc="T T T"' in header and "Lattice=" in header


def expected_jobs(config_paths):
    """Normalize paired and single-job experiment configs by Slurm job id."""
    expected = {}
    for config_path in config_paths or []:
        experiment = json.loads(config_path.read_text())
        candidate_path = Path(experiment["candidate_manifest"]) if experiment.get("candidate_manifest") else None
        candidate_payload = None
        if candidate_path and candidate_path.is_file():
            if sha256_file(candidate_path) != experiment.get("candidate_manifest_sha256"):
                raise ValueError(f"candidate manifest hash drift: {candidate_path}")
            candidate_payload = json.loads(candidate_path.read_text())
        candidate_rows = {
            row["branch_id"]: row for row in (candidate_payload or {}).get("candidates", [])
        }
        rows = experiment.get("jobs")
        if rows is None and experiment.get("job_id") is not None:
            rows = [{
                "job_id": experiment["job_id"],
                "job_name": experiment.get("job_name"),
                "path_id": experiment.get("path_id"),
                "input_sha256": experiment.get("input_images_sha256"),
            }]
        for row in rows or []:
            normalized = dict(row)
            for key in (
                "seed",
                "calculator",
                "steps",
                "fmax_target_eV_A",
                "spring_constant_eV_A2",
                "model_path",
                "migrant_tether_k_eV_A2",
                "host_tether_k_eV_A2",
            ):
                if normalized.get(key) is None:
                    normalized[key] = experiment.get(key)
            normalized["input_sha256"] = row.get("input_sha256") or row.get("local_images_sha256")
            normalized["model_path_sha256"] = (
                row.get("model_path_sha256") or experiment.get("model_path_sha256")
            )
            normalized["candidate_manifest_sha256"] = (
                row.get("candidate_manifest_sha256")
                or experiment.get("candidate_manifest_sha256")
            )
            normalized["candidate"] = candidate_rows.get(row.get("branch_id"))
            normalized["migrant_element"] = (
                row.get("migrant_element")
                or (normalized["candidate"] or {}).get("migrant_element")
                or experiment.get("migrant_element")
            )
            job_id = str(normalized["job_id"])
            if job_id in expected:
                raise ValueError(f"job {job_id} appears in multiple experiment configs")
            expected[job_id] = normalized
    return expected


def mic_displacements(left, right):
    if len(left) != len(right):
        raise ValueError("Atom count changed")
    if left.get_chemical_symbols() != right.get_chemical_symbols():
        raise ValueError("Atom order changed")
    if left.cell.rank == 3 and right.cell.rank == 3 and any(left.pbc):
        if not np.allclose(left.cell.array, right.cell.array, atol=1.0e-8):
            raise ValueError("Cell changed during fixed-cell MLFF NEB")
        delta = right.get_scaled_positions(wrap=False) - left.get_scaled_positions(wrap=False)
        delta -= np.rint(delta)
        return np.dot(delta, left.cell.array)
    return right.positions - left.positions


def path_change_metrics(input_path, output_path, migrant_element="Cr"):
    initial = read(input_path, index=":")
    final = read(output_path, index=":")
    if len(initial) != len(final) or len(initial) < 4:
        raise ValueError("Initial/final image count mismatch or fewer than four images")
    migrant = choose_migrant(initial, migrant_element)
    displacements = [mic_displacements(left, right) for left, right in zip(initial, final)]
    internal = displacements[1:-1]
    all_atom_rms = float(np.sqrt(np.mean(np.square(np.asarray(internal)))))
    migrant_rms = float(np.sqrt(np.mean([
        np.dot(delta[migrant], delta[migrant]) for delta in internal
    ])))
    endpoint_max = max(
        float(np.linalg.norm(displacements[0], axis=1).max()),
        float(np.linalg.norm(displacements[-1], axis=1).max()),
    )
    return {
        "initial_geometry": path_stats(initial, migrant),
        "final_geometry": path_stats(final, migrant),
        "all_atom_internal_coordinate_rms_A": all_atom_rms,
        "migrant_internal_coordinate_rms_A": migrant_rms,
        "endpoint_max_displacement_A": endpoint_max,
    }


def terminal_evidence(manifest_path, manifest, history, expected_row):
    base = manifest_path.parent
    outputs = manifest.get("outputs", {})
    history_outputs = (history or {}).get("outputs", {})
    output_images = local_artifact(base, outputs.get("images"))
    trajectory = local_artifact(base, outputs.get("trajectory"))
    history_extxyz = local_artifact(base, history_outputs.get("extxyz"))
    history_csv = local_artifact(base, history_outputs.get("csv"))
    runtime_record = (history or {}).get("runtime_provenance") or {}
    runtime_path = local_artifact(base, runtime_record.get("uri"))
    runtime = json.loads(runtime_path.read_text()) if runtime_path else {}
    runtime_inputs = runtime.get("inputs", {})
    tether = manifest.get("reference_tether", {})
    tether_enabled = tether.get("enabled") is True
    dual_extxyz = local_artifact(base, outputs.get("dual_potential_history_extxyz"))
    dual_csv = local_artifact(base, outputs.get("dual_potential_history_csv"))
    dual_manifest_path = local_artifact(base, outputs.get("dual_potential_history_manifest"))
    dual_manifest = (
        json.loads(dual_manifest_path.read_text())
        if dual_manifest_path and dual_manifest_path.is_file() else None
    )
    dual_csv_rows = None
    if dual_csv and dual_csv.is_file():
        with dual_csv.open(errors="replace") as handle:
            dual_csv_rows = max(sum(1 for _ in handle) - 1, 0)
    expected_input = expected_row.get("input_sha256")
    expected_model = expected_row.get("model_path_sha256")
    expected_candidate = expected_row.get("candidate_manifest_sha256")
    expected_code = expected_row.get("submitted_code_snapshot", {}).get("sha256", {})
    checks = {
        "input_hash": bool(expected_input)
        and manifest.get("input_images_sha256") == expected_input,
        "model_hash": bool(expected_model)
        and manifest.get("model_path_sha256") == expected_model,
        "candidate_manifest_hash": bool(expected_candidate)
        and manifest.get("source_candidate_manifest_sha256") == expected_candidate,
        "runtime_identity_complete": bool(runtime_path)
        and hash_matches(runtime_path, runtime_record.get("sha256"))
        and bool(runtime.get("python", {}).get("executable"))
        and bool(runtime.get("generator", {}).get("sha256"))
        and bool(runtime.get("packages", {}).get("mace-torch", {}).get("module_file_sha256")),
        "runtime_input_hashes": bool(runtime_inputs)
        and runtime_inputs.get("images", {}).get("sha256") == expected_input
        and runtime_inputs.get("model", {}).get("sha256") == expected_model
        and runtime_inputs.get("source_manifest", {}).get("sha256") == expected_candidate,
        "runtime_code_snapshot": (not expected_code) or (
            runtime_inputs.get("submitted_runner", {}).get("sha256")
            == expected_code.get("run_mlff_neb.py")
            and runtime_inputs.get("submitted_wrapper", {}).get("sha256")
            == expected_code.get("submit_slurm_mlff_neb_cpu.sh")
            and runtime_inputs.get("submitted_history_exporter", {}).get("sha256")
            == expected_code.get("export_mlff_neb_iteration_history.py")
            and bool(runtime_inputs.get("submitted_code_checksums", {}).get("sha256"))
        ),
        "output_images_hash": hash_matches(output_images, outputs.get("images_sha256")),
        "trajectory_hash": hash_matches(trajectory, (history or {}).get("source_trajectory_sha256")),
        "history_output_hashes": hash_matches(history_extxyz, history_outputs.get("extxyz_sha256"))
        and hash_matches(history_csv, history_outputs.get("csv_sha256")),
        "history_complete": bool(history)
        and len(history.get("iteration_summaries", []))
        == int(manifest.get("optimizer_steps_completed", -1)) + 1
        and manifest.get("optimizer_iterations_recorded_including_initial")
        == len(history.get("iteration_summaries", [])),
        "dual_potential_history": (not tether_enabled) or (
            hash_matches(dual_extxyz, outputs.get("dual_potential_history_extxyz_sha256"))
            and hash_matches(dual_csv, outputs.get("dual_potential_history_csv_sha256"))
            and dual_csv_rows
            == (int(manifest.get("optimizer_steps_completed", -1)) + 1)
            * int(manifest.get("n_images", 0))
            and (
                outputs.get("dual_potential_history_manifest") is None
                or (
                    hash_matches(
                        dual_manifest_path,
                        outputs.get("dual_potential_history_manifest_sha256"),
                    )
                    and dual_manifest.get("complete") is True
                    and dual_manifest.get("reported_barrier_source")
                    == "base_MACE_without_reference_tether"
                    and dual_manifest.get("row_count") == dual_csv_rows
                )
            )
        ),
        "protocol_match": bool(expected_row)
        and manifest.get("seed") == expected_row.get("seed")
        and manifest.get("calculator") == expected_row.get("calculator")
        and manifest.get("steps_requested") == expected_row.get("steps")
        and abs(float(manifest.get("fmax_target_eV_A", np.inf)) - float(expected_row.get("fmax_target_eV_A", np.nan))) < 1.0e-12
        and abs(float((history or {}).get("spring_constant_eV_A2", np.inf)) - float(expected_row.get("spring_constant_eV_A2", np.nan))) < 1.0e-12,
    }
    if tether_enabled or expected_row.get("migrant_tether_k_eV_A2", 0) or expected_row.get("host_tether_k_eV_A2", 0):
        checks["protocol_match"] = checks["protocol_match"] and (
            abs(float(tether.get("migrant_k_eV_A2", np.inf)) - float(expected_row.get("migrant_tether_k_eV_A2", 0))) < 1.0e-12
            and abs(float(tether.get("host_k_eV_A2", np.inf)) - float(expected_row.get("host_tether_k_eV_A2", 0))) < 1.0e-12
        )
    metrics = {}
    if output_images and expected_row.get("local_images") and Path(expected_row["local_images"]).is_file():
        migrant_element = (
            expected_row.get("migrant_element")
            or (expected_row.get("candidate") or {}).get("migrant_element")
            or manifest.get("reference_tether", {}).get("migrant_element")
            or "Cr"
        )
        metrics = path_change_metrics(Path(expected_row["local_images"]), output_images, migrant_element)
    internal_energy_changes = [
        abs(float(row["delta_energy_eV"]))
        for row in manifest.get("relaxation_energy_delta", [])[1:-1]
    ]
    metrics["max_abs_internal_relaxation_energy_delta_eV"] = (
        max(internal_energy_changes) if internal_energy_changes else None
    )
    final_geometry = metrics.get("final_geometry", {})
    checks.update({
        "final_geometry": bool(final_geometry)
        and final_geometry.get("min_pair_distance_A", -np.inf) >= 1.80
        and final_geometry.get("max_migrant_step_A", np.inf) <= 2.50,
        "endpoints_preserved": metrics.get("endpoint_max_displacement_A", np.inf) <= 1.0e-6,
        "no_large_energy_collapse": metrics.get(
            "max_abs_internal_relaxation_energy_delta_eV", np.inf
        ) <= 5.0,
    })
    return checks, metrics


def classify(slurm_state, manifest, checks):
    state = slurm_state.split("+", 1)[0]
    if state in ACTIVE:
        return state.lower()
    if manifest is None:
        return "terminal_missing_manifest"
    if checks.get("periodic_output") is not True:
        return "invalid_nonperiodic_output"
    if not all(checks.get(name) is True for name in PROVENANCE_CHECKS):
        return "invalid_provenance_or_history"
    if state != "COMPLETED":
        return "terminal_scheduler_failure"
    if not checks.get("final_geometry") or not checks.get("endpoints_preserved"):
        return "terminal_geometry_failure"
    if not checks.get("no_large_energy_collapse"):
        return "terminal_basin_collapse"
    if manifest.get("reference_tether", {}).get("enabled") is True:
        if manifest.get("optimizer_converged_under_optimization_potential"):
            return "ready_for_dft_preconditioner_review"
        return "terminal_restrained_unconverged"
    if not (
        manifest.get("optimizer_converged")
        and manifest.get("final_max_internal_neb_force_eV_A", float("inf"))
        <= manifest["fmax_target_eV_A"]
    ):
        return "terminal_unconverged"
    return "ready_for_mechanism_clustering"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, action="append", default=[])
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--remote-run-root", default="/scratch16/pclancy3/yi/revision1_migrationbench_runs")
    parser.add_argument("--local-cluster-root", type=Path, default=Path("./cluster"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs_file.read_text())["jobs"]
    expected = expected_jobs(args.experiment_config)
    results = []
    for job in jobs:
        job_id = str(job["job_id"])
        main, accounting_rows = accounting(job_id, args.ssh_host)
        state = main.get("State", "UNKNOWN").split("+", 1)[0]
        remote_dir = f"{args.remote_run_root.rstrip('/')}/{job['job_name']}_{job_id}"
        local_dir = args.local_cluster_root / Path(remote_dir).name
        local_dir.mkdir(parents=True, exist_ok=True)
        if state in ACTIVE:
            include = [
                "--include", "*/", "--include", "*.json", "--include", "*.csv",
                "--include", "*.log", "--include", "slurm-*.out", "--include", "slurm-*.err",
                "--include", "mlff_neb_images.extxyz", "--exclude", "*",
            ]
            command(["rsync", "-a", *include, f"{args.ssh_host}:{remote_dir}/", str(local_dir) + "/"], check=False)
        else:
            command(["rsync", "-a", f"{args.ssh_host}:{remote_dir}/", str(local_dir) + "/"], check=False)
        manifests = sorted(local_dir.rglob("mlff_neb_manifest.json"))
        manifest_path = manifests[0] if len(manifests) == 1 else None
        manifest = json.loads(manifest_path.read_text()) if manifest_path else None
        history_path = manifest_path.with_name("mlff_neb_iteration_history_manifest.json") if manifest_path else None
        history = json.loads(history_path.read_text()) if history_path and history_path.exists() else None
        expected_row = expected.get(job_id, {})
        output_images = manifest_path.with_name("mlff_neb_images.extxyz") if manifest_path else Path("missing")
        checks = {
            "single_manifest": len(manifests) == 1,
            "periodic_output": first_header_has_pbc(output_images) if manifest else None,
        }
        mechanism_metrics = {}
        if manifest and history and state not in ACTIVE:
            terminal_checks, mechanism_metrics = terminal_evidence(
                manifest_path, manifest, history, expected_row
            )
            checks.update(terminal_checks)
        results.append({
            **job,
            "remote_run_dir": remote_dir,
            "local_dir": str(local_dir),
            "sacct_main": main,
            "sacct_rows": accounting_rows,
            "manifest": str(manifest_path) if manifest_path else None,
            "checks": checks,
            "classification": classify(state, manifest, checks),
            "mechanism_metrics": mechanism_metrics,
            "metrics": ({
                "optimizer_steps_completed": manifest.get("optimizer_steps_completed"),
                "optimizer_converged": manifest.get("optimizer_converged"),
                "optimizer_converged_under_optimization_potential": manifest.get("optimizer_converged_under_optimization_potential"),
                "reference_tether": manifest.get("reference_tether"),
                "final_max_internal_true_force_eV_A": manifest.get("final_max_internal_true_force_eV_A"),
                "final_max_internal_neb_force_eV_A": manifest.get("final_max_internal_neb_force_eV_A"),
                "barrier_proxy_eV": manifest.get("barrier_proxy_eV"),
                "max_abs_internal_relaxation_energy_delta_eV": mechanism_metrics.get(
                    "max_abs_internal_relaxation_energy_delta_eV"
                ),
            } if manifest else {}),
        })
    payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"jobs": len(results), "classifications": {row["job_id"]: row["classification"] for row in results}}, indent=2))


if __name__ == "__main__":
    main()
