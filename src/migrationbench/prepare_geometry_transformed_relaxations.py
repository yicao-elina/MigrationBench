#!/usr/bin/env python3
"""Create paired QE relax inputs after a constrained Cr local-void transform."""

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_neb_path_topology import displacement, norm, read_qe_image  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def softplus_and_sigmoid(value):
    if value > 30.0:
        return value, 1.0
    if value < -30.0:
        exp_value = math.exp(value)
        return exp_value, exp_value
    exp_value = math.exp(value)
    return math.log1p(exp_value), exp_value / (1.0 + exp_value)


def objective_gradient(position, origin, hosts, cell, tether, safe_distance, sigma):
    delta = displacement(origin, position, cell)
    objective = tether * norm(delta) ** 2
    gradient = [2.0 * tether * value for value in delta]
    for host in hosts:
        vector = displacement(host, position, cell)
        distance = norm(vector)
        if distance < 1.0e-10:
            continue
        z_value = (safe_distance - distance) / sigma
        softplus, sigmoid = softplus_and_sigmoid(z_value)
        objective += softplus * softplus
        derivative_distance = -2.0 * softplus * sigmoid / sigma
        for axis in range(3):
            gradient[axis] += derivative_distance * vector[axis] / distance
    return objective, tuple(gradient)


def constrained_void_transform(position, hosts, cell, args):
    current = tuple(position)
    initial_objective, _ = objective_gradient(
        current, position, hosts, cell, args.tether, args.safe_distance_A, args.sigma_A
    )
    step_size = args.step_size
    iterations = 0
    for iterations in range(1, args.max_iterations + 1):
        objective, gradient = objective_gradient(
            current, position, hosts, cell, args.tether, args.safe_distance_A, args.sigma_A
        )
        if norm(gradient) <= args.gradient_tolerance:
            break
        accepted = False
        trial_step = step_size
        for _ in range(20):
            trial = tuple(current[axis] - trial_step * gradient[axis] for axis in range(3))
            total_delta = displacement(position, trial, cell)
            total_norm = norm(total_delta)
            if total_norm > args.max_shift_A:
                scale = args.max_shift_A / total_norm
                trial = tuple(position[axis] + scale * total_delta[axis] for axis in range(3))
            trial_objective, _ = objective_gradient(
                trial, position, hosts, cell, args.tether, args.safe_distance_A, args.sigma_A
            )
            if trial_objective < objective - 1.0e-12:
                current = trial
                step_size = min(args.step_size, trial_step * 1.2)
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            break
    final_objective, final_gradient = objective_gradient(
        current, position, hosts, cell, args.tether, args.safe_distance_A, args.sigma_A
    )
    return current, {
        "objective_before": initial_objective,
        "objective_after": final_objective,
        "iterations": iterations,
        "final_gradient_norm": norm(final_gradient),
        "cr_displacement_A": norm(displacement(position, current, cell)),
    }


def min_host_distance(position, hosts, cell):
    return min(norm(displacement(position, host, cell)) for host in hosts)


def replace_cr_position(input_path, output_path, new_position, new_prefix):
    lines = input_path.read_text().splitlines()
    in_positions = False
    replaced = 0
    for index, line in enumerate(lines):
        if re.match(r"^\s*prefix\s*=", line, flags=re.IGNORECASE):
            lines[index] = "  prefix = '{}'".format(new_prefix)
        if line.strip().upper().startswith("ATOMIC_POSITIONS"):
            in_positions = True
            continue
        if in_positions:
            fields = line.split()
            if len(fields) < 4:
                in_positions = False
                continue
            try:
                [float(value) for value in fields[1:4]]
            except ValueError:
                in_positions = False
                continue
            if fields[0] == "Cr":
                suffix = fields[4:]
                lines[index] = "Cr  {:.12f}  {:.12f}  {:.12f}".format(*new_position)
                if suffix:
                    lines[index] += "  " + " ".join(suffix)
                replaced += 1
    if replaced != 1:
        raise ValueError("Expected exactly one Cr coordinate in {}, found {}".format(input_path, replaced))
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-batch-manifest", type=Path, required=True)
    parser.add_argument("--submitted-direct-jobs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--safe-distance-A", type=float, default=2.80)
    parser.add_argument("--sigma-A", type=float, default=0.25)
    parser.add_argument("--tether", type=float, default=2.0)
    parser.add_argument("--max-shift-A", type=float, default=0.35)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--minimum-shift-A", type=float, default=1.0e-4)
    parser.add_argument(
        "--arm", choices=("all", "proposal", "control"), default="all",
        help="Optionally transform only one arm of a prospective site batch.",
    )
    args = parser.parse_args()

    direct = json.loads(args.direct_batch_manifest.read_text())
    submitted = json.loads(args.submitted_direct_jobs.read_text())
    job_id_by_branch = {row["branch_id"]: row["job_id"] for row in submitted["jobs"]}
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for source in direct["jobs"]:
        if args.arm != "all" and source.get("arm") != args.arm:
            continue
        input_path = Path(source["relax_input"])
        image = read_qe_image(input_path)
        cr_indices = [index for index, symbol in enumerate(image["symbols"]) if symbol == "Cr"]
        if len(cr_indices) != 1:
            raise ValueError("Expected one Cr in {}".format(input_path))
        cr_index = cr_indices[0]
        cr_before = image["positions"][cr_index]
        hosts = [position for index, position in enumerate(image["positions"]) if index != cr_index]
        cr_after, metrics = constrained_void_transform(cr_before, hosts, image["cell"], args)
        metrics["minimum_cr_host_distance_before_A"] = min_host_distance(cr_before, hosts, image["cell"])
        metrics["minimum_cr_host_distance_after_A"] = min_host_distance(cr_after, hosts, image["cell"])
        if metrics["cr_displacement_A"] < args.minimum_shift_A:
            continue

        source_path = source["source_path_id"]
        image_index = source["source_image_index_qe"]
        safe_path = re.sub(r"[^A-Za-z0-9]+", "", source_path).lower()
        if source.get("arm") and source.get("pair_index") is not None:
            branch_id = "{}_geomvoid".format(source["branch_id"])
            arm_code = "p" if source["arm"] == "proposal" else "c"
            job_name = "mb_s{}x{:02d}_s{}".format(
                arm_code, int(source["pair_index"]), source["seed"]
            )
        else:
            branch_id = "{}_img{}_geomvoid_s{}".format(source_path, image_index, source["seed"])
            job_name = "mb_epx_{}_i{}_s{}".format(safe_path, image_index, source["seed"])
        branch_dir = out_dir / branch_id
        branch_dir.mkdir(parents=True, exist_ok=True)
        relax_input = branch_dir / "relax.in"
        replace_cr_position(
            input_path,
            relax_input,
            cr_after,
            "epx_{}_i{}_s{}".format(safe_path, image_index, source["seed"]),
        )
        manifest = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "branch_id": branch_id,
            "job_name": job_name,
            "seed": source["seed"],
            "scientific_role": "paired_geometry_preconditioned_endpoint_relax",
            "source_path_id": source_path,
            "source_image_index_qe": image_index,
            "source_arm": source.get("arm"),
            "source_pair_index": source.get("pair_index"),
            "source_site_id": source.get("site_id"),
            "source_raw_site_id": source.get("raw_site_id"),
            "direct_baseline_branch_id": source["branch_id"],
            "direct_baseline_job_id": job_id_by_branch[source["branch_id"]],
            "source_pw_input": str(input_path),
            "source_pw_sha256": sha256_file(input_path),
            "transform": {
                "name": "constrained_periodic_local_void_projection",
                "formula": "lambda*|dr|^2 + sum_j softplus((d_safe-d_j)/sigma)^2",
                "parameters": {
                    "safe_distance_A": args.safe_distance_A,
                    "sigma_A": args.sigma_A,
                    "tether": args.tether,
                    "max_shift_A": args.max_shift_A,
                    "step_size": args.step_size,
                    "max_iterations": args.max_iterations,
                    "gradient_tolerance": args.gradient_tolerance,
                },
                "cr_position_before_A": cr_before,
                "cr_position_after_A": cr_after,
                "metrics": metrics,
            },
            "comparison_gate": {
                "same_final_basin_required_for_speedup": True,
                "primary_metrics": ["ionic_steps", "total_scf_iterations", "force_threshold_work"],
                "secondary_metrics": ["walltime", "cpu_hours"],
                "basin_change_is_not_speedup": True,
            },
            "resources": source["resources"],
            "relax_input": str(relax_input),
            "relax_input_sha256": sha256_file(relax_input),
        }
        manifest_path = branch_dir / "endpoint_relax_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        jobs.append({**manifest, "manifest_path": str(manifest_path)})

    batch = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": "paired direct versus geometry-transformed DFT relax",
        "direct_batch_manifest": str(args.direct_batch_manifest.resolve()),
        "source_arm_filter": args.arm,
        "jobs": jobs,
    }
    output = out_dir / "endpoint_relax_batch_manifest.json"
    output.write_text(json.dumps(batch, indent=2) + "\n")
    print(json.dumps({"prepared": len(jobs), "manifest": str(output)}, indent=2))


if __name__ == "__main__":
    main()
