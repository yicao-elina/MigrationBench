#!/usr/bin/env python3
"""Prepare direct/transformed QE relax pairs from one exact NEB path iteration."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from analyze_neb_path_topology import read_qe_image
from parse_qe_neb_output import parse_neb_out
from parse_qe_neb_path_history import parse_path_file
from prepare_geometry_transformed_relaxations import constrained_void_transform, min_host_distance
from prepare_qe_endpoint_relaxations import transform_pw


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def replace_positions(text, symbols, positions):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().upper().startswith("ATOMIC_POSITIONS"))
    if len(lines) < start + 1 + len(symbols):
        raise ValueError("Incomplete ATOMIC_POSITIONS block")
    lines[start] = "ATOMIC_POSITIONS angstrom"
    for offset, (symbol, position) in enumerate(zip(symbols, positions), start=1):
        old = lines[start + offset].split()
        if not old or old[0] != symbol:
            raise ValueError(f"Atom order mismatch at {offset}: expected {symbol}, found {old[:1]}")
        suffix = old[4:]
        lines[start + offset] = "{}  {:.12f}  {:.12f}  {:.12f}".format(symbol, *position)
        if suffix:
            lines[start + offset] += "  " + " ".join(suffix)
    return "\n".join(lines) + "\n"


def write_job(branch_dir, manifest, input_text):
    branch_dir.mkdir(parents=True, exist_ok=True)
    relax_input = branch_dir / "relax.in"
    relax_input.write_text(input_text)
    manifest["relax_input"] = str(relax_input.resolve())
    manifest["relax_input_sha256"] = sha256_file(relax_input)
    manifest_path = branch_dir / "endpoint_relax_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--image", type=int, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-seconds", type=int, default=84600)
    parser.add_argument("--ntasks", type=int, default=2)
    parser.add_argument("--memory", default="160G")
    parser.add_argument("--safe-distance-A", type=float, default=2.80)
    parser.add_argument("--sigma-A", type=float, default=0.25)
    parser.add_argument("--tether", type=float, default=2.0)
    parser.add_argument("--max-shift-A", type=float, default=0.25)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    path_file = run_dir / "out" / f"sb2te3.path{args.iteration}"
    template_path = run_dir / "pw_1.in"
    template = read_qe_image(template_path)
    history = parse_path_file(path_file, len(template["symbols"]))
    if history["iteration"] != args.iteration:
        raise ValueError("Path filename and restart iteration disagree")
    qe_summary = parse_neb_out(run_dir / "neb.out")
    if qe_summary["last_complete_iteration"] != args.iteration:
        raise ValueError(
            f"Requested iteration {args.iteration} is not latest complete neb.out iteration "
            f"{qe_summary['last_complete_iteration']}"
        )
    qe_images = {row["image_index"]: row for row in qe_summary["last_images"]}
    history_images = {row["image_index"]: row for row in history["images"]}
    for image_index, image in history_images.items():
        difference = abs(image["energy_eV"] - qe_images[image_index]["energy_eV"])
        if difference > 2.0e-4:
            raise ValueError(f"Energy provenance mismatch for image {image_index}: {difference} eV")

    resources = {
        "walltime": args.walltime,
        "pw_max_seconds": args.max_seconds,
        "ntasks": args.ntasks,
        "memory": args.memory,
    }
    safe_path = re.sub(r"[^A-Za-z0-9]+", "", args.path_id).lower()
    out_dir = args.out_dir.resolve()
    direct_dir = out_dir / "direct"
    transformed_dir = out_dir / "transformed"
    direct_jobs = []
    transformed_jobs = []
    path_hash = sha256_file(path_file)
    for image_index in args.image:
        image = history_images[image_index]
        positions = [atom["position_A"] for atom in image["atoms"]]
        common = {
            "schema_version": "1.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "source_path_id": args.path_id,
            "source_iteration": args.iteration,
            "source_image_index_qe": image_index,
            "source_path_file": str(path_file),
            "source_path_file_sha256": path_hash,
            "source_energy_eV": image["energy_eV"],
            "source_relative_energy_eV": image["energy_eV"] - history["images"][0]["energy_eV"],
            "source_neb_error_eV_A": qe_images[image_index]["error_eV_A"],
            "template_pw_input": str(template_path),
            "template_pw_sha256": sha256_file(template_path),
            "resources": resources,
            "comparison_gate": {
                "both_bfgs_converged": True,
                "both_final_max_force_eV_A_lte": 0.05,
                "same_final_basin_required": True,
                "primary_metrics": ["ionic_steps", "total_scf_iterations_seen"],
                "secondary_metrics": ["elapsed_seconds"],
            },
        }
        direct_branch = f"{args.path_id}_iter{args.iteration}_img{image_index}_direct_s{args.seed}"
        direct_prefix = f"d_{safe_path}_{args.iteration}_{image_index}_s{args.seed}"
        direct_text = replace_positions(
            transform_pw(template_path.read_text(), direct_prefix, args.max_seconds),
            template["symbols"],
            positions,
        )
        direct_manifest = dict(common)
        direct_manifest.update(
            {
                "branch_id": direct_branch,
                "job_name": f"mb_d{safe_path}t{args.iteration}i{image_index}s{args.seed}",
                "scientific_role": "exact_iteration_direct_qe_relax_control",
                "transform": {"name": "identity", "cr_displacement_A": 0.0},
            }
        )
        direct_jobs.append(write_job(direct_dir / direct_branch, direct_manifest, direct_text))

        cr_indices = [index for index, symbol in enumerate(template["symbols"]) if symbol == "Cr"]
        if len(cr_indices) != 1:
            raise ValueError("Expected exactly one Cr atom")
        cr_index = cr_indices[0]
        hosts = [position for index, position in enumerate(positions) if index != cr_index]
        cr_after, metrics = constrained_void_transform(positions[cr_index], hosts, template["cell"], args)
        metrics["minimum_cr_host_distance_before_A"] = min_host_distance(positions[cr_index], hosts, template["cell"])
        metrics["minimum_cr_host_distance_after_A"] = min_host_distance(cr_after, hosts, template["cell"])
        transformed_positions = list(positions)
        transformed_positions[cr_index] = cr_after
        transformed_branch = f"{args.path_id}_iter{args.iteration}_img{image_index}_geomvoid_s{args.seed}"
        transformed_prefix = f"x_{safe_path}_{args.iteration}_{image_index}_s{args.seed}"
        transformed_text = replace_positions(
            transform_pw(template_path.read_text(), transformed_prefix, args.max_seconds),
            template["symbols"],
            transformed_positions,
        )
        transformed_manifest = dict(common)
        transformed_manifest.update(
            {
                "branch_id": transformed_branch,
                "job_name": f"mb_x{safe_path}t{args.iteration}i{image_index}s{args.seed}",
                "scientific_role": "exact_iteration_geometry_preconditioned_qe_relax",
                "direct_baseline_branch_id": direct_branch,
                "transform": {
                    "name": "constrained_periodic_local_void_projection",
                    "formula": "lambda*|dr|^2 + sum_j softplus((d_safe-d_j)/sigma)^2",
                    "parameters": {
                        "safe_distance_A": args.safe_distance_A,
                        "sigma_A": args.sigma_A,
                        "tether": args.tether,
                        "max_shift_A": args.max_shift_A,
                    },
                    "cr_position_before_A": positions[cr_index],
                    "cr_position_after_A": cr_after,
                    "metrics": metrics,
                },
            }
        )
        transformed_jobs.append(
            write_job(transformed_dir / transformed_branch, transformed_manifest, transformed_text)
        )

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "paired_direct_vs_geometry_transform_from_exact_qe_neb_iteration",
        "path_id": args.path_id,
        "source_iteration": args.iteration,
        "source_path_file": str(path_file),
        "source_path_file_sha256": path_hash,
    }
    for target, jobs in [(direct_dir, direct_jobs), (transformed_dir, transformed_jobs)]:
        payload = {**provenance, "jobs": jobs}
        (target / "endpoint_relax_batch_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "paired_experiment_manifest.json").write_text(
        json.dumps({**provenance, "direct_jobs": direct_jobs, "transformed_jobs": transformed_jobs}, indent=2) + "\n"
    )
    print(json.dumps({"direct": len(direct_jobs), "transformed": len(transformed_jobs), "output": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
