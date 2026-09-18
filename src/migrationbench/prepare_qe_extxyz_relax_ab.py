#!/usr/bin/env python3
"""Prepare paired QE relaxations from direct and transformed extxyz images."""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read

from prepare_qe_endpoint_relaxations import transform_pw
from prepare_qe_path_iteration_relax_pairs import replace_positions


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def frame_sha256(atoms):
    digest = hashlib.sha256()
    digest.update("\n".join(atoms.get_chemical_symbols()).encode())
    digest.update(np.asarray(atoms.cell.array, dtype="<f8").tobytes())
    digest.update(np.asarray(atoms.positions, dtype="<f8").tobytes())
    return digest.hexdigest()


def mic_displacements(left, right):
    scaled = right.get_scaled_positions(wrap=False) - left.get_scaled_positions(wrap=False)
    scaled -= np.rint(scaled)
    return np.dot(scaled, left.cell.array)


def write_branch(root, common, variant, atoms, template_text, prefix, job_name):
    branch_id = f"{common['path_id']}_img{common['source_image_index_qe']}_{variant}_s{common['seed']}"
    branch_dir = root / variant / branch_id
    branch_dir.mkdir(parents=True, exist_ok=True)
    relax_input = branch_dir / "relax.in"
    text = transform_pw(template_text, prefix, common["resources"]["pw_max_seconds"])
    text = replace_positions(text, atoms.get_chemical_symbols(), atoms.positions.tolist())
    relax_input.write_text(text)
    manifest = {
        **common,
        "branch_id": branch_id,
        "job_name": job_name,
        "variant": variant,
        "scientific_role": "direct_qe_relax_control" if variant == "direct" else "geometry_preconditioned_qe_relax",
        "source_frame_sha256": frame_sha256(atoms),
        "relax_input": str(relax_input.resolve()),
    }
    if variant == "direct":
        manifest["transform"] = {
            "name": "identity",
            "cr_displacement_A": 0.0,
            "metrics": {"cr_displacement_A": 0.0},
        }
    manifest["relax_input_sha256"] = sha256_file(relax_input)
    manifest_path = branch_dir / "endpoint_relax_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_path"] = str(manifest_path.resolve())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--transformed", type=Path, required=True)
    parser.add_argument("--transform-manifest", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--image-index-zero", type=int, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-seconds", type=int, default=84600)
    parser.add_argument("--ntasks", type=int, default=2)
    parser.add_argument("--memory", default="160G")
    args = parser.parse_args()

    direct_path, transformed_path = args.direct.resolve(), args.transformed.resolve()
    transform_manifest_path = args.transform_manifest.resolve()
    transform_manifest = json.loads(transform_manifest_path.read_text())
    if transform_manifest.get("output_sha256") != sha256_file(transformed_path):
        raise ValueError("Transformed extxyz is not bound to its repair manifest")
    direct, transformed = read(direct_path, index=":"), read(transformed_path, index=":")
    if len(direct) != len(transformed):
        raise ValueError("Direct/transformed image counts differ")
    if not all(image.pbc.all() and image.cell.rank == 3 for image in direct + transformed):
        raise ValueError("All A/B images must carry a complete periodic cell")
    template_text = args.template.read_text()
    out_dir = args.out_dir.resolve()
    jobs = []
    safe_path = re.sub(r"[^A-Za-z0-9]+", "", args.path_id).lower()
    for image_zero in args.image_index_zero:
        left, right = direct[image_zero], transformed[image_zero]
        if left.get_chemical_symbols() != right.get_chemical_symbols():
            raise ValueError("Direct/transformed atom order differs")
        if not np.allclose(left.cell.array, right.cell.array, atol=1.0e-10):
            raise ValueError("Direct/transformed cells differ")
        cr_indices = [i for i, symbol in enumerate(left.get_chemical_symbols()) if symbol == "Cr"]
        if len(cr_indices) != 1:
            raise ValueError("Expected exactly one Cr migrant")
        cr_index = cr_indices[0]
        displacement = mic_displacements(left, right)
        host_max = float(np.linalg.norm(np.delete(displacement, cr_index, axis=0), axis=1).max())
        cr_shift = float(np.linalg.norm(displacement[cr_index]))
        if host_max > 1.0e-8 or cr_shift <= 1.0e-8:
            raise ValueError(f"A/B pair must change only Cr: host={host_max}, Cr={cr_shift}")
        common = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "path_id": args.path_id,
            "source_path_id": args.path_id,
            "source_image_index_zero_based": image_zero,
            "source_image_index_qe": image_zero + 1,
            "direct_images": str(direct_path),
            "direct_images_sha256": sha256_file(direct_path),
            "transformed_images": str(transformed_path),
            "transformed_images_sha256": sha256_file(transformed_path),
            "transform_manifest": str(transform_manifest_path),
            "transform_manifest_sha256": sha256_file(transform_manifest_path),
            "template_pw_input": str(args.template.resolve()),
            "template_pw_sha256": sha256_file(args.template),
            "transform": {
                "name": "sequential_periodic_hard_clearance_projection",
                "formula": "r_(k+1)=r_k+min(delta,d_min-d_j)*(r_k-r_j)/|r_k-r_j|",
                "cr_displacement_A": cr_shift,
                "maximum_host_displacement_A": host_max,
                "distance_gate_A": transform_manifest["parameters"]["min_pair_distance_A"],
                "metrics": {
                    "cr_displacement_A": cr_shift,
                    "maximum_host_displacement_A": host_max,
                },
            },
            "resources": {
                "walltime": args.walltime,
                "pw_max_seconds": args.max_seconds,
                "ntasks": args.ntasks,
                "memory": args.memory,
            },
            "comparison_gate": {
                "both_bfgs_converged": True,
                "both_final_max_force_eV_A_lte": 0.05,
                "same_final_basin_required": True,
                "primary_metrics": ["ionic_steps", "total_scf_iterations_seen"],
                "secondary_metrics": ["elapsed_seconds"],
                "basin_change_is_not_speedup": True,
            },
        }
        qe_index = image_zero + 1
        jobs.append(write_branch(
            out_dir, common, "direct", left, template_text,
            f"d_{safe_path}_i{qe_index}_s{args.seed}", f"mb_d{safe_path}i{qe_index}s{args.seed}",
        ))
        jobs.append(write_branch(
            out_dir, common, "transformed", right, template_text,
            f"x_{safe_path}_i{qe_index}_s{args.seed}", f"mb_x{safe_path}i{qe_index}s{args.seed}",
        ))
    batch = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "paired_direct_vs_hard_clearance_projection_qe_relax",
        "jobs": jobs,
    }
    (out_dir / "paired_experiment_manifest.json").write_text(json.dumps(batch, indent=2) + "\n")
    print(json.dumps({"prepared": len(jobs), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
