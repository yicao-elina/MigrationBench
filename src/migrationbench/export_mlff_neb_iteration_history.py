#!/usr/bin/env python3
"""Export every recorded MLFF NEB snapshot with true and projected forces."""

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.mep import NEB


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmax(forces):
    return float(np.linalg.norm(np.asarray(forces), axis=1).max())


def calculator_forces(images):
    """Read calculator forces before ASE NEB replaces fixed endpoints by zeros."""
    forces = []
    for image_index, image in enumerate(images):
        try:
            value = np.asarray(image.get_forces(), dtype=float)
        except Exception as error:
            raise ValueError(
                f"Missing true calculator forces for image {image_index}; refusing fabricated zeros"
            ) from error
        if value.shape != (len(image), 3) or not np.isfinite(value).all():
            raise ValueError(f"Invalid true calculator forces for image {image_index}")
        forces.append(value.copy())
    return np.asarray(forces)


def export_history(trajectory, n_images, out_dir, spring_constant, climb, method):
    trajectory = Path(trajectory).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = read(trajectory, index=":")
    if not frames or len(frames) % n_images:
        raise ValueError(
            f"Trajectory has {len(frames)} frames, not complete groups of {n_images}"
        )
    output_frames = []
    rows = []
    iteration_summaries = []
    iterations = len(frames) // n_images
    for iteration in range(iterations):
        images = frames[iteration * n_images:(iteration + 1) * n_images]
        if len({len(image) for image in images}) != 1:
            raise ValueError(f"Atom count changes in optimizer iteration {iteration}")
        true_calculator_forces = calculator_forces(images)
        neb = NEB(images, k=spring_constant, climb=climb, method=method)
        projected = np.asarray(neb.get_forces()).reshape(n_images - 2, len(images[0]), 3)
        energies = np.asarray(neb.energies, dtype=float)
        relative = energies - energies[0]
        residuals = list(neb.residuals)
        iteration_summaries.append({
            "optimizer_iteration": iteration,
            "barrier_proxy_eV": float(relative.max()),
            "max_internal_true_force_eV_A": max(fmax(true_calculator_forces[index]) for index in range(1, n_images - 1)),
            "max_internal_neb_force_eV_A": max(fmax(projected[index]) for index in range(n_images - 2)),
            "max_neb_residual_eV_A": max(float(value) for value in residuals),
        })
        for image_index, atoms in enumerate(images):
            true_forces = true_calculator_forces[image_index]
            exported = atoms.copy()
            exported.calc = None
            exported.arrays["forces"] = true_forces.copy()
            neb_forces = None
            residual = None
            if 0 < image_index < n_images - 1:
                neb_forces = projected[image_index - 1]
                residual = float(residuals[image_index - 1])
                exported.arrays["neb_forces"] = neb_forces.copy()
            exported.info.update({
                "optimizer_iteration": iteration,
                "image_index": image_index,
                "energy_eV": float(energies[image_index]),
                "relative_energy_eV": float(relative[image_index]),
                "true_force_fmax_eV_A": fmax(true_forces),
                "neb_force_available": neb_forces is not None,
                "spring_constant_eV_A2": spring_constant,
                "climb": bool(climb),
                "neb_method": method,
            })
            if neb_forces is not None:
                exported.info["neb_force_fmax_eV_A"] = fmax(neb_forces)
                exported.info["neb_residual_eV_A"] = residual
            output_frames.append(exported)
            rows.append({
                "optimizer_iteration": iteration,
                "image_index": image_index,
                "energy_eV": float(energies[image_index]),
                "relative_energy_eV": float(relative[image_index]),
                "true_force_fmax_eV_A": fmax(true_forces),
                "neb_force_fmax_eV_A": fmax(neb_forces) if neb_forces is not None else None,
                "neb_residual_eV_A": residual,
                "is_endpoint": image_index in {0, n_images - 1},
            })
    extxyz = out_dir / "mlff_neb_iteration_history.extxyz"
    table = out_dir / "mlff_neb_iteration_history.csv"
    write(extxyz, output_frames, format="extxyz")
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    runtime_source = next(
        (parent / "runtime_provenance.json" for parent in (trajectory.parent, *trajectory.parents) if (parent / "runtime_provenance.json").is_file()),
        None,
    )
    runtime_record = None
    if runtime_source is not None:
        runtime_output = out_dir / "runtime_provenance.json"
        shutil.copy2(runtime_source, runtime_output)
        runtime_record = {
            "uri": str(runtime_output),
            "sha256": sha256_file(runtime_output),
            "source_uri": str(runtime_source),
            "source_sha256": sha256_file(runtime_source),
        }
        if runtime_record["sha256"] != runtime_record["source_sha256"]:
            raise RuntimeError("runtime provenance copy hash mismatch")

    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_trajectory": str(trajectory),
        "source_trajectory_sha256": sha256_file(trajectory),
        "n_optimizer_iterations_including_initial": iterations,
        "n_images": n_images,
        "n_atoms": len(frames[0]),
        "spring_constant_eV_A2": spring_constant,
        "climb": bool(climb),
        "neb_method": method,
        "force_semantics": {
            "forces": "true calculator atomic forces read from each trajectory image before ASE NEB projection; fixed endpoints are never taken from zero-filled NEB.real_forces",
            "neb_forces": "projected NEB optimization forces for internal images only; absent on endpoints",
            "neb_residual_eV_A": "ASE NEB residual for each internal image",
        },
        "runtime_provenance": runtime_record,
        "iteration_summaries": iteration_summaries,
        "final_iteration_summary": iteration_summaries[-1],
        "outputs": {
            "extxyz": str(extxyz),
            "extxyz_sha256": sha256_file(extxyz),
            "csv": str(table),
            "csv_sha256": sha256_file(table),
        },
    }
    manifest_path = out_dir / "mlff_neb_iteration_history_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--n-images", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spring-constant", type=float, default=0.1)
    parser.add_argument("--climb", action="store_true")
    parser.add_argument("--method", default="improvedtangent")
    args = parser.parse_args()
    result = export_history(
        args.trajectory, args.n_images, args.out_dir,
        args.spring_constant, args.climb, args.method,
    )
    print(json.dumps({
        "iterations": result["n_optimizer_iterations_including_initial"],
        "images": result["n_images"],
        "output": result["outputs"]["extxyz"],
    }, indent=2))


if __name__ == "__main__":
    main()
