#!/usr/bin/env python3
"""Rescore a historical ASE NEB band with one calculator identity.

This repairs the *audit representation*, not the historical optimization. If
the original band mixed calculator identities, the rescored result remains a
diagnostic and must not be relabeled as a valid converged NEB run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.mep import NEB
from mace.calculators import MACECalculator


STEP_RE = re.compile(r"^BFGS:\s+(\d+)\s+\S+\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_fmax(forces) -> float:
    array = np.asarray(forces, dtype=float)
    return float(np.linalg.norm(array, axis=1).max())


def parse_optimizer_log(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = STEP_RE.match(line.strip())
        if match:
            rows.append(
                {
                    "iteration": int(match.group(1)),
                    "optimizer_band_energy_eV": float(match.group(2)),
                    "optimizer_neb_fmax_eV_A": float(match.group(3)),
                }
            )
    return rows


def barrier_metrics(energies: list[float]) -> dict:
    peak = int(np.argmax(energies))
    minimum = int(np.argmin(energies))
    maximum = float(energies[peak])
    return {
        "forward_barrier_eV": maximum - float(energies[0]),
        "reverse_barrier_eV": maximum - float(energies[-1]),
        "energy_span_eV": maximum - float(energies[minimum]),
        "endpoint_delta_eV": float(energies[-1]) - float(energies[0]),
        "peak_image_index_qe": peak + 1,
        "minimum_image_index_qe": minimum + 1,
        "barrier_shape": "endpoint_dominated" if peak in {0, len(energies) - 1} else "interior_peak",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--optimizer-log", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-images", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory = Trajectory(args.trajectory)
    if len(trajectory) % args.n_images:
        raise ValueError("trajectory frame count is not divisible by n_images")
    n_iterations = len(trajectory) // args.n_images
    source_band = [trajectory[len(trajectory) - args.n_images + index].copy() for index in range(args.n_images)]

    source_energies = []
    source_calculators = []
    for index in range(args.n_images):
        frame = trajectory[len(trajectory) - args.n_images + index]
        source_energies.append(float(frame.get_potential_energy()))
        source_calculators.append(type(frame.calc).__name__ if frame.calc else None)

    calculator = MACECalculator(
        model_paths=str(args.model),
        device=args.device,
        default_dtype="float64",
    )
    rescored = []
    for source in source_band:
        evaluation = source.copy()
        evaluation.calc = calculator
        energy = float(evaluation.get_potential_energy())
        forces = np.asarray(evaluation.get_forces(), dtype=float)
        saved = source.copy()
        saved.calc = SinglePointCalculator(saved, energy=energy, forces=forces)
        rescored.append(saved)

    energies = [float(image.get_potential_energy()) for image in rescored]
    atomic_fmax = [vector_fmax(image.get_forces()) for image in rescored]
    neb = NEB(rescored, climb=True)
    projected = np.asarray(neb.get_forces(), dtype=float)
    reconstructed_neb_fmax = vector_fmax(projected)
    metrics = barrier_metrics(energies)

    optimizer_rows = parse_optimizer_log(args.optimizer_log)
    optimizer_final = optimizer_rows[-1] if optimizer_rows else None
    mixed_scale = max(source_energies) - min(source_energies) > 1000.0
    source_endpoint_internal_mismatch = mixed_scale and abs(source_energies[0] - source_energies[1]) > 1000.0

    image_rows = []
    for index, (energy, fmax) in enumerate(zip(energies, atomic_fmax), start=1):
        image_rows.append(
            {
                "image_index_qe": index,
                "rescored_energy_eV": energy,
                "relative_to_initial_eV": energy - energies[0],
                "relative_to_minimum_eV": energy - min(energies),
                "atomic_fmax_eV_A": fmax,
                "source_stored_energy_eV": source_energies[index - 1],
                "source_calculator_class": source_calculators[index - 1],
                "frozen_endpoint": index in {1, args.n_images},
            }
        )

    extxyz = args.output_dir / "foundation_self_neb_final_band_rescored.extxyz"
    write(extxyz, rescored, format="extxyz")
    csv_path = args.output_dir / "foundation_self_neb_final_band_rescored.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]))
        writer.writeheader()
        writer.writerows(image_rows)

    report = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_images": args.n_images,
        "n_optimizer_iterations_in_trajectory": n_iterations,
        "trajectory": str(args.trajectory),
        "trajectory_sha256": sha256_file(args.trajectory),
        "optimizer_log": str(args.optimizer_log),
        "optimizer_log_sha256": sha256_file(args.optimizer_log),
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "source_final_band_stored_energies_eV": source_energies,
        "source_final_band_calculator_classes": source_calculators,
        "source_endpoint_internal_energy_scale_mismatch": source_endpoint_internal_mismatch,
        "rescored_image_energies_eV": energies,
        "rescored_atomic_fmax_eV_A": atomic_fmax,
        "reconstructed_neb_fmax_eV_A": reconstructed_neb_fmax,
        "historical_optimizer_final": optimizer_final,
        **metrics,
        "scientific_status": "diagnostic_invalid_historical_mixed_calculator_optimization"
        if source_endpoint_internal_mismatch
        else "diagnostic_rescored_historical_band",
        "interpretation": (
            "Uniform rescoring repairs the final energy profile only. It cannot repair an optimization whose "
            "NEB tangent/path evolution used endpoint and internal energies from incompatible calculators."
        ),
    }
    report_path = args.output_dir / "foundation_self_neb_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
