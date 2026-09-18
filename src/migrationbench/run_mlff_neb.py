#!/usr/bin/env python3
"""Run a coarse NEB with an ASE calculator and export an auditable image path.

Use this on rockfish with --calculator mace-foundation or --calculator mace-model
to precondition slow DFT NEB jobs. The --calculator emt mode is for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE

from export_mlff_neb_iteration_history import export_history


class ReferenceTetherCalculator(Calculator):
    """Add a PBC-aware harmonic trust region around one reference image."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, base, reference, force_constants):
        super().__init__()
        self.base = base
        self.reference = reference.copy()
        self.force_constants = np.asarray(force_constants, dtype=float)
        self.base_results = {}
        self.tether_energy_eV = 0.0
        self.max_displacement_A = 0.0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.base.calculate(atoms, ["energy", "forces"], system_changes)
        base_energy = float(self.base.results["energy"])
        base_forces = np.asarray(self.base.results["forces"], dtype=float).copy()
        delta = atoms.positions - self.reference.positions
        if atoms.cell.rank == 3 and any(atoms.pbc):
            fractional = np.linalg.solve(atoms.cell.array.T, delta.T).T
            fractional -= np.rint(fractional)
            delta = fractional @ atoms.cell.array
        weighted = self.force_constants[:, None] * delta
        tether_energy = 0.5 * float(np.sum(weighted * delta))
        self.base_results = {"energy": base_energy, "forces": base_forces}
        self.tether_energy_eV = tether_energy
        self.max_displacement_A = float(np.linalg.norm(delta, axis=1).max())
        self.results = {
            "energy": base_energy + tether_energy,
            "free_energy": base_energy + tether_energy,
            "forces": base_forces - weighted,
        }


def sha256_file(path: Path) -> str | None:
    if not path or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_calculator(args: argparse.Namespace):
    if args.calculator == "emt":
        from ase.calculators.emt import EMT

        return EMT(), "EMT"

    if args.calculator == "mace-foundation":
        from mace.calculators import mace_mp
        if not args.model_path:
            raise ValueError(
                "--model-path is required for reproducible mace-foundation runs; "
                "do not rely on the version-dependent mace_mp default"
            )
        resolved = str(Path(args.model_path).resolve())
        return mace_mp(
            model=resolved, device=args.device, default_dtype=args.default_dtype
        ), resolved

    if args.calculator == "mace-model":
        from mace.calculators import MACECalculator

        if not args.model_path:
            raise ValueError("--model-path is required for --calculator mace-model")
        try:
            return MACECalculator(model_paths=args.model_path, device=args.device), args.model_path
        except TypeError:
            return MACECalculator(model_path=args.model_path, device=args.device), args.model_path

    raise ValueError(f"Unsupported calculator: {args.calculator}")


def software_versions():
    versions = {}
    for distribution in ("ase", "mace-torch", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def load_or_interpolate_images(args: argparse.Namespace):
    if args.images:
        images = read(args.images, index=":")
        if len(images) < 4:
            raise ValueError("NEB needs at least 4 images.")
        return images

    if not (args.initial and args.final):
        raise ValueError("Provide either --images or both --initial and --final.")
    initial = read(args.initial)
    final = read(args.final)
    images = [initial.copy()]
    images += [initial.copy() for _ in range(args.n_images - 2)]
    images += [final.copy()]
    neb = NEB(images, climb=False, method="improvedtangent")
    neb.interpolate(method=args.interpolate_method, mic=args.mic)
    return images


def energy_profile(images) -> list[dict]:
    energies = [float(atoms.get_potential_energy()) for atoms in images]
    e0 = energies[0]
    return [
        {
            "image_index": i,
            "energy_eV": e,
            "rel_energy_eV": e - e0,
            "fmax_eV_A": float(np.linalg.norm(atoms.get_forces(), axis=1).max()),
        }
        for i, (atoms, e) in enumerate(zip(images, energies))
    ]


def base_energy_profile(images) -> list[dict]:
    rows = []
    energies = []
    forces = []
    for atoms in images:
        calc = atoms.calc
        atoms.get_potential_energy()
        atoms.get_forces()
        if isinstance(calc, ReferenceTetherCalculator):
            energies.append(float(calc.base_results["energy"]))
            forces.append(np.asarray(calc.base_results["forces"], dtype=float))
        else:
            energies.append(float(atoms.get_potential_energy()))
            forces.append(np.asarray(atoms.get_forces(), dtype=float))
    e0 = energies[0]
    for index, (energy, atomic_forces) in enumerate(zip(energies, forces)):
        rows.append({
            "image_index": index,
            "energy_eV": energy,
            "rel_energy_eV": energy - e0,
            "fmax_eV_A": float(np.linalg.norm(atomic_forces, axis=1).max()),
        })
    return rows


def attach_dual_potential_history(opt, images, out_dir):
    """Record base-MACE and restrained quantities at every FIRE iteration."""
    extxyz = out_dir / "mlff_neb_dual_potential_history.extxyz"
    csv_path = out_dir / "mlff_neb_dual_potential_history.csv"
    if extxyz.exists():
        extxyz.unlink()
    fields = [
        "optimizer_iteration", "image_index", "base_energy_eV",
        "restrained_energy_eV", "tether_energy_eV", "base_fmax_eV_A",
        "restrained_fmax_eV_A", "max_reference_displacement_A",
    ]
    handle = csv_path.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    seen = set()

    def record():
        iteration = int(opt.nsteps)
        if iteration in seen:
            return
        seen.add(iteration)
        frames = []
        for image_index, atoms in enumerate(images):
            restrained_energy = float(atoms.get_potential_energy())
            restrained_forces = np.asarray(atoms.get_forces(), dtype=float)
            calc = atoms.calc
            if isinstance(calc, ReferenceTetherCalculator):
                base_energy = float(calc.base_results["energy"])
                base_forces = np.asarray(calc.base_results["forces"], dtype=float)
                tether_energy = float(calc.tether_energy_eV)
                max_displacement = float(calc.max_displacement_A)
            else:
                base_energy = restrained_energy
                base_forces = restrained_forces.copy()
                tether_energy = 0.0
                max_displacement = 0.0
            frame = atoms.copy()
            frame.calc = None
            frame.info.update({
                "optimizer_iteration": iteration,
                "neb_image": image_index,
                "base_energy_eV": base_energy,
                "restrained_energy_eV": restrained_energy,
                "tether_energy_eV": tether_energy,
                "max_reference_displacement_A": max_displacement,
            })
            frame.arrays["base_forces"] = base_forces
            frame.arrays["restrained_forces"] = restrained_forces
            frames.append(frame)
            writer.writerow({
                "optimizer_iteration": iteration,
                "image_index": image_index,
                "base_energy_eV": base_energy,
                "restrained_energy_eV": restrained_energy,
                "tether_energy_eV": tether_energy,
                "base_fmax_eV_A": float(np.linalg.norm(base_forces, axis=1).max()),
                "restrained_fmax_eV_A": float(np.linalg.norm(restrained_forces, axis=1).max()),
                "max_reference_displacement_A": max_displacement,
            })
        write(extxyz, frames, append=True)
        handle.flush()

    opt.attach(record, interval=1)
    return record, handle, extxyz, csv_path


def finalize_dual_history_manifest(
    out_dir, extxyz_path, csv_path, n_images, optimizer_steps, tether
):
    with csv_path.open(errors="replace") as handle:
        row_count = max(sum(1 for _ in handle) - 1, 0)
    expected_rows = (int(optimizer_steps) + 1) * int(n_images)
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_role": "dual_potential_trust_region_neb",
        "optimizer_coordinate_source": "base_MACE_plus_reference_tether",
        "physical_energy_force_source": "base_MACE_without_reference_tether",
        "reported_barrier_source": "base_MACE_without_reference_tether",
        "reference_tether": tether,
        "n_images": int(n_images),
        "n_optimizer_iterations_including_initial": int(optimizer_steps) + 1,
        "row_count": row_count,
        "expected_row_count": expected_rows,
        "complete": row_count == expected_rows,
        "outputs": {
            "extxyz": str(extxyz_path),
            "extxyz_sha256": sha256_file(extxyz_path),
            "csv": str(csv_path),
            "csv_sha256": sha256_file(csv_path),
        },
        "field_semantics": {
            "base_energy_eV": "energy from the unmodified base calculator",
            "base_forces": "atomic forces from the unmodified base calculator",
            "restrained_energy_eV": "base energy plus harmonic reference tether",
            "restrained_forces": "base forces minus the tether gradient",
            "tether_energy_eV": "nonphysical optimization regularizer; never a barrier label",
        },
    }
    output = out_dir / "mlff_neb_dual_potential_history_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload, output


def profile_energy_delta(initial: list[dict], final: list[dict]) -> list[dict]:
    if len(initial) != len(final):
        return []
    return [
        {
            "image_index": after["image_index"],
            "initial_energy_eV": before["energy_eV"],
            "final_energy_eV": after["energy_eV"],
            "delta_energy_eV": after["energy_eV"] - before["energy_eV"],
            "initial_rel_energy_eV": before["rel_energy_eV"],
            "final_rel_energy_eV": after["rel_energy_eV"],
            "delta_rel_energy_eV": after["rel_energy_eV"] - before["rel_energy_eV"],
            "initial_fmax_eV_A": before["fmax_eV_A"],
            "final_fmax_eV_A": after["fmax_eV_A"],
        }
        for before, after in zip(initial, final)
    ]


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images = load_or_interpolate_images(args)
    references = [atoms.copy() for atoms in images]
    built = [build_calculator(args) for _ in images]
    calculators = [item[0] for item in built]
    calc_label = built[0][1]
    tether_enabled = args.migrant_tether_k > 0 or args.host_tether_k > 0
    if args.migrant_tether_k < 0 or args.host_tether_k < 0:
        raise ValueError("Reference-tether force constants must be non-negative")
    migrant_indices = [
        [index for index, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == args.migrant_element]
        for atoms in images
    ]
    if tether_enabled and any(len(indices) != 1 for indices in migrant_indices):
        raise ValueError(f"Each image must contain exactly one {args.migrant_element} for tethered NEB")
    for image_index, atoms in enumerate(images):
        if tether_enabled and 0 < image_index < len(images) - 1:
            constants = np.full(len(atoms), args.host_tether_k, dtype=float)
            constants[migrant_indices[image_index][0]] = args.migrant_tether_k
            atoms.calc = ReferenceTetherCalculator(
                calculators[image_index], references[image_index], constants
            )
        else:
            atoms.calc = calculators[image_index]

    initial_profile = base_energy_profile(images)

    neb = NEB(
        images,
        climb=args.climb,
        k=args.spring_constant,
        method="improvedtangent",
        remove_rotation_and_translation=args.remove_rotation_translation,
    )
    opt = FIRE(neb, trajectory=str(out_dir / "mlff_neb.traj"), logfile=str(out_dir / "mlff_neb.log"))
    _, dual_history_handle, dual_history_extxyz, dual_history_csv = attach_dual_potential_history(
        opt, images, out_dir
    )
    try:
        opt.run(fmax=args.fmax, steps=args.steps)
    finally:
        dual_history_handle.close()
    optimizer_converged_under_optimization_potential = bool(opt.converged())
    optimizer_converged = optimizer_converged_under_optimization_potential and not tether_enabled

    iteration_history = export_history(
        out_dir / "mlff_neb.traj", len(images), out_dir,
        args.spring_constant, args.climb, "improvedtangent",
    )

    tether_config = {
        "enabled": tether_enabled,
        "migrant_element": args.migrant_element,
        "migrant_k_eV_A2": args.migrant_tether_k,
        "host_k_eV_A2": args.host_tether_k,
        "formula": "E_opt=E_base+0.5*sum_a k_a*||MIC(r_a-r_a_ref)||^2",
    }
    history_manifest_path = out_dir / "mlff_neb_iteration_history_manifest.json"
    history_manifest = json.loads(history_manifest_path.read_text())
    history_manifest["potential_semantics"] = {
        "energy_and_forces": "optimization_potential",
        "optimization_potential": (
            "base_MACE_plus_reference_tether" if tether_enabled else "base_calculator"
        ),
        "physical_base_quantities": (
            "see mlff_neb_dual_potential_history_manifest.json"
            if tether_enabled else "identical_to_optimization_potential"
        ),
    }
    history_manifest_path.write_text(json.dumps(history_manifest, indent=2, sort_keys=True) + "\n")
    iteration_history = history_manifest
    dual_history, dual_history_manifest_path = finalize_dual_history_manifest(
        out_dir,
        dual_history_extxyz,
        dual_history_csv,
        len(images),
        opt.nsteps,
        tether_config,
    )

    restrained_profile = energy_profile(images)
    profile = base_energy_profile(images)
    energy_delta = profile_energy_delta(initial_profile, profile)
    for row, atoms in zip(profile, images):
        atoms.get_potential_energy()
        atoms.get_forces()
        forces = (
            np.asarray(atoms.calc.base_results["forces"], dtype=float)
            if isinstance(atoms.calc, ReferenceTetherCalculator)
            else np.asarray(atoms.get_forces(), dtype=float)
        )
        atoms.info["energy"] = row["energy_eV"]
        atoms.info["mlff_rel_energy_eV"] = row["rel_energy_eV"]
        atoms.info["mlff_fmax_eV_A"] = row["fmax_eV_A"]
        atoms.info["calculator_label"] = calc_label
        atoms.info["neb_image"] = row["image_index"]
        atoms.info["neb_protocol"] = "mlff_pre_neb"
        atoms.arrays["forces"] = forces
        atoms.calc = None
    write(out_dir / "mlff_neb_images.extxyz", images)
    final_images_path = out_dir / "mlff_neb_images.extxyz"

    barrier = max(row["rel_energy_eV"] for row in profile)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_role": args.run_role,
        "source_candidate_manifest": str(Path(args.source_candidate_manifest).resolve()) if args.source_candidate_manifest else None,
        "source_candidate_manifest_sha256": sha256_file(Path(args.source_candidate_manifest).resolve()) if args.source_candidate_manifest else None,
        "seed": args.seed,
        "input_images": str(Path(args.images).resolve()) if args.images else None,
        "input_images_sha256": sha256_file(Path(args.images).resolve()) if args.images else None,
        "initial": str(Path(args.initial).resolve()) if args.initial else None,
        "initial_sha256": sha256_file(Path(args.initial).resolve()) if args.initial else None,
        "final": str(Path(args.final).resolve()) if args.final else None,
        "final_sha256": sha256_file(Path(args.final).resolve()) if args.final else None,
        "calculator": args.calculator,
        "calculator_label": calc_label,
        "model_path": calc_label if Path(calc_label).is_file() else None,
        "model_path_sha256": sha256_file(Path(calc_label)) if Path(calc_label).is_file() else None,
        "software_versions": software_versions(),
        "default_dtype": args.default_dtype if args.calculator == "mace-foundation" else "calculator_checkpoint_default",
        "n_images": len(images),
        "fmax_target_eV_A": args.fmax,
        "steps_requested": args.steps,
        "optimizer_steps_completed": int(opt.nsteps),
        "optimizer_converged": optimizer_converged,
        "optimizer_converged_under_optimization_potential": optimizer_converged_under_optimization_potential,
        "optimization_potential": "base_plus_reference_tether" if tether_enabled else "base",
        "reference_tether": tether_config,
        "initial_profile": initial_profile,
        "barrier_proxy_eV": barrier,
        "profile": profile,
        "restrained_profile": restrained_profile,
        "relaxation_energy_delta": energy_delta,
        "outputs": {
            "images": str(final_images_path),
            "images_sha256": sha256_file(final_images_path),
            "trajectory": str(out_dir / "mlff_neb.traj"),
            "log": str(out_dir / "mlff_neb.log"),
            "iteration_history": iteration_history["outputs"],
            "iteration_history_manifest": str(out_dir / "mlff_neb_iteration_history_manifest.json"),
            "dual_potential_history_extxyz": str(dual_history_extxyz),
            "dual_potential_history_extxyz_sha256": sha256_file(dual_history_extxyz),
            "dual_potential_history_csv": str(dual_history_csv),
            "dual_potential_history_csv_sha256": sha256_file(dual_history_csv),
            "dual_potential_history_manifest": str(dual_history_manifest_path),
            "dual_potential_history_manifest_sha256": sha256_file(dual_history_manifest_path),
        },
        "optimizer_iterations_recorded_including_initial": iteration_history["n_optimizer_iterations_including_initial"],
        "final_max_internal_true_force_eV_A": iteration_history["final_iteration_summary"]["max_internal_true_force_eV_A"],
        "final_max_internal_neb_force_eV_A": iteration_history["final_iteration_summary"]["max_internal_neb_force_eV_A"],
        "final_max_neb_residual_eV_A": iteration_history["final_iteration_summary"]["max_neb_residual_eV_A"],
        "dual_potential_history_complete": dual_history["complete"],
        "warning": "MLFF barrier is a preconditioner/proposal only; final paper reference must come from DFT NEB.",
    }
    (out_dir / "mlff_neb_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"barrier_proxy_eV": barrier, "images": manifest["outputs"]["images"]}, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", help="Existing ASE-readable path. If omitted, --initial/--final are interpolated.")
    p.add_argument("--initial", help="Initial structure for interpolation.")
    p.add_argument("--final", help="Final structure for interpolation.")
    p.add_argument("--n-images", type=int, default=7)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--calculator", default="mace-foundation", choices=["mace-foundation", "mace-model", "emt"])
    p.add_argument("--model-path", help="Explicit MACE checkpoint path; required for both MACE modes.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--default-dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spring-constant", type=float, default=0.1)
    p.add_argument("--migrant-element", default="Cr")
    p.add_argument("--migrant-tether-k", type=float, default=0.0, help="Harmonic reference-path restraint in eV/A^2.")
    p.add_argument("--host-tether-k", type=float, default=0.0, help="Harmonic host reference restraint in eV/A^2.")
    p.add_argument("--climb", action="store_true")
    p.add_argument("--mic", action="store_true", default=True)
    p.add_argument("--interpolate-method", default="idpp", choices=["linear", "idpp"])
    p.add_argument("--remove-rotation-translation", action="store_true")
    p.add_argument(
        "--run-role", default="production_preconditioner",
        choices=["production_preconditioner", "diagnostic_preconditioner", "unverified_smoke"],
    )
    p.add_argument("--source-candidate-manifest")
    return p


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
