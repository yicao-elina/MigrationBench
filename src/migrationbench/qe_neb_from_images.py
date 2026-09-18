#!/usr/bin/env python3
"""Generate a Quantum ESPRESSO neb.x input from existing NEB images.

This is intended for the MLFF -> DFT workflow: relax a band cheaply with MACE,
freeze those images as FIRST/INTERMEDIATE/LAST_IMAGE cards, and let QE refine
the same path with a fully traceable input file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ase.io import read
from audit_qe_calculator_identity import identity as qe_calculator_identity


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_bool(value: bool) -> str:
    return ".true." if value else ".false."


def path_block(args: argparse.Namespace, n_images: int) -> str:
    return "\n".join(
        [
            "BEGIN_PATH_INPUT",
            "&PATH",
            "  string_method = 'neb',",
            f"  restart_mode = '{args.restart_mode}',",
            f"  nstep_path = {args.nstep_path},",
            f"  num_of_images = {n_images},",
            f"  opt_scheme = '{args.opt_scheme}',",
            f"  CI_scheme = '{args.ci_scheme}',",
            f"  first_last_opt = {fmt_bool(args.first_last_opt)},",
            f"  minimum_image = {fmt_bool(args.minimum_image)},",
            f"  ds = {args.ds},",
            f"  k_min = {args.k_min},",
            f"  k_max = {args.k_max},",
            f"  path_thr = {args.path_thr},",
            "/",
            "END_PATH_INPUT",
        ]
    )


def positions_block(images, coordinate_mode: str) -> str:
    lines = ["BEGIN_POSITIONS"]
    for idx, atoms in enumerate(images):
        if idx == 0:
            label = "FIRST_IMAGE"
        elif idx == len(images) - 1:
            label = "LAST_IMAGE"
        else:
            label = "INTERMEDIATE_IMAGE"
        lines.append(label)
        lines.append(f"ATOMIC_POSITIONS {coordinate_mode}")
        for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            lines.append(f"{sym:2s} {pos[0]: .10f} {pos[1]: .10f} {pos[2]: .10f}")
    lines.append("END_POSITIONS")
    return "\n".join(lines)


def load_json(path: str | None, label: str) -> tuple[Path, dict]:
    if not path:
        raise ValueError(f"{label} is required")
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved, json.loads(resolved.read_text())


def manifest_artifact(manifest_path: Path, recorded_path: str | None) -> Path | None:
    """Resolve a manifested artifact after a Rockfish run has been mirrored locally."""
    if not recorded_path:
        return None
    recorded = Path(recorded_path)
    if recorded.is_file():
        return recorded.resolve()
    mirrored = manifest_path.parent / recorded.name
    return mirrored.resolve() if mirrored.is_file() else None


def candidate_for_images(candidate_path: Path, candidate: dict, images_path: Path) -> dict:
    image_hash = sha256_file(images_path)
    matches = [row for row in candidate.get("candidates", []) if row.get("images_sha256") == image_hash]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one candidate row matching images SHA-256 {image_hash}; found {len(matches)}"
        )
    row = matches[0]
    recorded = (candidate_path.parent / row["images"]).resolve()
    if recorded.is_file() and sha256_file(recorded) != image_hash:
        raise ValueError("Candidate row path and supplied images have inconsistent hashes")
    return row


def pre_output_gate(args: argparse.Namespace, images_path: Path) -> dict:
    gate = {
        "run_role": args.run_role,
        "initialization_role": args.initialization_role,
        "status": "diagnostic_not_production_eligible",
        "checks": {},
    }
    if args.run_role == "diagnostic":
        return gate

    candidate_path, candidate = load_json(args.source_candidate_manifest, "source candidate manifest")
    candidate_hash = sha256_file(candidate_path)
    row = candidate_for_images(candidate_path, candidate, Path(args.direct_baseline_images).resolve())
    checks = gate["checks"]
    checks["candidate_endpoint_status"] = candidate.get("endpoint_status") == "accepted_local_minima"
    checks["candidate_geometry_gate"] = row.get("geometry_gate", {}).get("status") == "pass"
    checks["candidate_not_duplicate"] = row.get("duplicate_of") is None
    checks["candidate_production_eligible"] = row.get("production_eligible_before_mace") is True

    endpoint_path, endpoint = load_json(args.endpoint_acceptance, "endpoint acceptance")
    endpoint_hash = sha256_file(endpoint_path)
    checks["endpoint_acceptance_status"] = endpoint.get("status") == "accepted"
    checks["candidate_endpoint_acceptance_binding"] = (
        candidate.get("endpoint_acceptance_sha256") == endpoint_hash
    )
    checks["initial_endpoint_structure_binding"] = (
        candidate.get("initial_endpoint_source_sha256")
        == endpoint.get("initial_structure_sha256")
    )
    checks["final_endpoint_structure_binding"] = (
        candidate.get("final_endpoint_source_sha256")
        == endpoint.get("final_structure_sha256")
    )

    mlff_path = None
    mlff = None
    if args.initialization_role in {"mlff_preconditioned", "trust_region_mlff_preconditioned"}:
        mlff_path, mlff = load_json(args.source_mlff_manifest, "source MLFF manifest")
        checks["mlff_candidate_binding"] = mlff.get("source_candidate_manifest_sha256") == candidate_hash
        checks["mlff_input_binding"] = mlff.get("input_images_sha256") == row.get("images_sha256")
        checks["mlff_output_binding"] = mlff.get("outputs", {}).get("images_sha256") == sha256_file(images_path)
        residual = mlff.get("final_max_neb_residual_eV_A")
        checks["mlff_residual_gate"] = residual is not None and residual <= args.max_mlff_residual
        if args.initialization_role == "mlff_preconditioned":
            checks["mlff_optimizer_converged"] = mlff.get("optimizer_converged") is True
            checks["mlff_unrestrained_potential"] = mlff.get("reference_tether", {}).get("enabled") is not True
        else:
            outputs = mlff.get("outputs", {})
            dual_extxyz = manifest_artifact(mlff_path, outputs.get("dual_potential_history_extxyz"))
            dual_csv = manifest_artifact(mlff_path, outputs.get("dual_potential_history_csv"))
            checks["trust_region_enabled"] = mlff.get("reference_tether", {}).get("enabled") is True
            checks["trust_region_optimizer_converged"] = (
                mlff.get("optimizer_converged_under_optimization_potential") is True
            )
            checks["trust_region_not_mislabeled_base_converged"] = mlff.get("optimizer_converged") is False
            checks["dual_history_extxyz"] = (
                dual_extxyz is not None
                and sha256_file(dual_extxyz) == outputs.get("dual_potential_history_extxyz_sha256")
            )
            checks["dual_history_csv"] = (
                dual_csv is not None
                and sha256_file(dual_csv) == outputs.get("dual_potential_history_csv_sha256")
            )
            acceptance_path, acceptance = load_json(
                getattr(args, "mlff_preconditioner_acceptance", None),
                "trust-region preconditioner acceptance",
            )
            checks["trust_region_acceptance_decision"] = (
                acceptance.get("decision") == "trust_region_candidate_ready_for_dft_ab_design"
            )
            checks["trust_region_physical_identity"] = acceptance.get("physical_identity_match") is True
            checks["trust_region_manifest_binding"] = (
                acceptance.get("provenance", {}).get("trust_manifest_sha256")
                == sha256_file(mlff_path)
            )
            gate["mlff_preconditioner_acceptance"] = str(acceptance_path)
            gate["mlff_preconditioner_acceptance_sha256"] = sha256_file(acceptance_path)
    else:
        checks["direct_images_binding"] = sha256_file(images_path) == row.get("images_sha256")

    failed = sorted(key for key, value in checks.items() if not value)
    gate.update(
        {
            "status": "pass" if not failed else "fail",
            "failed_checks": failed,
            "candidate_manifest": str(candidate_path),
            "candidate_manifest_sha256": candidate_hash,
            "candidate_branch_id": row.get("branch_id"),
            "endpoint_acceptance": str(endpoint_path),
            "endpoint_acceptance_sha256": endpoint_hash,
            "source_mlff_manifest": str(mlff_path) if mlff_path else None,
            "source_mlff_manifest_sha256": sha256_file(mlff_path) if mlff_path else None,
        }
    )
    if failed:
        raise ValueError("Production handoff gate failed: " + ", ".join(failed))
    return gate


def identity_from_text(text: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="migrationbench-qe-identity-") as directory:
        path = Path(directory) / "neb.in"
        path.write_text(text)
        return qe_calculator_identity(path, "generated_handoff")


def write_qe_neb_input(args: argparse.Namespace) -> dict:
    images_path = Path(args.images).resolve()
    engine_template_path = Path(args.engine_template).resolve()
    out_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else out_path.with_suffix(".manifest.json")

    images = read(images_path, index=":")
    if len(images) < 4:
        raise ValueError("QE NEB requires num_of_images > 3; provide at least 4 images.")
    handoff_gate = pre_output_gate(args, images_path)

    engine_template = engine_template_path.read_text()
    if "BEGIN_POSITIONS" in engine_template or "ATOMIC_POSITIONS" in engine_template:
        raise ValueError("Engine template must omit ATOMIC_POSITIONS; positions are injected by this script.")
    if "CELL_PARAMETERS" not in engine_template:
        cell = images[0].get_cell()
        cell_block = ["CELL_PARAMETERS angstrom"]
        for row in cell:
            cell_block.append(f"{row[0]: .10f} {row[1]: .10f} {row[2]: .10f}")
        engine_template = engine_template.rstrip() + "\n" + "\n".join(cell_block) + "\n"

    text = "\n".join(
        [
            "BEGIN",
            path_block(args, len(images)),
            "BEGIN_ENGINE_INPUT",
            engine_template.rstrip(),
            positions_block(images, args.coordinate_mode),
            "END_ENGINE_INPUT",
            "END",
            "",
        ]
    )
    calculator_identity = identity_from_text(text)
    if args.run_role == "production":
        acceptance_path, acceptance = load_json(args.calculator_acceptance, "calculator acceptance")
        accepted_identity = acceptance.get("accepted_calculator_identity")
        calculator_pass = (
            acceptance.get("status") == "pass"
            and calculator_identity["identity_complete"]
            and accepted_identity == calculator_identity["calculator_identity"]
        )
        handoff_gate["checks"]["calculator_identity_accepted"] = calculator_pass
        handoff_gate["calculator_acceptance"] = str(acceptance_path)
        handoff_gate["calculator_acceptance_sha256"] = sha256_file(acceptance_path)
        if not calculator_pass:
            raise ValueError(
                "Production handoff gate failed: generated calculator identity is not accepted"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_id": args.path_id,
        "path_family": args.path_family,
        "measurement_purpose": args.measurement_purpose,
        "run_role": args.run_role,
        "initialization_role": args.initialization_role,
        "images": str(images_path),
        "images_sha256": sha256_file(images_path),
        "engine_template": str(engine_template_path),
        "engine_template_sha256": sha256_file(engine_template_path),
        "source_neb_output": str(Path(args.source_neb_output).resolve()) if args.source_neb_output else None,
        "source_neb_output_sha256": sha256_file(Path(args.source_neb_output).resolve()) if args.source_neb_output else None,
        "source_mlff_manifest": str(Path(args.source_mlff_manifest).resolve()) if args.source_mlff_manifest else None,
        "source_mlff_manifest_sha256": sha256_file(Path(args.source_mlff_manifest).resolve()) if args.source_mlff_manifest else None,
        "mlff_preconditioner_acceptance": (
            str(Path(args.mlff_preconditioner_acceptance).resolve())
            if getattr(args, "mlff_preconditioner_acceptance", None) else None
        ),
        "mlff_preconditioner_acceptance_sha256": (
            sha256_file(Path(args.mlff_preconditioner_acceptance).resolve())
            if getattr(args, "mlff_preconditioner_acceptance", None) else None
        ),
        "source_candidate_manifest": str(Path(args.source_candidate_manifest).resolve()) if args.source_candidate_manifest else None,
        "source_candidate_manifest_sha256": sha256_file(Path(args.source_candidate_manifest).resolve()) if args.source_candidate_manifest else None,
        "direct_baseline_images": str(Path(args.direct_baseline_images).resolve()) if args.direct_baseline_images else None,
        "direct_baseline_images_sha256": sha256_file(Path(args.direct_baseline_images).resolve()) if args.direct_baseline_images else None,
        "output": str(out_path),
        "output_sha256": sha256_file(out_path),
        "calculator_identity": calculator_identity,
        "production_handoff_gate": handoff_gate,
        "num_of_images": len(images),
        "path_settings": {
            "restart_mode": args.restart_mode,
            "nstep_path": args.nstep_path,
            "opt_scheme": args.opt_scheme,
            "CI_scheme": args.ci_scheme,
            "first_last_opt": args.first_last_opt,
            "minimum_image": args.minimum_image,
            "ds": args.ds,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "path_thr": args.path_thr,
        },
        "notes": [
            "Generated from fixed images; inspect chemistry and pseudopotentials before submission.",
            "QE neb.x does not read standard input; submit with neb.x -inp this file.",
        ],
    }
    if args.source_mlff_manifest:
        mlff = json.loads(Path(args.source_mlff_manifest).read_text())
        manifest["mlff_preconditioning"] = {
            "calculator": mlff.get("calculator"),
            "calculator_label": mlff.get("calculator_label"),
            "model_path": mlff.get("model_path"),
            "model_path_sha256": mlff.get("model_path_sha256"),
            "seed": mlff.get("seed"),
            "fmax_target_eV_A": mlff.get("fmax_target_eV_A"),
            "steps_requested": mlff.get("steps_requested"),
            "optimizer_steps_completed": mlff.get("optimizer_steps_completed"),
            "optimizer_converged": mlff.get("optimizer_converged"),
            "final_max_neb_residual_eV_A": mlff.get("final_max_neb_residual_eV_A"),
            "barrier_proxy_eV": mlff.get("barrier_proxy_eV"),
            "initial_profile": mlff.get("initial_profile"),
            "final_profile": mlff.get("profile"),
            "relaxation_energy_delta": mlff.get("relaxation_energy_delta"),
            "warning": mlff.get("warning"),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", required=True, help="Input ASE-readable trajectory with all NEB images.")
    p.add_argument("--engine-template", required=True, help="PWscf input body without ATOMIC_POSITIONS.")
    p.add_argument("--output", required=True, help="Output QE neb.x input path.")
    p.add_argument("--manifest", help="Output manifest JSON path. Default: output with .manifest.json suffix.")
    p.add_argument("--path-id", default="unknown", help="Stable MigrationBench path id.")
    p.add_argument("--path-family", default="unclassified", help="Path family or continuous class label.")
    p.add_argument("--measurement-purpose", default="dft_refinement_of_mlff_preconditioned_neb")
    p.add_argument("--run-role", default="diagnostic", choices=["diagnostic", "production"])
    p.add_argument(
        "--initialization-role", default="mlff_preconditioned",
        choices=["direct_baseline", "mlff_preconditioned", "trust_region_mlff_preconditioned"],
    )
    p.add_argument("--source-neb-output", help="Historical QE neb.out used to seed this path, if any.")
    p.add_argument("--source-mlff-manifest", help="MLFF manifest used to generate the input images, if any.")
    p.add_argument(
        "--mlff-preconditioner-acceptance",
        help="Comparator artifact accepting a trust-region band for a paired DFT design.",
    )
    p.add_argument("--source-candidate-manifest", help="Nonlinear candidate manifest that owns the direct path.")
    p.add_argument("--direct-baseline-images", help="Frozen candidate images before MLFF preconditioning.")
    p.add_argument("--endpoint-acceptance", help="Accepted endpoint-pair artifact bound to the candidate manifest.")
    p.add_argument("--calculator-acceptance", help="N24 accepted QE calculator-identity artifact.")
    p.add_argument("--max-mlff-residual", type=float, default=0.20)
    p.add_argument("--coordinate-mode", default="angstrom", choices=["angstrom"], help="Position coordinate mode.")
    p.add_argument("--restart-mode", default="from_scratch", choices=["from_scratch", "restart"])
    p.add_argument("--nstep-path", type=int, default=200)
    p.add_argument("--max-seconds", type=int, default=84600)
    p.add_argument("--opt-scheme", default="broyden", choices=["sd", "broyden", "broyden2", "quick-min"])
    p.add_argument("--ci-scheme", default="auto", choices=["no-CI", "auto", "manual"])
    p.add_argument("--first-last-opt", action="store_true")
    p.add_argument("--minimum-image", action="store_true", default=True)
    p.add_argument("--ds", type=float, default=0.2)
    p.add_argument("--k-min", type=float, default=0.1)
    p.add_argument("--k-max", type=float, default=0.3)
    p.add_argument("--path-thr", type=float, default=0.03)
    return p


def main() -> None:
    manifest = write_qe_neb_input(build_parser().parse_args())
    print(json.dumps({"wrote": manifest["output"], "num_of_images": manifest["num_of_images"]}, indent=2))


if __name__ == "__main__":
    main()
