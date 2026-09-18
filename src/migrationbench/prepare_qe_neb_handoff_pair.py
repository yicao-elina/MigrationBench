#!/usr/bin/env python3
"""Build matched direct and MLFF-preconditioned QE NEB inputs with hard gates."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from qe_neb_from_images import sha256_file, write_qe_neb_input


def generation_args(args, images, output, manifest, initialization_role, source_mlff):
    return SimpleNamespace(
        images=str(images),
        engine_template=str(args.engine_template),
        output=str(output),
        manifest=str(manifest),
        path_id=args.path_id,
        path_family=args.path_family,
        measurement_purpose="matched_dft_initialization_efficiency",
        run_role=args.run_role,
        initialization_role=initialization_role,
        source_neb_output=args.source_neb_output,
        source_mlff_manifest=str(source_mlff) if source_mlff else None,
        mlff_preconditioner_acceptance=(
            str(args.mlff_preconditioner_acceptance)
            if args.mlff_preconditioner_acceptance else None
        ),
        source_candidate_manifest=str(args.source_candidate_manifest),
        direct_baseline_images=str(args.direct_images),
        endpoint_acceptance=str(args.endpoint_acceptance) if args.endpoint_acceptance else None,
        calculator_acceptance=str(args.calculator_acceptance) if args.calculator_acceptance else None,
        max_mlff_residual=args.max_mlff_residual,
        coordinate_mode="angstrom",
        restart_mode="from_scratch",
        nstep_path=args.nstep_path,
        opt_scheme=args.opt_scheme,
        ci_scheme=args.ci_scheme,
        first_last_opt=False,
        minimum_image=True,
        ds=args.ds,
        k_min=args.k_min,
        k_max=args.k_max,
        path_thr=args.path_thr,
    )


def prepare(args):
    if (
        args.mlff_initialization_role == "trust_region_mlff_preconditioned"
        and args.mlff_preconditioner_acceptance is None
    ):
        raise ValueError("Trust-region handoff requires --mlff-preconditioner-acceptance")
    if (
        args.mlff_initialization_role == "mlff_preconditioned"
        and args.mlff_preconditioner_acceptance is not None
    ):
        raise ValueError("Unrestrained MLFF handoff must not use a trust-region acceptance artifact")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    direct_dir = out_dir / "direct_baseline"
    mlff_dir = out_dir / "mlff_preconditioned"
    direct_dir.mkdir(exist_ok=True)
    mlff_dir.mkdir(exist_ok=True)

    candidate = json.loads(args.source_candidate_manifest.read_text())
    mlff = json.loads(args.source_mlff_manifest.read_text())
    direct_hash = sha256_file(args.direct_images.resolve())
    if mlff.get("input_images_sha256") != direct_hash:
        raise ValueError("MLFF input is not the frozen direct-baseline candidate")
    if mlff.get("source_candidate_manifest_sha256") != sha256_file(args.source_candidate_manifest.resolve()):
        raise ValueError("MLFF manifest is not bound to the supplied candidate manifest")
    if mlff.get("seed") != args.seed:
        raise ValueError("MLFF and handoff-pair seeds must match")

    direct_manifest = write_qe_neb_input(
        generation_args(
            args, args.direct_images, direct_dir / "neb.in", direct_dir / "qe_handoff_manifest.json",
            "direct_baseline", None,
        )
    )
    mlff_manifest = write_qe_neb_input(
        generation_args(
            args, args.mlff_images, mlff_dir / "neb.in", mlff_dir / "qe_handoff_manifest.json",
            args.mlff_initialization_role, args.source_mlff_manifest,
        )
    )
    direct_identity = direct_manifest["calculator_identity"]["calculator_identity"]
    mlff_identity = mlff_manifest["calculator_identity"]["calculator_identity"]
    if direct_identity != mlff_identity:
        raise ValueError("Generated A/B inputs have different QE calculator identities")

    pair = {
        "schema_version": "1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": (
            "matched_direct_vs_trust_region_mlff_preconditioned_dft_neb"
            if args.mlff_initialization_role == "trust_region_mlff_preconditioned"
            else "matched_direct_vs_mlff_preconditioned_dft_neb"
        ),
        "run_role": args.run_role,
        "path_id": args.path_id,
        "path_family": args.path_family,
        "seed": args.seed,
        "candidate_generation_seed": candidate.get("seed"),
        "source_candidate_manifest": str(args.source_candidate_manifest.resolve()),
        "source_candidate_manifest_sha256": sha256_file(args.source_candidate_manifest.resolve()),
        "source_mlff_manifest": str(args.source_mlff_manifest.resolve()),
        "source_mlff_manifest_sha256": sha256_file(args.source_mlff_manifest.resolve()),
        "mlff_initialization_role": args.mlff_initialization_role,
        "mlff_preconditioner_acceptance": (
            str(args.mlff_preconditioner_acceptance.resolve())
            if args.mlff_preconditioner_acceptance else None
        ),
        "mlff_preconditioner_acceptance_sha256": (
            sha256_file(args.mlff_preconditioner_acceptance.resolve())
            if args.mlff_preconditioner_acceptance else None
        ),
        "direct_input": "direct_baseline/neb.in",
        "direct_input_sha256": direct_manifest["output_sha256"],
        "direct_handoff_manifest": "direct_baseline/qe_handoff_manifest.json",
        "direct_handoff_manifest_sha256": sha256_file(
            direct_dir / "qe_handoff_manifest.json"
        ),
        "mlff_input": "mlff_preconditioned/neb.in",
        "mlff_input_sha256": mlff_manifest["output_sha256"],
        "mlff_handoff_manifest": "mlff_preconditioned/qe_handoff_manifest.json",
        "mlff_handoff_manifest_sha256": sha256_file(
            mlff_dir / "qe_handoff_manifest.json"
        ),
        "shared_calculator_identity": direct_identity,
        "calculator_identities_match": True,
        "production_submission_eligible": (
            args.run_role == "production"
            and direct_manifest["production_handoff_gate"]["status"] == "pass"
            and mlff_manifest["production_handoff_gate"]["status"] == "pass"
        ),
        "comparison_required_conditions": [
            "both_qe_neb_converged",
            "same_final_mechanism",
            "same_calculator_identity",
            "ionic_and_scf_cost_reported",
            "barrier_proxy_excluded_from_final_dft_barrier",
        ],
        "comparison_status": "not_submitted",
    }
    pair_path = out_dir / "qe_handoff_pair_manifest.json"
    pair_path.write_text(json.dumps(pair, indent=2) + "\n")
    return pair_path, pair


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--direct-images", type=Path, required=True)
    p.add_argument("--mlff-images", type=Path, required=True)
    p.add_argument("--engine-template", type=Path, required=True)
    p.add_argument("--source-candidate-manifest", type=Path, required=True)
    p.add_argument("--source-mlff-manifest", type=Path, required=True)
    p.add_argument(
        "--mlff-initialization-role",
        choices=["mlff_preconditioned", "trust_region_mlff_preconditioned"],
        default="mlff_preconditioned",
    )
    p.add_argument("--mlff-preconditioner-acceptance", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--path-id", required=True)
    p.add_argument("--path-family", default="unclassified")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-role", choices=["diagnostic", "production"], default="diagnostic")
    p.add_argument("--endpoint-acceptance", type=Path)
    p.add_argument("--calculator-acceptance", type=Path)
    p.add_argument("--source-neb-output")
    p.add_argument("--max-mlff-residual", type=float, default=0.20)
    p.add_argument("--nstep-path", type=int, default=200)
    p.add_argument("--opt-scheme", default="broyden")
    p.add_argument("--ci-scheme", default="auto")
    p.add_argument("--ds", type=float, default=0.2)
    p.add_argument("--k-min", type=float, default=0.1)
    p.add_argument("--k-max", type=float, default=0.3)
    p.add_argument("--path-thr", type=float, default=0.03)
    return p


if __name__ == "__main__":
    output, manifest = prepare(parser().parse_args())
    print(json.dumps({"pair_manifest": str(output), "production_submission_eligible": manifest["production_submission_eligible"]}, indent=2))
