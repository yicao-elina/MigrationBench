#!/usr/bin/env python3
"""Compare direct and MLFF-preconditioned QE NEB lineages without prefix bias."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from analyze_neb_path_topology import displacement, read_qe_image
from audit_qe_calculator_identity import identity as qe_calculator_identity
from parse_qe_neb_output import parse_neb_out
from parse_qe_neb_path_history import parse_path_file
from validate_runtime_provenance import validate_runtime_provenance


SCF_RE = re.compile(r"convergence has been achieved in\s+(\d+) iterations", re.I)
PATH_RE = re.compile(r"\.path(\d+)$")
JOB_ID_RE = re.compile(r"_(\d+)$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def latest_path(run_dir: Path) -> Path:
    paths = []
    for path in (run_dir / "out").glob("*.path[0-9]*"):
        match = PATH_RE.search(path.name)
        if match and path.stat().st_size:
            paths.append((int(match.group(1)), path))
    if not paths:
        raise FileNotFoundError(f"No nonempty QE path file under {run_dir / 'out'}")
    return max(paths)[1]


def scf_work(run_dirs: list[Path]) -> dict:
    iterations = []
    files = []
    for run_dir in run_dirs:
        for path in sorted((run_dir / "out").glob("*/PW.out")):
            values = [int(value) for value in SCF_RE.findall(path.read_text(errors="replace"))]
            iterations.extend(values)
            files.append({"path": str(path.resolve()), "sha256": sha256(path), "scf_calls": len(values)})
    return {
        "scf_calls": len(iterations),
        "scf_iterations": sum(iterations),
        "pw_outputs": files,
    }


def final_path(run_dir: Path) -> dict:
    template = read_qe_image(run_dir / "pw_1.in")
    path_file = latest_path(run_dir)
    parsed = parse_path_file(path_file, len(template["symbols"]))
    return {
        "symbols": template["symbols"],
        "cell": template["cell"],
        "images": [
            [atom["position_A"] for atom in image["atoms"]]
            for image in parsed["images"]
        ],
        "path_file": str(path_file.resolve()),
        "path_file_sha256": sha256(path_file),
    }


def lineage_runtime_provenance(run_dirs: list[Path]) -> dict:
    rows = []
    for run_dir in run_dirs:
        match = JOB_ID_RE.search(run_dir.name)
        expected_job_id = match.group(1) if match else None
        input_path = run_dir / "neb.in"
        if expected_job_id is None or not input_path.is_file():
            rows.append({
                "run_dir": str(run_dir.resolve()),
                "passed": False,
                "failed_checks": ["neb_input_or_job_id_missing"],
            })
            continue
        validation = validate_runtime_provenance(
            run_dir / "runtime_provenance.json",
            {"neb_input": sha256(input_path)},
            ["neb.x", "pw.x"],
            expected_job_id,
        )
        rows.append({"run_dir": str(run_dir.resolve()), **validation})
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "segment_count": len(rows),
        "segments": rows,
    }


def path_distance(left: dict, right: dict, migrant_symbol: str = "Cr") -> dict:
    if left["symbols"] != right["symbols"]:
        raise ValueError("A/B final paths have different atom ordering or composition")
    if len(left["images"]) != len(right["images"]):
        raise ValueError("A/B final paths have different image counts")
    migrant_indices = [i for i, symbol in enumerate(left["symbols"]) if symbol == migrant_symbol]
    if len(migrant_indices) != 1:
        raise ValueError(f"Expected exactly one {migrant_symbol} migrant; found {len(migrant_indices)}")
    migrant = migrant_indices[0]
    cell_delta = max(
        abs(float(a) - float(b))
        for left_row, right_row in zip(left["cell"], right["cell"])
        for a, b in zip(left_row, right_row)
    )
    all_squared = []
    migrant_squared = []
    endpoint_squared = []
    for image_index, (left_image, right_image) in enumerate(zip(left["images"], right["images"])):
        for atom_index, (left_position, right_position) in enumerate(zip(left_image, right_image)):
            vector = displacement(left_position, right_position, left["cell"])
            squared = sum(value * value for value in vector)
            all_squared.append(squared)
            if atom_index == migrant:
                migrant_squared.append(squared)
                if image_index in {0, len(left["images"]) - 1}:
                    endpoint_squared.append(squared)
    return {
        "cell_max_abs_delta_A": cell_delta,
        "all_atom_rms_A": math.sqrt(sum(all_squared) / len(all_squared)),
        "migrant_curve_rms_A": math.sqrt(sum(migrant_squared) / len(migrant_squared)),
        "migrant_curve_max_A": math.sqrt(max(migrant_squared)),
        "migrant_endpoint_max_A": math.sqrt(max(endpoint_squared)),
    }


def lineage_metrics(run_dirs: list[Path]) -> dict:
    if not run_dirs:
        raise ValueError("At least one ordered run directory is required")
    parsed_segments = [parse_neb_out(path / "neb.out") for path in run_dirs]
    final = parsed_segments[-1]
    work = scf_work(run_dirs)
    residual_curve = []
    cumulative_iteration = 0
    for segment_index, parsed in enumerate(parsed_segments, 1):
        for iteration in parsed.get("complete_iterations", []):
            residuals = [
                float(image["error_eV_A"])
                for image in iteration.get("images", [])
                if image.get("error_eV_A") is not None and not image.get("frozen")
            ]
            if residuals:
                residual_curve.append({
                    "cumulative_path_iteration": cumulative_iteration,
                    "segment_index": segment_index,
                    "segment_iteration": iteration.get("iteration"),
                    "max_movable_image_error_eV_A": max(residuals),
                })
                cumulative_iteration += 1
    residual_auc = sum(
        0.5
        * (left["max_movable_image_error_eV_A"] + right["max_movable_image_error_eV_A"])
        * (right["cumulative_path_iteration"] - left["cumulative_path_iteration"])
        for left, right in zip(residual_curve, residual_curve[1:])
    )
    first_threshold = {
        str(threshold): next(
            (
                row["cumulative_path_iteration"]
                for row in residual_curve
                if row["max_movable_image_error_eV_A"] <= threshold
            ),
            None,
        )
        for threshold in (1.0, 0.5, 0.1, 0.05, 0.03)
    }
    runtime = lineage_runtime_provenance(run_dirs)
    return {
        "run_dirs": [str(path.resolve()) for path in run_dirs],
        "segment_count": len(run_dirs),
        "converged": final.get("converged_by_default_gate") is True,
        "has_overflow": any(row.get("has_overflow") for row in parsed_segments),
        "path_iterations": sum(row.get("n_complete_iterations_parsed", 0) for row in parsed_segments),
        "initial_max_image_error_eV_A": (
            residual_curve[0]["max_movable_image_error_eV_A"] if residual_curve else None
        ),
        "residual_auc_eV_A_path_iteration": residual_auc if residual_curve else None,
        "first_path_iteration_at_or_below_eV_A": first_threshold,
        "residual_curve": residual_curve,
        "activation_forward_eV": final.get("activation_forward_eV"),
        "activation_reverse_eV": final.get("activation_reverse_eV"),
        "max_image_error_eV_A": final.get("max_image_error_eV_A"),
        "barrier_drift_last_three_eV": final.get("barrier_drift_last_three_eV"),
        **work,
        "final_path": final_path(run_dirs[-1]),
        "calculator_identity": qe_calculator_identity(run_dirs[-1] / "neb.in", run_dirs[-1].name),
        "initial_input_sha256": sha256(run_dirs[0] / "neb.in"),
        "neb_outputs": [
            {"path": str((path / "neb.out").resolve()), "sha256": sha256(path / "neb.out")}
            for path in run_dirs
        ],
        "runtime_provenance": runtime,
        "runtime_provenance_complete": runtime["passed"],
    }


def comparison_decision(
    direct: dict,
    preconditioned: dict,
    distances: dict,
    identity_match: bool,
    calculator_accepted: bool,
    max_all_atom_rms_A: float,
    max_migrant_rms_A: float,
    max_barrier_delta_eV: float,
    pair_binding: bool = True,
    runtime_provenance_complete: bool = False,
) -> tuple[str, bool]:
    if direct["has_overflow"] or preconditioned["has_overflow"]:
        return "invalid_nonphysical_or_overflow", False
    if not direct["converged"] or not preconditioned["converged"]:
        return "pending_or_censored_not_both_converged", False
    if not identity_match:
        return "invalid_calculator_identity_mismatch", False
    if not pair_binding:
        return "invalid_pair_manifest_input_binding", False
    if not runtime_provenance_complete:
        return "invalid_incomplete_qe_runtime_provenance", False
    if not calculator_accepted:
        return "same_calculator_identity_pending_n24_acceptance", False
    if direct.get("activation_forward_eV") is None or preconditioned.get("activation_forward_eV") is None:
        return "invalid_missing_final_barrier", False
    barrier_delta = abs(direct["activation_forward_eV"] - preconditioned["activation_forward_eV"])
    same_mechanism = (
        distances.get("cell_max_abs_delta_A", math.inf) <= 1.0e-6
        and
        distances["all_atom_rms_A"] <= max_all_atom_rms_A
        and distances["migrant_curve_rms_A"] <= max_migrant_rms_A
        and barrier_delta <= max_barrier_delta_eV
    )
    if not same_mechanism:
        return "different_final_mechanism_no_speedup", False
    return "comparable_same_mechanism_speedup_measurable", True


def accepted_calculator(path: Path | None, calculator_identity: str) -> tuple[bool, str | None]:
    if path is None:
        return False, None
    payload = json.loads(path.read_text())
    return (
        payload.get("status") == "pass"
        and payload.get("accepted_calculator_identity") == calculator_identity,
        sha256(path),
    )


def compare(args: argparse.Namespace) -> dict:
    direct = lineage_metrics([path.resolve() for path in args.direct_run_dir])
    preconditioned = lineage_metrics([path.resolve() for path in args.preconditioned_run_dir])
    distances = path_distance(direct["final_path"], preconditioned["final_path"], args.migrant_symbol)
    direct_identity = direct["calculator_identity"]
    preconditioned_identity = preconditioned["calculator_identity"]
    identity_match = (
        direct_identity.get("identity_complete")
        and preconditioned_identity.get("identity_complete")
        and direct_identity.get("calculator_identity") == preconditioned_identity.get("calculator_identity")
    )
    calculator_accepted, acceptance_hash = accepted_calculator(
        args.calculator_acceptance,
        direct_identity.get("calculator_identity"),
    )
    pair_manifest = json.loads(args.pair_manifest.read_text())
    pair_binding = (
        direct["initial_input_sha256"] == pair_manifest.get("direct_input_sha256")
        and preconditioned["initial_input_sha256"] == pair_manifest.get("mlff_input_sha256")
    )
    decision, speedup_eligible = comparison_decision(
        direct,
        preconditioned,
        distances,
        identity_match,
        calculator_accepted,
        args.max_all_atom_rms,
        args.max_migrant_rms,
        args.max_barrier_delta,
        pair_binding,
        direct["runtime_provenance_complete"] and preconditioned["runtime_provenance_complete"],
    )
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_id": pair_manifest.get("path_id"),
        "pair_manifest": str(args.pair_manifest.resolve()),
        "pair_manifest_sha256": sha256(args.pair_manifest),
        "direct": direct,
        "preconditioned": preconditioned,
        "final_path_distance": distances,
        "forward_barrier_delta_eV": (
            abs(direct["activation_forward_eV"] - preconditioned["activation_forward_eV"])
            if direct["activation_forward_eV"] is not None
            and preconditioned["activation_forward_eV"] is not None else None
        ),
        "calculator_identities_match": bool(identity_match),
        "pair_manifest_inputs_match_lineages": pair_binding,
        "runtime_provenance_complete": (
            direct["runtime_provenance_complete"]
            and preconditioned["runtime_provenance_complete"]
        ),
        "calculator_identity_accepted_by_n24": calculator_accepted,
        "calculator_acceptance": str(args.calculator_acceptance.resolve()) if args.calculator_acceptance else None,
        "calculator_acceptance_sha256": acceptance_hash,
        "thresholds": {
            "max_all_atom_rms_A": args.max_all_atom_rms,
            "max_migrant_curve_rms_A": args.max_migrant_rms,
            "max_forward_barrier_delta_eV": args.max_barrier_delta,
        },
        "decision": decision,
        "speedup_eligible": speedup_eligible,
        "path_iteration_speedup_direct_over_preconditioned": (
            direct["path_iterations"] / preconditioned["path_iterations"]
            if speedup_eligible and preconditioned["path_iterations"] else None
        ),
        "scf_iteration_speedup_direct_over_preconditioned": (
            direct["scf_iterations"] / preconditioned["scf_iterations"]
            if speedup_eligible and preconditioned["scf_iterations"] else None
        ),
        "notes": [
            "No speedup is calculated from a running or one-sided-converged prefix.",
            "A different final mechanism is a scientific outcome, not an acceleration.",
            "MACE and tether energies are excluded; barriers come only from converged QE NEB outputs.",
            "Every cost-bearing QE segment must bind neb.in, Slurm job id, neb.x, and pw.x runtime identity.",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--direct-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--preconditioned-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--calculator-acceptance", type=Path)
    parser.add_argument("--migrant-symbol", default="Cr")
    parser.add_argument("--max-all-atom-rms", type=float, default=0.15)
    parser.add_argument("--max-migrant-rms", type=float, default=0.25)
    parser.add_argument("--max-barrier-delta", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "qe_neb_initialization_ab.json"
    csv_path = args.out_dir / "qe_neb_initialization_ab.csv"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    rows = []
    for role in ("direct", "preconditioned"):
        metrics = result[role]
        rows.append({
            "role": role,
            "converged": metrics["converged"],
            "path_iterations": metrics["path_iterations"],
            "scf_calls": metrics["scf_calls"],
            "scf_iterations": metrics["scf_iterations"],
            "activation_forward_eV": metrics["activation_forward_eV"],
            "max_image_error_eV_A": metrics["max_image_error_eV_A"],
            "decision": result["decision"],
        })
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(json_path), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
