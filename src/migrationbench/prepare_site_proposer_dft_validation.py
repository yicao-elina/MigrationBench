#!/usr/bin/env python3
"""Pre-register a leakage-isolated prospective DFT site-validation batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from prepare_qe_endpoint_relaxations import transform_pw


FEATURE_PREFIXES = ("radial_", "nearest_", "coord_", "angular_", "minimum_", "contact_")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def reference_cell_block(path: Path) -> str:
    text = path.read_text()
    matches = re.findall(
        r"(?im)^\s*CELL_PARAMETERS[^\n]*\n(?:\s*[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s*\n?){3}",
        text,
    )
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one CELL_PARAMETERS block in {path}")
    return matches[0].strip() + "\n"


def inject_reference_cell(text: str, cell_block: str) -> str:
    if re.search(r"(?im)^\s*CELL_PARAMETERS\b", text):
        raise ValueError("Candidate source unexpectedly already contains CELL_PARAMETERS")
    marker = re.search(r"(?im)^\s*ATOMIC_POSITIONS\b", text)
    if not marker:
        raise ValueError("Candidate source has no ATOMIC_POSITIONS block")
    return text[: marker.start()] + cell_block + "\n" + text[marker.start() :]


def feature_columns(rows: list[dict]) -> list[str]:
    return [key for key in rows[0] if key.startswith(FEATURE_PREFIXES)]


def energy_quantiles(rows: list[dict], bins: int = 5) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (float(row["screening_energy_eV"]), row["raw_site_id"]))
    return {
        row["raw_site_id"]: min(bins - 1, int(index * bins / len(ordered)))
        for index, row in enumerate(ordered)
    }


def select_proposals(rows: list[dict], count: int) -> list[dict]:
    ranked = sorted(
        (
            row for row in rows
            if row.get("recommended_rank") not in (None, "")
            and row.get("dft_relax_status") == "missing"
        ),
        key=lambda row: (int(row["recommended_rank"]), row["raw_site_id"]),
    )
    selected = []
    used_groups = set()
    for row in ranked:
        if row["void_family_group"] in used_groups:
            continue
        selected.append(row)
        used_groups.add(row["void_family_group"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} independent prospective proposals available")
    return selected


def match_controls(rows: list[dict], proposals: list[dict], count: int) -> list[tuple[dict, dict, dict]]:
    from scipy.optimize import linear_sum_assignment

    columns = feature_columns(rows)
    matrix = np.asarray([[float(row[key]) for key in columns] for row in rows], dtype=float)
    center = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), 1.0e-12)
    vectors = {
        row["raw_site_id"]: (values - center) / scale
        for row, values in zip(rows, matrix)
    }
    quantiles = energy_quantiles(rows)
    forbidden_sites = {
        row["raw_site_id"] for row in rows
        if row.get("recommended_rank") not in (None, "")
    }
    proposal_groups = {row["void_family_group"] for row in proposals}
    eligible = [
        row for row in rows
        if row["raw_site_id"] not in forbidden_sites
        and row["void_family_group"] not in proposal_groups
        and row.get("dft_relax_status") == "missing"
    ]
    by_group = {}
    for row in eligible:
        by_group.setdefault(row["void_family_group"], []).append(row)
    groups = sorted(by_group)
    if len(groups) < count:
        raise ValueError("Too few independent control void families")

    def match_level(proposal, row):
        same_namespace = row["source_namespace"] == proposal["source_namespace"]
        same_geometry = row["geometry"] == proposal["geometry"]
        if same_namespace and same_geometry:
            return 0
        if same_namespace:
            return 1
        if same_geometry:
            return 2
        return 3

    def metrics(proposal, row):
        descriptor = float(np.linalg.norm(vectors[row["raw_site_id"]] - vectors[proposal["raw_site_id"]]))
        energy = abs(float(row["screening_energy_eV"]) - float(proposal["screening_energy_eV"]))
        level = match_level(proposal, row)
        return level, descriptor, energy, 25.0 * level + descriptor + 0.25 * energy

    best_by_proposal_group = {}
    costs = np.empty((count, len(groups)), dtype=float)
    for proposal_index, proposal in enumerate(proposals[:count]):
        for group_index, group in enumerate(groups):
            best = min(
                by_group[group],
                key=lambda row: (*metrics(proposal, row), row["raw_site_id"]),
            )
            best_by_proposal_group[(proposal_index, group_index)] = best
            costs[proposal_index, group_index] = metrics(proposal, best)[-1]
    proposal_indices, group_indices = linear_sum_assignment(costs)
    assigned = dict(zip(proposal_indices, group_indices))
    pairs = []
    for proposal_index, proposal in enumerate(proposals[:count]):
        group_index = assigned[proposal_index]
        control = best_by_proposal_group[(proposal_index, group_index)]
        level, descriptor, _, _ = metrics(proposal, control)
        pairs.append((proposal, control, {
            "match_level": level,
            "descriptor_distance_standardized_l2": descriptor,
            "mace_screening_energy_delta_eV": float(control["screening_energy_eV"]) - float(proposal["screening_energy_eV"]),
            "proposal_energy_quantile": quantiles[proposal["raw_site_id"]],
            "control_energy_quantile": quantiles[control["raw_site_id"]],
        }))
    return pairs


def select_matchable_pairs(rows: list[dict], count: int, max_descriptor_distance: float) -> list[tuple[dict, dict, dict]]:
    ranked = sorted(
        (
            row for row in rows
            if row.get("recommended_rank") not in (None, "")
            and row.get("dft_relax_status") == "missing"
        ),
        key=lambda row: (int(row["recommended_rank"]), row["raw_site_id"]),
    )
    feasible = []
    for proposals in itertools.combinations(ranked, count):
        if len({row["void_family_group"] for row in proposals}) != count:
            continue
        pairs = match_controls(rows, list(proposals), count)
        if not all(
            match["match_level"] == 0
            and match["descriptor_distance_standardized_l2"] <= max_descriptor_distance
            for _, _, match in pairs
        ):
            continue
        objective = (
            sum(int(row["recommended_rank"]) for row in proposals),
            max(match["descriptor_distance_standardized_l2"] for _, _, match in pairs),
            sum(match["descriptor_distance_standardized_l2"] for _, _, match in pairs),
            tuple(row["raw_site_id"] for row in proposals),
        )
        feasible.append((objective, pairs))
    if not feasible:
        raise ValueError("No complete proposal/control assignment passes the matching caliper")
    return min(feasible, key=lambda item: item[0])[1]


def prepare_job(row: dict, arm: str, pair_index: int, args, out_dir: Path, cell_block: str) -> dict:
    source = (args.qe_input_root / row["raw_site_id"] / "relax.in").resolve()
    if not source.is_file() or sha256_file(source) != row["local_input_sha256"]:
        raise ValueError(f"Source input hash mismatch for {row['raw_site_id']}")
    short_arm = "p" if arm == "proposal" else "c"
    job_name = f"mb_sp{short_arm}{pair_index:02d}_s{args.seed}"
    branch_id = f"site_proposer_{arm}_{pair_index:02d}_{row['raw_site_id']}_s{args.seed}"
    branch_dir = out_dir / branch_id
    branch_dir.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^A-Za-z0-9_]", "_", job_name)
    output = branch_dir / "relax.in"
    source_with_cell = inject_reference_cell(source.read_text(), cell_block)
    output.write_text(transform_pw(source_with_cell, prefix, args.max_seconds, args.nstep))
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "prospective_dft_site_proposer_validation",
        "branch_id": branch_id,
        "job_name": job_name,
        "seed": args.seed,
        "arm": arm,
        "pair_index": pair_index,
        "site_id": row["site_id"],
        "raw_site_id": row["raw_site_id"],
        "source_namespace": row["source_namespace"],
        "geometry": row["geometry"],
        "void_family_group": row["void_family_group"],
        "source_pw_input": str(source),
        "source_pw_sha256": sha256_file(source),
        "reference_cell_input": str(args.reference_cell_input.resolve()),
        "reference_cell_input_sha256": sha256_file(args.reference_cell_input),
        "reference_cell_block_sha256": hashlib.sha256(cell_block.encode()).hexdigest(),
        "source_missing_cell_repaired_explicitly": True,
        "source_historical_dft_status": row["dft_relax_status"],
        "selection_was_blind_to_prospective_dft_result": True,
        "mace_screening_energy_eV": float(row["screening_energy_eV"]),
        "oof_predicted_mace_energy_eV": float(row["oof_predicted_mace_energy_eV"]),
        "oof_predictive_std_eV": float(row["oof_predictive_std_eV"]),
        "proposal_score_lower_is_better": float(row["proposal_score_lower_is_better"]),
        "resources": {
            "walltime": args.walltime,
            "pw_max_seconds": args.max_seconds,
            "nstep": args.nstep,
            "ntasks": args.ntasks,
            "memory": args.memory,
        },
        "fixed_cell": True,
        "relax_input": str(output.resolve()),
        "relax_input_sha256": sha256_file(output),
        "acceptance": {
            "job_done": True,
            "bfgs_converged": True,
            "no_scf_nonconvergence": True,
            "final_max_atom_force_eV_A_lte": 0.05,
            "minimum_pair_distance_A_gte": 1.8,
        },
        # Compatibility fields for submit_qe_endpoint_relax_batch.py.
        "source_path_id": row["site_id"],
        "source_image_index_qe": 1,
    }
    manifest_path = branch_dir / "endpoint_relax_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def prepare(args) -> tuple[Path, dict]:
    rows = read_csv(args.site_table)
    proposer = json.loads(args.proposer_manifest.read_text())
    if proposer["decision"].get("dft_stability_model_allowed"):
        raise ValueError("This script is for pre-model prospective validation, not model deployment")
    if not args.prospective_dft_validation:
        raise ValueError("Explicit --prospective-dft-validation acknowledgement is required")
    pairs = select_matchable_pairs(rows, args.pairs, args.max_descriptor_distance)
    cell_block = reference_cell_block(args.reference_cell_input.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, pair_rows = [], []
    for pair_index, (proposal, control, match) in enumerate(pairs, start=1):
        proposal_job = prepare_job(proposal, "proposal", pair_index, args, out_dir, cell_block)
        control_job = prepare_job(control, "control", pair_index, args, out_dir, cell_block)
        jobs.extend([proposal_job, control_job])
        pair_rows.append({
            "pair_index": pair_index,
            "proposal_site": proposal["raw_site_id"],
            "control_site": control["raw_site_id"],
            **match,
        })
    groups = [job["void_family_group"] for job in jobs]
    if len(groups) != len(set(groups)):
        raise ValueError("Prospective batch leaks a void family across selected jobs")
    batch = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "prospective_dft_validation_of_mace_label_site_screening",
        "seed": args.seed,
        "site_table": str(args.site_table.resolve()),
        "site_table_sha256": sha256_file(args.site_table),
        "qe_input_root": str(args.qe_input_root.resolve()),
        "proposer_manifest": str(args.proposer_manifest.resolve()),
        "proposer_manifest_sha256": sha256_file(args.proposer_manifest),
        "reference_cell_input": str(args.reference_cell_input.resolve()),
        "reference_cell_input_sha256": sha256_file(args.reference_cell_input),
        "reference_cell_block_sha256": hashlib.sha256(cell_block.encode()).hexdigest(),
        "selection_frozen_before_dft_submission": True,
        "prospective_outputs_excluded_from_current_proposer": True,
        "pairing_rule": "global one-to-one assignment over unique void families; prioritize same namespace and geometry, then minimize standardized descriptor distance plus MACE-energy difference",
        "matching_caliper": {
            "same_source_namespace": True,
            "same_geometry_label": True,
            "maximum_standardized_descriptor_l2": args.max_descriptor_distance,
        },
        "primary_estimands": [
            "accepted_local_minimum_yield_difference",
            "ionic_steps_distribution",
            "scf_iterations_distribution",
            "basin_collapse_rate",
        ],
        "not_an_estimand": "same-structure_geometry_transform_speedup",
        "pairs": pair_rows,
        "jobs": jobs,
    }
    output = out_dir / "endpoint_relax_batch_manifest.json"
    output.write_text(json.dumps(batch, indent=2) + "\n")
    return output, batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-table", type=Path, required=True)
    parser.add_argument("--qe-input-root", type=Path, required=True)
    parser.add_argument("--proposer-manifest", type=Path, required=True)
    parser.add_argument("--reference-cell-input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--max-descriptor-distance", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-seconds", type=int, default=84600)
    parser.add_argument("--nstep", type=int, default=200)
    parser.add_argument("--ntasks", type=int, default=2)
    parser.add_argument("--memory", default="160G")
    parser.add_argument("--prospective-dft-validation", action="store_true")
    args = parser.parse_args()
    output, batch = prepare(args)
    print(json.dumps({"output": str(output), "pairs": len(batch["pairs"]), "jobs": len(batch["jobs"])}, indent=2))


if __name__ == "__main__":
    main()
