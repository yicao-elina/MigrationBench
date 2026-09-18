#!/usr/bin/env python3
"""Build and validate a lossless relaxation-pair table for MigrationBench."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


FORMAL_STATUS = "comparable_same_basin_accepted_calculator"


def stable_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_id(value):
    return "RP_" + hashlib.sha256(stable_json(value).encode()).hexdigest()[:32]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(rows, out_dir):
    jsonl = out_dir / "relaxation_pairs.jsonl"
    csv_path = out_dir / "relaxation_pairs.csv"
    jsonl.write_text("".join(stable_json(row) + "\n" for row in rows))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: stable_json(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })
    return {
        "jsonl": str(jsonl),
        "jsonl_sha256": sha256_file(jsonl),
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "rows": len(rows),
    }


def build_rows(experiment_id, transform_manifest, comparison, manifest_path, comparison_path):
    transforms = {
        (row["source_path_id"], int(row["source_image_index_qe"])): row
        for row in transform_manifest["jobs"]
    }
    comparisons = {
        (row["path_id"], int(row["image_index_qe"])): row
        for row in comparison["rows"]
    }
    if set(transforms) != set(comparisons):
        raise ValueError("Transform manifest and comparison cover different pairs")
    rows = []
    for key in sorted(transforms):
        transform = transforms[key]
        result = comparisons[key]
        direct_input = Path(transform["source_pw_input"])
        transformed_input = Path(transform["relax_input"])
        if not direct_input.is_file() or sha256_file(direct_input) != transform["source_pw_sha256"]:
            raise ValueError("Direct input hash mismatch")
        if not transformed_input.is_file() or sha256_file(transformed_input) != transform["relax_input_sha256"]:
            raise ValueError("Transformed input hash mismatch")
        formal = result["comparison_status"] == FORMAL_STATUS
        row = {
            "relaxation_pair_id": stable_id({
                "experiment_id": experiment_id,
                "path_id": key[0],
                "image_index_qe": key[1],
                "direct_input_sha256": transform["source_pw_sha256"],
                "transformed_input_sha256": transform["relax_input_sha256"],
            }),
            "experiment_id": experiment_id,
            "scientific_role": "same_structure_relaxation_efficiency",
            "path_id": key[0],
            "image_index_qe": key[1],
            "source_arm": transform.get("source_arm"),
            "source_pair_index": transform.get("source_pair_index"),
            "direct_branch_id": transform["direct_baseline_branch_id"],
            "direct_job_id": str(result["direct_job_id"]),
            "direct_input": str(direct_input.resolve()),
            "direct_input_sha256": transform["source_pw_sha256"],
            "transformed_branch_id": transform["branch_id"],
            "transformed_job_id": str(result["transformed_job_id"]),
            "transformed_input": str(transformed_input.resolve()),
            "transformed_input_sha256": transform["relax_input_sha256"],
            "transform": transform["transform"],
            "direct_calculator_identity": result["direct_calculator_identity"],
            "transformed_calculator_identity": result["transformed_calculator_identity"],
            "calculator_identities_match": result["calculator_identities_match"],
            "calculator_identity_accepted": result["calculator_identity_accepted_by_n24"],
            "direct_status": result["direct_status"],
            "transformed_status": result["transformed_status"],
            "direct_repeat_acceptance": result.get("direct_repeat_acceptance", "pending"),
            "transformed_repeat_acceptance": result.get("transformed_repeat_acceptance", "pending"),
            "basin_equivalence": result["basin_equivalence"],
            "basin_metrics": {
                "cr_distance_A": result.get("final_cr_distance_A"),
                "host_rmsd_A": result.get("final_host_rmsd_A"),
                "local_signature_rms_A": result.get("final_local_signature_rms_A"),
            },
            "work_metrics": {
                "direct_ionic_steps": result["direct_ionic_steps"],
                "transformed_ionic_steps": result["transformed_ionic_steps"],
                "direct_scf_iterations": result["direct_scf_iterations"],
                "transformed_scf_iterations": result["transformed_scf_iterations"],
                "direct_elapsed_seconds": result["direct_elapsed_seconds"],
                "transformed_elapsed_seconds": result["transformed_elapsed_seconds"],
            },
            "speedup_metrics": {
                "ionic_step_speedup": result["ionic_step_speedup"],
                "scf_iteration_speedup": result["scf_iteration_speedup"],
                "walltime_speedup": result["walltime_speedup"],
            },
            "comparison_status": result["comparison_status"],
            "formal_speedup": formal,
            "runtime_provenance_complete": result["lineage_runtime_provenance_complete"],
            "prospective_output_role": "evaluation_only",
            "eligible_for_current_proposer_training": False,
            "metadata_json": {
                "direct_lineage_runtime_provenance": result["direct_lineage_runtime_provenance"],
                "transformed_lineage_runtime_provenance": result["transformed_lineage_runtime_provenance"],
                "transform_manifest": str(manifest_path.resolve()),
                "transform_manifest_sha256": sha256_file(manifest_path),
                "comparison": str(comparison_path.resolve()),
                "comparison_sha256": sha256_file(comparison_path),
                "repeat_acceptance_artifact": result.get("repeat_acceptance_artifact"),
                "direct_repeat_acceptance_artifact": result.get(
                    "direct_repeat_acceptance_artifact"
                ),
                "direct_repeat_acceptance_artifact_sha256": result.get(
                    "direct_repeat_acceptance_artifact_sha256"
                ),
                "transformed_repeat_acceptance_artifact": result.get(
                    "transformed_repeat_acceptance_artifact"
                ),
                "transformed_repeat_acceptance_artifact_sha256": result.get(
                    "transformed_repeat_acceptance_artifact_sha256"
                ),
            },
        }
        rows.append(row)
    return rows


def validate_rows(rows):
    failures = []
    ids = [row["relaxation_pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        failures.append("duplicate relaxation_pair_id")
    for row in rows:
        if row["eligible_for_current_proposer_training"] is not False:
            failures.append("prospective evaluation row entered current proposer training")
        speedups = row["speedup_metrics"]
        if row["formal_speedup"]:
            if not (
                row["comparison_status"] == FORMAL_STATUS
                and row["basin_equivalence"] == "same"
                and row["direct_repeat_acceptance"] == "accepted"
                and row["transformed_repeat_acceptance"] == "accepted"
                and row["runtime_provenance_complete"] is True
                and row["calculator_identity_accepted"] is True
                and all(speedups[key] is not None for key in speedups)
            ):
                failures.append(f"formal speedup gate failure: {row['relaxation_pair_id']}")
            metadata = row.get("metadata_json", {})
            for arm in ("direct", "transformed"):
                artifact = metadata.get(f"{arm}_repeat_acceptance_artifact")
                expected = metadata.get(
                    f"{arm}_repeat_acceptance_artifact_sha256"
                )
                artifact_path = Path(artifact) if artifact else None
                if not (
                    artifact_path
                    and artifact_path.is_file()
                    and expected
                    and sha256_file(artifact_path) == expected
                ):
                    failures.append(
                        f"formal speedup {arm} acceptance provenance failure: "
                        f"{row['relaxation_pair_id']}"
                    )
        elif any(speedups[key] is not None for key in speedups):
            failures.append(f"non-formal row contains speedup: {row['relaxation_pair_id']}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--transform-manifest", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.transform_manifest.resolve()
    comparison_path = args.comparison.resolve()
    rows = build_rows(
        args.experiment_id,
        json.loads(manifest_path.read_text()),
        json.loads(comparison_path.read_text()),
        manifest_path,
        comparison_path,
    )
    failures = validate_rows(rows)
    if failures:
        raise ValueError("; ".join(failures))
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table = write_rows(rows, out_dir)
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_repo_id": "alinacao2000/MigrationBench",
        "publication_status": "local_staging_evaluation_only",
        "table": table,
        "formal_speedup_rows": sum(row["formal_speedup"] for row in rows),
        "leakage_policy": "prospective A/B outputs are evaluation-only and excluded from the frozen proposer",
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
