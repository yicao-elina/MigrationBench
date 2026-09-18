#!/usr/bin/env python3
"""Monitor and rebuild the prospective site-proposer transform A/B result."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command):
    subprocess.run([str(value) for value in command], check=True)


def validate_pairing(direct_registry, transformed_registry, transform_manifest):
    direct = {row["branch_id"]: str(row["job_id"]) for row in direct_registry["jobs"]}
    transformed = {
        row["branch_id"]: str(row["job_id"]) for row in transformed_registry["jobs"]
    }
    rows = transform_manifest["jobs"]
    if len(rows) != 6 or len(transformed) != 6:
        raise ValueError("Prospective transform A/B requires exactly six transformed jobs")
    keys = []
    for row in rows:
        if row.get("source_arm") != "proposal":
            raise ValueError("Transform batch contains a non-proposal source")
        if direct.get(row["direct_baseline_branch_id"]) != str(row["direct_baseline_job_id"]):
            raise ValueError("Direct baseline job binding mismatch")
        if row["branch_id"] not in transformed:
            raise ValueError("Transformed submission binding is missing")
        keys.append((row["source_path_id"], int(row["source_image_index_qe"])))
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate site/image key in transform batch")
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-direct-monitor", action="store_true")
    parser.add_argument("--skip-transformed-monitor", action="store_true")
    parser.add_argument("--calculator-acceptance", type=Path)
    parser.add_argument("--endpoint-acceptance", type=Path)
    parser.add_argument("--direct-endpoint-acceptance", type=Path)
    parser.add_argument("--transformed-endpoint-acceptance", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    scripts = root / "scripts" / "migrationbench"
    direct_root = root / "data_processed" / "site_stability_proposer" / "prospective_dft_v5_s42"
    transform_root = root / "data_processed" / "site_stability_proposer" / "prospective_transform_ab_v1_s42"
    direct_jobs = direct_root / "submitted_endpoint_relax_jobs.json"
    transformed_jobs = transform_root / "submitted_endpoint_relax_jobs.json"
    transform_manifest_path = transform_root / "endpoint_relax_batch_manifest.json"
    direct_status = root / "state" / "qe_site_proposer_prospective_dft_status_s42.json"
    transformed_status = root / "state" / "qe_site_proposer_transform_status_s42.json"
    cluster_root = root / "data_processed" / "cluster"

    direct_registry = json.loads(direct_jobs.read_text())
    transformed_registry = json.loads(transformed_jobs.read_text())
    transform_manifest = json.loads(transform_manifest_path.read_text())
    expected_keys = validate_pairing(
        direct_registry, transformed_registry, transform_manifest
    )

    if not args.skip_direct_monitor:
        run([
            sys.executable, scripts / "monitor_qe_relax_jobs.py",
            "--jobs-file", direct_jobs,
            "--local-cluster-root", cluster_root,
            "--output", direct_status,
        ])
    if not args.skip_transformed_monitor:
        run([
            sys.executable, scripts / "monitor_qe_relax_jobs.py",
            "--jobs-file", transformed_jobs,
            "--local-cluster-root", cluster_root,
            "--output", transformed_status,
        ])
    if not direct_status.is_file() or not transformed_status.is_file():
        raise FileNotFoundError("A/B status files are missing")

    comparison_dir = transform_root / "comparison"
    command = [
        sys.executable, scripts / "compare_endpoint_relaxation_speed.py",
        "--direct-status", direct_status,
        "--transformed-status", transformed_status,
        "--transform-manifest", transform_manifest_path,
        "--out-dir", comparison_dir,
    ]
    if args.calculator_acceptance:
        command.extend(["--calculator-acceptance", args.calculator_acceptance.resolve()])
    if args.endpoint_acceptance:
        command.extend(["--endpoint-acceptance", args.endpoint_acceptance.resolve()])
    if args.direct_endpoint_acceptance:
        command.extend([
            "--direct-endpoint-acceptance",
            args.direct_endpoint_acceptance.resolve(),
        ])
    if args.transformed_endpoint_acceptance:
        command.extend([
            "--transformed-endpoint-acceptance",
            args.transformed_endpoint_acceptance.resolve(),
        ])
    run(command)

    comparison_path = comparison_dir / "paired_relaxation_comparison.json"
    comparison = json.loads(comparison_path.read_text())
    observed_keys = {
        (row["path_id"], int(row["image_index_qe"])) for row in comparison["rows"]
    }
    if observed_keys != set(expected_keys):
        raise ValueError("Comparison does not cover the exact frozen six-site batch")
    summary = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "prospective_same_structure_direct_vs_transform_qe_relax",
        "pair_count": len(comparison["rows"]),
        "formal_speedup_count": sum(
            row["comparison_status"] == "comparable_same_basin_accepted_calculator"
            for row in comparison["rows"]
        ),
        "comparison_statuses": sorted({
            row["comparison_status"] for row in comparison["rows"]
        }),
        "formal_result_available": all(
            row["comparison_status"] in {
                "comparable_same_basin_accepted_calculator",
                "basin_changed_no_speedup",
                "same_basin_repeat_acceptance_rejected",
            }
            for row in comparison["rows"]
        ),
        "inputs": {
            "direct_status": str(direct_status),
            "direct_status_sha256": sha256_file(direct_status),
            "transformed_status": str(transformed_status),
            "transformed_status_sha256": sha256_file(transformed_status),
            "transform_manifest": str(transform_manifest_path),
            "transform_manifest_sha256": sha256_file(transform_manifest_path),
            "comparison": str(comparison_path),
            "comparison_sha256": sha256_file(comparison_path),
            "endpoint_acceptance": (
                str(args.endpoint_acceptance.resolve()) if args.endpoint_acceptance else None
            ),
            "direct_endpoint_acceptance": (
                str(args.direct_endpoint_acceptance.resolve())
                if args.direct_endpoint_acceptance else None
            ),
            "direct_endpoint_acceptance_sha256": (
                sha256_file(args.direct_endpoint_acceptance.resolve())
                if args.direct_endpoint_acceptance else None
            ),
            "transformed_endpoint_acceptance": (
                str(args.transformed_endpoint_acceptance.resolve())
                if args.transformed_endpoint_acceptance else None
            ),
            "transformed_endpoint_acceptance_sha256": (
                sha256_file(args.transformed_endpoint_acceptance.resolve())
                if args.transformed_endpoint_acceptance else None
            ),
        },
        "rows": comparison["rows"],
    }
    summary_path = comparison_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    dataset_dir = (
        root / "data_processed" / "hf_dataset_staging"
        / "site_proposer_transform_ab_v1_s42"
    )
    run([
        sys.executable, scripts / "build_relaxation_pair_dataset.py",
        "--experiment-id", "site_proposer_transform_ab_v1_s42",
        "--transform-manifest", transform_manifest_path,
        "--comparison", comparison_path,
        "--out-dir", dataset_dir,
    ])
    print(json.dumps({
        "pairs": summary["pair_count"],
        "formal_speedups": summary["formal_speedup_count"],
        "statuses": summary["comparison_statuses"],
        "summary": str(summary_path),
        "dataset": str(dataset_dir / "dataset_manifest.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
