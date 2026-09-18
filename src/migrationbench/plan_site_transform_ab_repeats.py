#!/usr/bin/env python3
"""Plan independent repeat-relax state machines for a site-transform A/B batch."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from plan_endpoint_discovery_progress import plan as endpoint_plan
from update_site_proposer_transform_ab import validate_pairing


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_direct_batch(source_batch, transform_manifest, source_path):
    direct_by_branch = {row["branch_id"]: row for row in source_batch["jobs"]}
    selected = []
    for transformed in transform_manifest["jobs"]:
        branch = transformed["direct_baseline_branch_id"]
        if branch not in direct_by_branch:
            raise ValueError("Transform references an unknown direct branch")
        row = direct_by_branch[branch]
        if row.get("arm") != "proposal":
            raise ValueError("A/B direct subset contains a non-proposal row")
        selected.append(row)
    if len(selected) != 6 or len({row["branch_id"] for row in selected}) != 6:
        raise ValueError("A/B direct subset must contain six unique proposal rows")
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "site_transform_ab_direct_first_relax_subset",
        "source_batch": str(source_path.resolve()),
        "source_batch_sha256": sha256_file(source_path),
        "jobs": selected,
    }


def filtered_status(status, branches, source_path):
    rows = [row for row in status["jobs"] if row["branch_id"] in branches]
    if {row["branch_id"] for row in rows} != set(branches):
        raise ValueError("Status does not cover the exact paired direct subset")
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_status": str(source_path.resolve()),
        "source_status_sha256": sha256_file(source_path),
        "jobs": rows,
    }


def filtered_endpoint_status(status, endpoint_keys, source_path):
    rows = [
        row for row in status["jobs"]
        if (row["path_id"], int(row["image_index_qe"])) in endpoint_keys
    ]
    observed = {
        (row["path_id"], int(row["image_index_qe"])) for row in rows
    }
    if observed != set(endpoint_keys) or len(rows) != len(observed):
        raise ValueError(
            "Repeat status does not cover the exact paired direct endpoint subset"
        )
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "site_transform_ab_direct_repeat_status_view",
        "source_status": str(source_path.resolve()),
        "source_status_sha256": sha256_file(source_path),
        "jobs": rows,
    }


def command_for_repeat(root, first_status, out_dir, tag):
    return [
        sys.executable,
        str(root / "scripts" / "migrationbench" / "prepare_qe_repeat_relaxations.py"),
        "--first-status", str(first_status),
        "--out-dir", str(out_dir),
        "--branch-tag", tag,
        "--seed", "42",
        "--walltime", "24:00:00",
        "--max-seconds", "84600",
        "--ntasks", "2",
        "--memory", "160G",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--direct-status", type=Path, required=True)
    parser.add_argument("--transformed-status", type=Path, required=True)
    parser.add_argument("--direct-repeat-status", type=Path)
    parser.add_argument("--transformed-repeat-status", type=Path)
    parser.add_argument("--direct-acceptance", type=Path)
    parser.add_argument("--transformed-acceptance", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    experiment = root / "data_processed" / "site_stability_proposer" / "prospective_transform_ab_v1_s42"
    source_batch_path = root / "data_processed" / "site_stability_proposer" / "prospective_dft_v5_s42" / "endpoint_relax_batch_manifest.json"
    direct_registry_path = root / "data_processed" / "site_stability_proposer" / "prospective_dft_v5_s42" / "submitted_endpoint_relax_jobs.json"
    transformed_registry_path = experiment / "submitted_endpoint_relax_jobs.json"
    transformed_batch_path = experiment / "endpoint_relax_batch_manifest.json"
    source_batch = json.loads(source_batch_path.read_text())
    transformed_batch = json.loads(transformed_batch_path.read_text())
    validate_pairing(
        json.loads(direct_registry_path.read_text()),
        json.loads(transformed_registry_path.read_text()),
        transformed_batch,
    )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    direct_batch = paired_direct_batch(source_batch, transformed_batch, source_batch_path)
    direct_batch_path = out_dir / "direct_proposal_batch_manifest.json"
    direct_batch_path.write_text(json.dumps(direct_batch, indent=2) + "\n")
    branches = {row["branch_id"] for row in direct_batch["jobs"]}
    direct_first = filtered_status(
        json.loads(args.direct_status.read_text()), branches, args.direct_status.resolve()
    )
    direct_first_path = out_dir / "direct_proposal_first_status.json"
    direct_first_path.write_text(json.dumps(direct_first, indent=2) + "\n")

    direct_repeat_path = None
    if args.direct_repeat_status:
        accepted_keys = {
            (row["path_id"], int(row["image_index_qe"]))
            for row in direct_first["jobs"]
            if row["classification"] == "accepted_local_minimum"
        }
        direct_repeat = filtered_endpoint_status(
            json.loads(args.direct_repeat_status.read_text()),
            accepted_keys,
            args.direct_repeat_status.resolve(),
        )
        direct_repeat_path = out_dir / "direct_proposal_repeat_status.json"
        direct_repeat_path.write_text(json.dumps(direct_repeat, indent=2) + "\n")

    direct_plan = endpoint_plan(
        direct_batch_path,
        direct_first_path,
        direct_repeat_path,
        args.direct_acceptance,
    )
    transformed_plan = endpoint_plan(
        transformed_batch_path,
        args.transformed_status.resolve(),
        args.transformed_repeat_status,
        args.transformed_acceptance,
    )
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "site_transform_ab_repeat_state_plan",
        "direct": direct_plan,
        "transformed": transformed_plan,
        "repeat_prepare_commands": {
            "direct": command_for_repeat(
                root, direct_first_path, out_dir / "direct_repeats", "direct"
            ) if direct_plan["stage"] == "prepare_repeat_relaxations" else None,
            "transformed": command_for_repeat(
                root, args.transformed_status.resolve(), out_dir / "transformed_repeats", "geomvoid"
            ) if transformed_plan["stage"] == "prepare_repeat_relaxations" else None,
        },
    }
    output = out_dir / "repeat_state_plan.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "direct_stage": direct_plan["stage"],
        "transformed_stage": transformed_plan["stage"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
