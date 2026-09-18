#!/usr/bin/env python3
"""Materialize the k-point sensitivity config only after a passed cutoff gate."""

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_kpoint_config(analysis, source_config, out_dir):
    if analysis.get("gate") != "pass":
        raise RuntimeError(
            "Cutoff stage must have gate='pass'; got "
            f"{analysis.get('gate')!r}: {analysis.get('gate_reason')}"
        )
    reference_id = analysis["reference_profile"]
    reference = next(
        (profile for profile in source_config["profiles"] if profile["id"] == reference_id),
        None,
    )
    if reference is None:
        raise ValueError(f"Reference profile {reference_id!r} is absent from source config")
    gamma = copy.deepcopy(reference)
    gamma["id"] = f"{reference_id}_kpoint_reference"
    gamma["job_slug"] = "krefg"
    gamma["kpoints"] = {"mode": "gamma"}
    dense = copy.deepcopy(reference)
    dense["id"] = f"{reference_id}_kpoint_2x2x1"
    dense["job_slug"] = "k221"
    dense["kpoints"] = {"mode": "automatic", "grid_shift": [2, 2, 1, 0, 0, 0]}
    # Keep pools at one until the actual irreducible k-point count is known.
    dense["kpoint_pools"] = 1
    config = {
        key: copy.deepcopy(source_config[key])
        for key in (
            "path_id", "source_run_dir", "source_iteration", "image_indices",
            "geometry_roles", "pair_delta_E_tolerance_eV", "seed", "resources",
        )
    }
    config.update({
        "stage": "kpoint_sensitivity",
        "blocked_by": None,
        "upstream_gate": {
            "stage": "cutoff_sensitivity",
            "gate": analysis["gate"],
            "gate_reason": analysis.get("gate_reason"),
        },
        "out_dir": str(out_dir),
        "profiles": [gamma, dense],
    })
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    source_config = json.loads(args.source_config.read_text())
    config = build_kpoint_config(analysis, source_config, args.out_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    receipt = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": "materialized_kpoint_config_not_submitted",
        "analysis": str(args.analysis.resolve()),
        "analysis_sha256": sha256_file(args.analysis),
        "source_config": str(args.source_config.resolve()),
        "source_config_sha256": sha256_file(args.source_config),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
