#!/usr/bin/env python3
"""Build a diagnostic MACE candidate manifest from audited periodic path repairs."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate(path_id, repair_path, output_dir, seed):
    repair_path = repair_path.resolve()
    repair = json.loads(repair_path.read_text())
    images = Path(repair["output_images"]).resolve()
    checks = {
        "repair_output_hash": repair.get("output_sha256") == sha256_file(images),
        "periodic_cell": repair.get("output_pbc") == [True, True, True],
        "endpoints": repair.get("endpoints_pass") is True,
        "intermediate_images": repair.get("all_intermediate_images_pass") is True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"{path_id} repair gate failed: {checks}")
    try:
        relative_images = images.relative_to(output_dir.resolve())
    except ValueError as error:
        raise ValueError("Images must be stored beside the candidate manifest") from error
    return {
        "branch_id": f"{path_id}_historical_periodic_direct_s{seed}",
        "path_id": path_id,
        "images": str(relative_images),
        "images_sha256": sha256_file(images),
        "source_repair_manifest": str(repair_path),
        "source_repair_manifest_sha256": sha256_file(repair_path),
        "source_images": repair["source_images"],
        "source_images_sha256": repair["source_sha256"],
        "cell_template": repair["cell_template"],
        "cell_template_sha256": repair["cell_template_sha256"],
        "geometry_metrics": repair["after"],
        "geometry_gate": {"status": "pass", "checks": checks},
        "duplicate_of": None,
        "production_eligible_before_mace": False,
        "scientific_role": "diagnostic_preconditioner_for_unconverged_historical_qe_path",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--path-id", action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output.resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        candidate(
            path_id,
            args.input_dir / f"{path_id}_periodic_direct.manifest.json",
            output_dir,
            args.seed,
        )
        for path_id in args.path_id
    ]
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint_status": "unverified_historical_neb_endpoints",
        "seed": args.seed,
        "purpose": "close_MACE_preconditioning_gaps_for_every_unconverged_path_queue_row",
        "candidates": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidates": len(rows), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
