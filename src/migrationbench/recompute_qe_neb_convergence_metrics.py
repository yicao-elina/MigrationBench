#!/usr/bin/env python3
"""Recompute QE NEB convergence metrics with movable/all-image force separation."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from parse_qe_neb_output import parse_neb_out


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    rows = []
    for item in config["runs"]:
        path = Path(item["neb_out"]).resolve()
        parsed = parse_neb_out(path)
        rows.append({
            "path_id": item["path_id"],
            "branch_id": item["branch_id"],
            "job_id": str(item["job_id"]),
            "neb_out": str(path),
            "neb_out_sha256": sha256_file(path),
            "last_complete_iteration": parsed["last_complete_iteration"],
            "barrier_forward_eV": parsed["activation_forward_eV"],
            "barrier_reverse_eV": parsed["activation_reverse_eV"],
            "max_movable_error_eV_A": parsed["max_image_error_movable_eV_A"],
            "max_all_error_eV_A": parsed["max_image_error_all_eV_A"],
            "barrier_drift_last_three_eV": parsed["barrier_drift_last_three_eV"],
            "last_iteration_complete": parsed["last_iteration_complete"],
            "job_done": parsed["job_done"],
            "converged_by_default_gate": parsed["converged_by_default_gate"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_config": str(args.config.resolve()),
        "source_config_sha256": sha256_file(args.config),
        "metric_definition": {
            "max_movable_error_eV_A": "maximum QE NEB error over images with frozen=F; canonical convergence metric",
            "max_all_error_eV_A": "diagnostic maximum over all images including frozen endpoints",
            "default_gate": "JOB DONE, complete final activation table, max movable error <=0.03 eV/A, barrier drift <=0.02 eV",
        },
        "rows": rows,
    }
    json_path = args.output_dir / "frozen_endpoint_error_correction.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    csv_path = args.output_dir / "frozen_endpoint_error_correction.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "runs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
