#!/usr/bin/env python3
"""Monitor and rebuild the coverage-gap QE geometry-transform A/B comparison."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(command):
    subprocess.run([str(value) for value in command], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-monitor", action="store_true")
    parser.add_argument("--calculator-acceptance", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    scripts = root / "scripts" / "migrationbench"
    direct_jobs = root / "configs" / "qe_geometry_transform_relax_direct_jobs_s47.json"
    transformed_jobs = root / "configs" / "qe_geometry_transform_relax_transformed_jobs_s47.json"
    direct_status = root / "state" / "qe_geometry_transform_direct_status_s47.json"
    transformed_status = root / "state" / "qe_geometry_transform_transformed_status_s47.json"
    if not args.skip_monitor:
        run([sys.executable, scripts / "monitor_qe_relax_jobs.py", "--jobs-file", direct_jobs, "--output", direct_status])
        run([sys.executable, scripts / "monitor_qe_relax_jobs.py", "--jobs-file", transformed_jobs, "--output", transformed_status])
    if not direct_status.exists() or not transformed_status.exists():
        raise FileNotFoundError("A/B status files do not exist; run without --skip-monitor")
    summaries = []
    for path_id in ("1-3", "1-5"):
        command = [
            sys.executable,
            scripts / "compare_endpoint_relaxation_speed.py",
            "--direct-status", direct_status,
            "--transformed-status", transformed_status,
            "--transform-manifest", root / "data_processed" / "qe_geometry_transform_relax_ab_s47" / path_id / "paired_experiment_manifest.json",
            "--out-dir", root / "data_processed" / "qe_geometry_transform_relax_ab_s47" / "comparison" / path_id,
        ]
        if args.calculator_acceptance:
            command.extend(["--calculator-acceptance", args.calculator_acceptance.resolve()])
        run(command)
        comparison = json.loads((root / "data_processed" / "qe_geometry_transform_relax_ab_s47" / "comparison" / path_id / "paired_relaxation_comparison.json").read_text())
        summaries.extend(comparison["rows"])
    print(json.dumps({
        "pairs": len(summaries),
        "formal_speedups": sum(row["comparison_status"] == "comparable_same_basin_accepted_calculator" for row in summaries),
        "statuses": sorted({row["comparison_status"] for row in summaries}),
    }, indent=2))


if __name__ == "__main__":
    main()
