#!/usr/bin/env python3
"""Cadence-gated monitor and comparator for the exact-iteration QE A/B pairs."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(command):
    subprocess.run([str(value) for value in command], check=True)


def hours_since_status(paths, now=None):
    timestamps = []
    for path in paths:
        if not path.exists():
            continue
        value = json.loads(path.read_text()).get("created_at_utc")
        if value:
            timestamps.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
    if not timestamps:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - max(timestamps)).total_seconds() / 3600.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-monitor", action="store_true")
    parser.add_argument("--force-poll", action="store_true")
    parser.add_argument("--min-poll-hours", type=float, default=3.0)
    parser.add_argument("--calculator-acceptance", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    scripts = root / "scripts" / "migrationbench"
    cluster_root = root / "cluster"
    status_paths = {
        "direct": root / "state" / "qe_endpoint_relax_exact_direct_job_status.json",
        "transformed": root / "state" / "qe_endpoint_relax_exact_transformed_job_status.json",
    }
    pair_roots = {
        "1-6": root / "data_processed" / "endpoint_relax_inputs" / "exact_iter_pairs" / "1-6_iter108_s42",
        "1-7": root / "data_processed" / "endpoint_relax_inputs" / "exact_iter_pairs" / "1-7_iter112_s42",
    }

    if not args.skip_monitor:
        age = hours_since_status(status_paths.values())
        if age is not None and age < args.min_poll_hours and not args.force_poll:
            raise RuntimeError(
                "Refusing high-frequency Rockfish poll: newest status is "
                f"{age:.2f} h old; minimum is {args.min_poll_hours:.2f} h"
            )
        for variant in ("direct", "transformed"):
            command = [sys.executable, scripts / "monitor_qe_relax_jobs.py"]
            for pair_root in pair_roots.values():
                command.extend([
                    "--jobs-file",
                    pair_root / variant / "submitted_endpoint_relax_jobs.json",
                ])
            command.extend([
                "--local-cluster-root", cluster_root,
                "--output", status_paths[variant],
            ])
            run(command)

    for path in status_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    rows = []
    for path_id, pair_root in pair_roots.items():
        out_dir = (
            root / "data_processed" / "endpoint_relax_comparison" / "exact_iter"
            / ("1-6_iter108" if path_id == "1-6" else "1-7_iter112")
        )
        command = [
            sys.executable,
            scripts / "compare_endpoint_relaxation_speed.py",
            "--direct-status", status_paths["direct"],
            "--transformed-status", status_paths["transformed"],
            "--transform-manifest", pair_root / "paired_experiment_manifest.json",
            "--out-dir", out_dir,
        ]
        if args.calculator_acceptance:
            command.extend(["--calculator-acceptance", args.calculator_acceptance.resolve()])
        run(command)
        rows.extend(json.loads((out_dir / "paired_relaxation_comparison.json").read_text())["rows"])

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "exact_iteration_direct_vs_geometry_transform_qe_relax",
        "pair_count": len(rows),
        "formal_speedup_count": sum(
            row["comparison_status"] == "comparable_same_basin_accepted_calculator"
            for row in rows
        ),
        "comparison_statuses": sorted({row["comparison_status"] for row in rows}),
        "rows": rows,
    }
    output = root / "data_processed" / "endpoint_relax_comparison" / "exact_iter" / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
