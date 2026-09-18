#!/usr/bin/env python3
"""Monitor corrected periodic MLFF A/B jobs and compare terminal pairs."""

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
    args = parser.parse_args()
    root = args.root.resolve()
    scripts = root / "scripts" / "migrationbench"
    jobs = root / "configs" / "active_mlff_diagnostic_jobs.json"
    config = root / "configs" / "mlff_repair_ab_coverage_gaps_pbc_s47.json"
    graph_config = root / "configs" / "mlff_clearance_graph_1-5_s48.json"
    status_path = root / "state" / "mlff_diagnostic_job_status.json"
    if not args.skip_monitor:
        run([
            sys.executable, scripts / "monitor_mlff_neb_jobs.py",
            "--jobs-file", jobs,
            "--experiment-config", config,
            "--experiment-config", graph_config,
            "--output", status_path,
        ])
    status = json.loads(status_path.read_text())
    by_id = {str(row["job_id"]): row for row in status["jobs"]}
    experiment = json.loads(config.read_text())
    outcomes = []
    for path_id in ("1-3", "1-5"):
        pair = [row for row in experiment["jobs"] if row["path_id"] == path_id]
        direct = next(row for row in pair if row["variant"] == "periodic_direct")
        repaired = next(row for row in pair if row["variant"] == "periodic_repaired")
        direct_status, repaired_status = by_id[str(direct["job_id"])], by_id[str(repaired["job_id"])]
        terminal = all(row["classification"] not in {"pending", "running", "configuring", "completing"} for row in (direct_status, repaired_status))
        if not terminal or not direct_status.get("manifest") or not repaired_status.get("manifest"):
            outcomes.append({"path_id": path_id, "decision": "waiting_for_terminal_pair"})
            continue
        out_dir = root / "data_processed" / "mlff_repair_ab" / "coverage_gaps_pbc_s47"
        repair_manifest = root / "data_processed" / "path_repairs" / "coverage_gaps_pbc_v3" / f"{path_id}_periodic_repaired.manifest.json"
        run([
            sys.executable, scripts / "compare_mlff_repair_ab.py",
            "--path-id", path_id,
            "--direct-manifest", direct_status["manifest"],
            "--repaired-manifest", repaired_status["manifest"],
            "--repair-manifest", repair_manifest,
            "--config", config,
            "--out-dir", out_dir,
        ])
        result = json.loads((out_dir / f"{path_id}_mlff_repair_ab.json").read_text())
        outcomes.append({"path_id": path_id, "decision": result["decision"], "similarity": result["final_path_similarity"]})
    print(json.dumps({"outcomes": outcomes}, indent=2))


if __name__ == "__main__":
    main()
