#!/usr/bin/env python3
"""Run configured, provenance-complete QE NEB history exports."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--summary", type=Path, default=Path("state/qe_neb_full_history_export_status.json"))
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    config = json.loads(args.config.read_text())
    exporter = Path(__file__).with_name("parse_qe_neb_full_history.py")
    results = []
    for item in config["exports"]:
        command = [
            sys.executable,
            str(exporter),
            "--run-dir",
            str(workspace / item["run_dir"]),
            "--pathway-id",
            item["pathway_id"],
            "--out-dir",
            str(workspace / item["out_dir"]),
        ]
        for neb_output in item["neb_lineage"]:
            command.extend(["--neb-out", str(workspace / neb_output)])
        completed = subprocess.run(command, text=True, capture_output=True)
        result = {
            "pathway_id": item["pathway_id"],
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "out_dir": str(workspace / item["out_dir"]),
        }
        results.append(result)
        if completed.returncode:
            raise RuntimeError(json.dumps(result, indent=2))
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.resolve()),
        "exports": results,
    }
    output = workspace / args.summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"exports_completed": len(results), "summary": str(output)}, indent=2))


if __name__ == "__main__":
    main()
