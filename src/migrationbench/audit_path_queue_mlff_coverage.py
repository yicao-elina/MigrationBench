#!/usr/bin/env python3
"""Validate one-to-one MACE preconditioning coverage for the NEB path queue."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, queue_path, evidence_path, registry_path):
    queue = yaml.safe_load(queue_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    registry = json.loads(registry_path.read_text())
    queue_ids = [row["path_id"] for row in queue["paths"]]
    rows = evidence["paths"]
    evidence_ids = [row["path_id"] for row in rows]
    if len(set(queue_ids)) != len(queue_ids) or len(set(evidence_ids)) != len(evidence_ids):
        raise RuntimeError("Duplicate path IDs in queue or evidence ledger")
    if set(queue_ids) != set(evidence_ids):
        raise RuntimeError(
            f"Queue/evidence mismatch: missing={sorted(set(queue_ids)-set(evidence_ids))}, "
            f"extra={sorted(set(evidence_ids)-set(queue_ids))}"
        )
    active = {str(row["job_id"]): row for row in registry["jobs"]}
    audited = []
    for row in rows:
        checks = {}
        if row["status"] == "terminal_local_asset":
            for key in ("mlff_manifest", "iteration_history_manifest", "qe_handoff_manifest"):
                path = root / row[key]
                checks[key] = {
                    "exists": path.is_file(),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
        elif row["status"] == "active_registered":
            job = active.get(str(row["job_id"]))
            checks["active_registry"] = {
                "exists": job is not None,
                "branch_matches": bool(job and job.get("branch_id") == row["branch_id"]),
            }
            candidate_manifest = root / row["candidate_manifest"]
            checks["candidate_manifest"] = {
                "exists": candidate_manifest.is_file(),
                "sha256": sha256_file(candidate_manifest) if candidate_manifest.is_file() else None,
            }
        else:
            raise ValueError(f"Unknown status {row['status']}")
        passed = all(
            all(value for key, value in check.items() if key != "sha256")
            for check in checks.values()
        )
        audited.append({**row, "checks": checks, "coverage_gate_pass": passed})
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue_sha256": sha256_file(queue_path),
        "evidence_sha256": sha256_file(evidence_path),
        "queue_paths": len(queue_ids),
        "terminal_local_assets": sum(row["status"] == "terminal_local_asset" for row in audited),
        "active_registered": sum(row["status"] == "active_registered" for row in audited),
        "uncovered": [row["path_id"] for row in audited if not row["coverage_gate_pass"]],
        "rows": audited,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--queue", type=Path, default=Path("configs/migrationbench_path_queue.yaml"))
    parser.add_argument("--evidence", type=Path, default=Path("configs/path_queue_mlff_evidence_s42.json"))
    parser.add_argument("--active-registry", type=Path, default=Path("configs/mlff_path_queue_gap_jobs_s42.json"))
    parser.add_argument("--output", type=Path, default=Path("data_processed/path_queue_mlff_coverage/audit.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    result = audit(root, resolve(args.queue), resolve(args.evidence), resolve(args.active_registry))
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Path Queue MACE Coverage", "",
        "Coverage means a local terminal MACE asset or a registered active MACE job. It does not mean convergence or scientific acceptance.", "",
        "| Path | State | Coverage gate |", "|---|---|---|",
    ]
    lines.extend(
        f"| `{row['path_id']}` | `{row['status']}` | {'pass' if row['coverage_gate_pass'] else 'fail'} |"
        for row in result["rows"]
    )
    lines.extend(["", f"Uncovered paths: `{result['uncovered']}`"])
    (output.parent / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
