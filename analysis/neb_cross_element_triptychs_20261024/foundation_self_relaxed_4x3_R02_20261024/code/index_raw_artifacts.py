#!/usr/bin/env python3
"""Index raw endpoint-run evidence without copying or rewriting trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "CAMPAIGN_MANIFEST.json").read_text())
    rows = []
    for attempt in sorted((root / "runs" / "idpp").glob("**/attempt_*")):
        if not attempt.is_dir():
            continue
        endpoint = attempt / "endpoint_only_manifest.json"
        engine = attempt / "mlff_neb_manifest.json"
        if endpoint.is_file():
            payload = json.loads(endpoint.read_text())
            rows.append(
                {
                    "record_id": payload.get("record_id", ""),
                    "attempt_id": attempt.name,
                    "attempt_dir": str(attempt),
                    "endpoint_manifest": str(endpoint),
                    "endpoint_manifest_sha256": sha256(endpoint),
                    "engine_manifest": str(engine) if engine.is_file() else "",
                    "engine_manifest_sha256": sha256(engine) if engine.is_file() else "",
                    "trajectory_sha256": payload.get("trajectory_sha256", ""),
                    "scientific_status": payload.get("scientific_status", ""),
                    "publication_status": payload.get("publication_status", ""),
                    "slurm_job_id": payload.get("slurm_job_id", ""),
                    "slurm_array_task_id": payload.get("slurm_array_task_id", ""),
                    "model_id": payload.get("model_id", ""),
                    "element": payload.get("element", ""),
                    "path_id": payload.get("path_id", ""),
                }
            )
    fields = list(rows[0]) if rows else ["record_id"]
    output = root / "data" / "RAW_ARTIFACT_INDEX.tsv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "foundation_self_relaxed_raw_artifact_index_v1",
        "campaign_id": manifest.get("campaign_id", root.name),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(root / "runs" / "idpp"),
        "attempt_manifest_count": len(rows),
        "output": str(output),
        "output_sha256": sha256(output),
        "raw_artifacts_rewritten": False,
    }
    (root / "data" / "RAW_ARTIFACT_INDEX.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
