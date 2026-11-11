#!/usr/bin/env python3
"""Audit the R02 canonical endpoint layer and target table before/after execution."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = list(csv.DictReader((root / "PATH_REGISTRY.tsv").open(), delimiter="\t"))
    jobs = list(csv.DictReader((root / "JOB_TABLE_IDPP.tsv").open(), delimiter="\t"))
    errors: list[str] = []
    endpoint_count = 0
    for row in paths:
        for role in ("initial", "final"):
            endpoint_count += 1
            path = root / row[f"{role}_endpoint_file"]
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            lines = path.read_text().splitlines()
            if len(lines) != int(row["atoms"]) + 2:
                errors.append(f"line count {path.name}")
            header = lines[1]
            required = [
                "campaign_id=foundation_self_relaxed_4x3_20261024_r02",
                f"path_id={row['path_id']}",
                f"element={row['element']}",
                f"role={role}",
                "endpoint_only_input=T",
                f"cell_sha256={row['cell_sha256']}",
                f"pbc=\"{row['pbc']}\"",
            ]
            for token in required:
                if token not in header:
                    errors.append(f"missing header token {token} in {path.name}")
            expected_hash = row[f"{role}_endpoint_sha256"]
            if sha256(path) != expected_hash:
                errors.append(f"hash mismatch {path.name}")
            for value in lines[2:]:
                parts = value.split()
                if len(parts) < 4 or not all(math.isfinite(float(x)) for x in parts[1:4]):
                    errors.append(f"non-finite coordinate in {path.name}")
                    break
    ids = [row["record_id"] for row in jobs]
    counts = {
        "elements": {element: sum(row["element"] == element for row in jobs) for element in ("Cr", "Cu", "Li", "Na")},
        "path_classes": {path_class: sum(row["path_class"] == path_class for row in jobs) for path_class in ("in_gap", "gap_to_ql", "within_ql")},
        "models": len({row["model_id"] for row in jobs}),
    }
    if len(jobs) != 108 or len(set(ids)) != 108:
        errors.append("target job table is not 108 unique rows")
    result = {
        "schema_version": "canonical_input_audit_v1",
        "campaign_id": "foundation_self_relaxed_4x3_20261024_r02",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint_files": endpoint_count,
        "job_rows": len(jobs),
        "counts": counts,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    output = root / "data/PREEXECUTION_DATA_AUDIT_R02.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

