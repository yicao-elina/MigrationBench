#!/usr/bin/env python3
"""Prove that each prospective A/B input changes only prefix and Cr position."""

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_neb_path_topology import displacement, norm, read_qe_image  # noqa: E402
from audit_qe_calculator_identity import identity as calculator_identity  # noqa: E402


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_line_role(line):
    if re.match(r"^\s*prefix\s*=", line, flags=re.IGNORECASE):
        return "prefix"
    fields = line.split()
    if len(fields) >= 4 and fields[0] == "Cr":
        try:
            [float(value) for value in fields[1:4]]
        except ValueError:
            return "unexpected"
        return "cr_position"
    return "unexpected"


def audit_pair(direct_path, transformed_path, manifest):
    direct_lines = direct_path.read_text().splitlines()
    transformed_lines = transformed_path.read_text().splitlines()
    changed = []
    if len(direct_lines) == len(transformed_lines):
        for index, (direct, transformed) in enumerate(
            zip(direct_lines, transformed_lines), start=1
        ):
            if direct != transformed:
                changed.append({
                    "line": index,
                    "role": changed_line_role(direct),
                    "direct": direct,
                    "transformed": transformed,
                })
    direct = read_qe_image(direct_path)
    transformed = read_qe_image(transformed_path)
    cr_indices = [
        index for index, symbol in enumerate(direct["symbols"]) if symbol == "Cr"
    ]
    cr_index = cr_indices[0] if len(cr_indices) == 1 else None
    host_max = None
    cr_shift = None
    if (
        cr_index is not None
        and direct["symbols"] == transformed["symbols"]
        and direct["cell"] == transformed["cell"]
    ):
        shifts = [
            norm(displacement(before, after, direct["cell"]))
            for before, after in zip(direct["positions"], transformed["positions"])
        ]
        cr_shift = shifts[cr_index]
        host_max = max(
            value for index, value in enumerate(shifts) if index != cr_index
        )
    direct_identity = calculator_identity(direct_path, "direct")
    transformed_identity = calculator_identity(transformed_path, "transformed")
    expected_shift = float(manifest["transform"]["metrics"]["cr_displacement_A"])
    checks = {
        "line_count_equal": len(direct_lines) == len(transformed_lines),
        "changed_line_roles_exact": sorted(row["role"] for row in changed)
        == ["cr_position", "prefix"],
        "symbols_equal": direct["symbols"] == transformed["symbols"],
        "cell_equal": direct["cell"] == transformed["cell"],
        "single_cr": len(cr_indices) == 1,
        "host_positions_equal": host_max is not None and host_max <= 1.0e-10,
        "cr_shift_matches_manifest": cr_shift is not None
        and math.isclose(cr_shift, expected_shift, abs_tol=1.0e-9),
        "direct_hash_matches_manifest": sha256_file(direct_path)
        == manifest["source_pw_sha256"],
        "transformed_hash_matches_manifest": sha256_file(transformed_path)
        == manifest["relax_input_sha256"],
        "calculator_identity_equal": direct_identity["calculator_identity"]
        == transformed_identity["calculator_identity"],
        "calculator_identity_complete": direct_identity["identity_complete"]
        and transformed_identity["identity_complete"],
    }
    return {
        "path_id": manifest["source_path_id"],
        "image_index_qe": int(manifest["source_image_index_qe"]),
        "direct_job_id": str(manifest["direct_baseline_job_id"]),
        "direct_input": str(direct_path.resolve()),
        "direct_input_sha256": sha256_file(direct_path),
        "transformed_input": str(transformed_path.resolve()),
        "transformed_input_sha256": sha256_file(transformed_path),
        "changed_lines": changed,
        "host_max_displacement_A": host_max,
        "cr_displacement_A": cr_shift,
        "calculator_identity": direct_identity["calculator_identity"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.transform_manifest.resolve()
    payload = json.loads(manifest_path.read_text())
    rows = [
        audit_pair(
            Path(row["source_pw_input"]),
            Path(row["relax_input"]),
            row,
        )
        for row in payload["jobs"]
    ]
    result = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "site_transform_ab_semantic_input_pair_audit",
        "transform_manifest": str(manifest_path),
        "transform_manifest_sha256": sha256_file(manifest_path),
        "pair_count": len(rows),
        "passed_count": sum(row["passed"] for row in rows),
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit("Semantic input-pair audit failed")
    print(json.dumps({
        "pairs": result["pair_count"],
        "passed": result["passed_count"],
        "output": str(args.output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
