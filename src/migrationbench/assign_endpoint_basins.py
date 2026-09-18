#!/usr/bin/env python3
"""Deduplicate accepted endpoint structures into versioned basin assignments."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from ase.io import read

from analyze_neb_path_topology import displacement, norm
from prepare_qe_repeat_relaxations import sha256_file


def composition(symbols):
    counts = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return "".join("{}{}".format(symbol, counts[symbol]) for symbol in sorted(counts))


def structure_record(endpoint, migrant_element):
    path = Path(endpoint["accepted_structure"]).resolve()
    if sha256_file(path) != endpoint["accepted_structure_sha256"]:
        raise ValueError("Accepted structure hash mismatch: {}".format(path))
    atoms = read(path, index=-1)
    symbols = atoms.get_chemical_symbols()
    migrant = [index for index, symbol in enumerate(symbols) if symbol == migrant_element]
    if len(migrant) != 1:
        raise ValueError("Expected exactly one {} in {}".format(migrant_element, path))
    cell = [tuple(float(value) for value in row) for row in atoms.get_cell()]
    cell_key = hashlib.sha256(json.dumps(
        [[round(value, 5) for value in row] for row in cell], separators=(",", ":")
    ).encode()).hexdigest()[:12]
    return {
        "path_id": endpoint["path_id"],
        "image_index_qe": int(endpoint["image_index_qe"]),
        "parent_job_id": str(endpoint["parent_job_id"]),
        "repeat_job_id": str(endpoint["repeat_job_id"]),
        "accepted_structure": str(path),
        "accepted_structure_sha256": endpoint["accepted_structure_sha256"],
        "calculator_identity": endpoint["calculator_identity"],
        "symbols": symbols,
        "positions": [tuple(float(value) for value in row) for row in atoms.get_positions()],
        "cell": cell,
        "migrant_index": migrant[0],
        "system_key": "{}_{}".format(composition(symbols), cell_key),
    }


def local_signature(record, neighbors):
    migrant = record["positions"][record["migrant_index"]]
    distances = [
        norm(displacement(migrant, position, record["cell"]))
        for index, position in enumerate(record["positions"])
        if index != record["migrant_index"]
    ]
    return sorted(distances)[:neighbors]


def basin_distance(left, right, neighbors):
    if left["system_key"] != right["system_key"] or left["symbols"] != right["symbols"]:
        return None
    vectors = [
        displacement(a, b, left["cell"])
        for a, b in zip(left["positions"], right["positions"])
    ]
    migrant_distance = norm(vectors[left["migrant_index"]])
    host = [norm(vector) for index, vector in enumerate(vectors) if index != left["migrant_index"]]
    host_rmsd = math.sqrt(sum(value * value for value in host) / len(host))
    left_signature = local_signature(left, neighbors)
    right_signature = local_signature(right, neighbors)
    local_rms = math.sqrt(sum(
        (a - b) ** 2 for a, b in zip(left_signature, right_signature)
    ) / len(left_signature))
    return {
        "migrant_distance_A": migrant_distance,
        "host_rmsd_A": host_rmsd,
        "local_signature_rms_A": local_rms,
    }


def assign(records, thresholds):
    comparisons = []
    equivalent_pairs = {}
    neighbors = int(thresholds["local_signature_neighbors"])
    for left in range(len(records)):
        for right in range(left):
            metrics = basin_distance(records[left], records[right], neighbors)
            equivalent = bool(metrics) and (
                metrics["migrant_distance_A"] <= thresholds["maximum_migrant_distance_A"]
                and metrics["host_rmsd_A"] <= thresholds["maximum_host_rmsd_A"]
                and metrics["local_signature_rms_A"] <= thresholds["maximum_local_signature_rms_A"]
            )
            comparisons.append({
                "left_structure_sha256": records[left]["accepted_structure_sha256"],
                "right_structure_sha256": records[right]["accepted_structure_sha256"],
                "comparable_system": metrics is not None,
                "equivalent": equivalent,
                **(metrics or {}),
            })
            equivalent_pairs[(right, left)] = equivalent
    clusters = [[index] for index in range(len(records))]
    changed = True
    while changed:
        changed = False
        for left in range(len(clusters)):
            merged = False
            for right in range(left):
                if all(
                    equivalent_pairs.get(tuple(sorted((a, b))), a == b)
                    for a in clusters[left] for b in clusters[right]
                ):
                    clusters[right].extend(clusters[left])
                    del clusters[left]
                    changed = True
                    merged = True
                    break
            if merged:
                break
    assignments = []
    for indices in clusters:
        members = [records[index] for index in indices]
        representative = min(members, key=lambda row: row["accepted_structure_sha256"])
        basin_token = hashlib.sha256(
            (representative["system_key"] + ":" + representative["accepted_structure_sha256"]).encode()
        ).hexdigest()[:16]
        basin_id = "BASIN_{}_{}".format(
            representative["system_key"], basin_token
        )
        for record in members:
            assignments.append({
                "basin_id": basin_id,
                "is_representative": record is representative,
                "system_key": record["system_key"],
                "path_id": record["path_id"],
                "image_index_qe": record["image_index_qe"],
                "parent_job_id": record["parent_job_id"],
                "repeat_job_id": record["repeat_job_id"],
                "accepted_structure": record["accepted_structure"],
                "accepted_structure_sha256": record["accepted_structure_sha256"],
                "calculator_identity": record["calculator_identity"],
            })
    return sorted(assignments, key=lambda row: (row["basin_id"], row["accepted_structure_sha256"])), comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, action="append", required=True)
    parser.add_argument("--policy", type=Path, default=Path("configs/representative_path_selection.json"))
    parser.add_argument("--migrant-element", default="Cr")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    policy_path = args.policy.resolve()
    policy = json.loads(policy_path.read_text())
    records = []
    sources = []
    for path in args.acceptance:
        resolved = path.resolve()
        payload = json.loads(resolved.read_text())
        sources.append({"path": str(resolved), "sha256": sha256_file(resolved), "status": payload.get("status")})
        for endpoint in payload.get("endpoint_records", []):
            if endpoint.get("status") == "accepted":
                records.append(structure_record(endpoint, args.migrant_element))
    if not records:
        raise ValueError("No accepted endpoint records")
    assignments, comparisons = assign(records, policy["basin_equivalence"])
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "migrant_element": args.migrant_element,
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "thresholds": policy["basin_equivalence"],
        "sources": sources,
        "counts": {"accepted_records": len(records), "basins": len({row["basin_id"] for row in assignments})},
        "assignments": assignments,
        "pairwise_comparisons": comparisons,
    }
    (out_dir / "endpoint_basin_assignments.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (out_dir / "endpoint_basin_assignments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)
    print(json.dumps(payload["counts"], indent=2))


if __name__ == "__main__":
    main()
