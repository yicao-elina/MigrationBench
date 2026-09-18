#!/usr/bin/env python3
"""Validate nonlinear candidate paths with ASE without creating labels."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(manifest_path):
    from ase.io import read

    manifest = json.loads(manifest_path.read_text())
    rows = []
    for candidate in manifest["candidates"]:
        path = Path(candidate["images"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        elif not path.exists():
            # Support legacy manifests after moving a self-contained directory.
            path = manifest_path.parent / path.name
        hash_matches = path.exists() and sha256_file(path) == candidate["images_sha256"]
        images = read(path, index=":") if path.exists() else []
        image_count_matches = len(images) == manifest["n_images"]
        atom_counts = [len(image) for image in images]
        atom_count_consistent = bool(atom_counts) and len(set(atom_counts)) == 1
        pbc_complete = bool(images) and all(bool(value) for image in images for value in image.pbc)
        forbidden_arrays = sorted({
            key for image in images for key in image.arrays
            if key.lower() in {"force", "forces", "energy", "energies"}
        })
        calculator_attached = any(image.calc is not None for image in images)
        endpoint_errors = [
            float(candidate["initial_endpoint_pbc_deviation_A"]),
            float(candidate["final_endpoint_pbc_deviation_A"]),
        ]
        accepted = all((
            hash_matches,
            image_count_matches,
            atom_count_consistent,
            pbc_complete,
            not forbidden_arrays,
            not calculator_attached,
            max(endpoint_errors) <= 1.0e-8,
            candidate["contains_energy_labels"] is False,
            candidate["contains_force_labels"] is False,
        ))
        rows.append({
            "branch_id": candidate["branch_id"],
            "images": str(path.resolve()),
            "sha256_matches_manifest": hash_matches,
            "n_images": len(images),
            "image_count_matches": image_count_matches,
            "atom_count": atom_counts[0] if atom_count_consistent else None,
            "atom_count_consistent": atom_count_consistent,
            "pbc_complete": pbc_complete,
            "forbidden_label_arrays": forbidden_arrays,
            "calculator_attached": calculator_attached,
            "max_endpoint_pbc_deviation_A": max(endpoint_errors),
            "accepted_geometry_only_candidate": accepted,
        })
    return {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "candidates": rows,
        "gate": "pass" if rows and all(row["accepted_geometry_only_candidate"] for row in rows) else "fail",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "gate": result["gate"],
        "candidates": len(result["candidates"]),
        "output": str(args.output.resolve()),
    }, indent=2))
    if result["gate"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
