#!/usr/bin/env python3
"""Validate the compact processed endpoint release without raw-data access."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "collaborator_dft_neb"
XYZ = DATA / "processed_endpoints.extxyz"
MANIFEST = DATA / "processed_endpoints_manifest.json"


def main() -> None:
    lines = XYZ.read_text(encoding="utf-8").splitlines()
    i = 0
    records = []
    while i < len(lines):
        natoms = int(lines[i]); comment = lines[i + 1]
        coords = lines[i + 2:i + 2 + natoms]
        if len(coords) != natoms or not re.search(r' Lattice="[^"]+"', comment):
            raise SystemExit(f"invalid block near line {i + 1}")
        required = ["endpoint_id=", "system=", "formula=", "energy_ry=", "max_force_ev_ang=",
                    "input_sha256=", "output_sha256=", 'pbc="T T T"']
        missing = [key for key in required if key not in comment]
        if missing:
            raise SystemExit(f"missing {missing} in block {len(records)}")
        for row in coords:
            fields = row.split()
            if len(fields) != 4 or any(not float(x) == float(x) for x in fields[1:]):
                raise SystemExit(f"invalid coordinate row in block {len(records)}")
        endpoint = re.search(r"endpoint_id=([^ ]+)", comment).group(1)
        records.append({"endpoint_id": endpoint, "natoms": natoms, "comment": comment})
        i += natoms + 2
    digest = hashlib.sha256(XYZ.read_bytes()).hexdigest()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "artifact": str(XYZ), "sha256": digest, "n_structures": len(records),
                "records": [{"endpoint_id": r["endpoint_id"], "natoms": r["natoms"]} for r in records],
                "raw_policy": "raw QE inputs/outputs remain on Rockfish; this is processed-only"}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "n_structures": len(records), "sha256": digest}))


if __name__ == "__main__":
    main()
