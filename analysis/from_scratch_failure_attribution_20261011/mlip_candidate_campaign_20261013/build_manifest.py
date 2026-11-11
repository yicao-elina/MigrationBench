from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 639:
        raise SystemExit(f"expected 639 candidates, found {len(rows)}")
    for idx, row in enumerate(rows, start=1):
        row.setdefault("candidate_id", f"cand_{idx:04d}")
        row["candidate_index"] = str(idx - 1)
        row["candidate_file_sha256"] = sha256(Path(row["prepared_extxyz"])) if Path(row["prepared_extxyz"]).is_file() else "remote_only"
        row["evaluation_scope"] = "same candidate structure evaluated by all three registered MLIP checkpoints"
    fieldnames = ["candidate_id", "candidate_index"] + [k for k in rows[0] if k not in {"candidate_id", "candidate_index"}]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "source": str(source),
        "source_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "n_candidates": len(rows),
        "candidate_id_scheme": "source candidate_id values preserved; 639 rows",
        "status": "VERIFIED" if all(r["candidate_file_sha256"] != "remote_only" for r in rows) else "PROVENANCE_PARTIAL",
    }
    output.with_name("manifest_build.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
