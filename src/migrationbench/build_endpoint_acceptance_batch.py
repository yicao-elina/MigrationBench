#!/usr/bin/env python3
"""Build individual endpoint acceptance records from first/repeat QE relaxations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from build_endpoint_pair_acceptance import evaluate_record, status_index
from prepare_qe_repeat_relaxations import sha256_file


def build(args) -> tuple[Path, dict]:
    first_path = args.first_status.resolve()
    repeat_path = args.repeat_status.resolve()
    policy_path = args.policy.resolve()
    first = status_index(first_path)
    repeat = status_index(repeat_path)
    if not repeat:
        raise ValueError("Repeat status contains no jobs")
    unknown = sorted(set(repeat) - set(first))
    if unknown:
        raise ValueError(f"Repeat jobs have no parent first-relax row: {unknown}")
    policy = json.loads(policy_path.read_text())
    selected = set(args.select or [])
    keys = sorted(repeat)
    if selected:
        parsed = set()
        for value in selected:
            path_id, image = value.rsplit(":", 1)
            parsed.add((path_id, int(image)))
        missing = sorted(parsed - set(repeat))
        if missing:
            raise ValueError(f"Selected endpoints missing from repeat status: {missing}")
        keys = sorted(parsed)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [evaluate_record(first[key], repeat[key], out_dir, policy) for key in keys]
    accepted = sum(record["status"] == "accepted" for record in records)
    status = "accepted_all" if accepted == len(records) else "rejected_all" if not accepted else "partially_accepted"
    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_role": "individual_repeat_validated_endpoint_acceptance_batch",
        "status": status,
        "counts": {"evaluated": len(records), "accepted": accepted, "rejected": len(records) - accepted},
        "endpoint_records": records,
        "first_status": str(first_path),
        "first_status_sha256": sha256_file(first_path),
        "repeat_status": str(repeat_path),
        "repeat_status_sha256": sha256_file(repeat_path),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
    }
    output = out_dir / "endpoint_acceptance_batch.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-status", type=Path, required=True)
    parser.add_argument("--repeat-status", type=Path, required=True)
    parser.add_argument("--select", action="append", help="PATH_ID:IMAGE_INDEX; default every repeated endpoint")
    parser.add_argument("--policy", type=Path, default=Path("configs/representative_path_selection.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output, payload = build(args)
    print(json.dumps({"output": str(output), **payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
