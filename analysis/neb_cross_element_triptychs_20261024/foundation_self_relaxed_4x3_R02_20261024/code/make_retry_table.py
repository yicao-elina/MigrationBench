#!/usr/bin/env python3
"""Create an explicit retry table from the finalized row-level failure ledger."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default="R02A")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--fire-dt", default="0.1")
    parser.add_argument("--fire-maxstep", default="0.2")
    parser.add_argument("--spring-constant", default="0.1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    collection = data / "COLLECTION_MANIFEST.json"
    if not collection.is_file():
        raise SystemExit("run final collection before creating a retry table")
    import json

    summary = json.loads(collection.read_text())
    if not summary.get("finalized") or summary.get("target_rows") != 108:
        raise SystemExit("retry table requires finalized collection of all 108 rows")
    with (root / "JOB_TABLE_IDPP.tsv").open() as handle:
        job_rows = list(csv.DictReader(handle, delimiter="\t"))
    with (data / "ROW_RESULTS.tsv").open() as handle:
        result_rows = list(csv.DictReader(handle, delimiter="\t"))
    failed = {
        row["record_id"]
        for row in result_rows
        if row.get("scientific_status") != "CONVERGED"
    }
    selected = [dict(row) for row in job_rows if row["record_id"] in failed]
    for row in selected:
        row["steps_max"] = str(args.steps)
        row["execution_role"] = "FORMAL_ARRAY"
        row["manifest_role"] = f"RETRY_ARRAY_{args.round}"
        row["retry_round"] = args.round
        row["retry_reason"] = "target-model convergence/residual gate failed in prior formal attempt"
        row["fire_dt"] = str(args.fire_dt)
        row["fire_maxstep"] = str(args.fire_maxstep)
        row["spring_constant"] = str(args.spring_constant)
    output = root / f"JOB_TABLE_IDPP_RETRY_{args.round}.tsv"
    if selected:
        fields = list(selected[0])
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(selected)
    else:
        output.write_text("record_id\n")
    print(json.dumps({"round": args.round, "failed_input_rows": len(failed), "retry_rows": len(selected), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
