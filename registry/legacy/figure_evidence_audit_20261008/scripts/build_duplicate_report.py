#!/usr/bin/env python3
"""Report identical rendered-export hashes across checkouts and formats."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/figure_evidence_audit_20261008")


def main() -> None:
    records = json.loads((ROOT / "figure_inventory.json").read_text(encoding="utf-8"))
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        for output in record["outputs"]:
            by_hash[output["sha256"]].append({
                "root_id": record["root_id"],
                "logical_figure": record["logical_figure"],
                "path": output["path"],
                "extension": output["extension"],
            })
    duplicates = {digest: items for digest, items in by_hash.items() if len(items) > 1}
    rows = []
    for digest, items in sorted(duplicates.items()):
        for item in items:
            rows.append({"sha256": digest, **item, "duplicate_group_size": len(items)})
    fields = ["sha256", "duplicate_group_size", "root_id", "logical_figure", "extension", "path"]
    with (ROOT / "duplicate_output_hashes.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "duplicate_hash_group_count": len(duplicates),
        "duplicate_output_file_count": len(rows),
        "unique_output_hash_count": len(by_hash),
        "rendered_output_file_count": sum(len(record["outputs"]) for record in records),
    }
    (ROOT / "duplicate_output_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
