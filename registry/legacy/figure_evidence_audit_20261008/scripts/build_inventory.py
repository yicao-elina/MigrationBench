#!/usr/bin/env python3
"""Build a logical-figure inventory with lightweight provenance metadata.

This is an audit helper, not a scientific validator. It groups image exports
by path-without-suffix, records hashes and recency, and lists nearby candidate
evidence files for later human verification.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


CURRENT_ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
ABLATION_WORKTREE_ROOT = Path("/Users/alina/.codex/worktrees/bf6e/Wiley-MigrationBench")
PAPER_ROOT = Path("/Users/alina/Project/migrationbench-paper-repro")
OUT_ROOT = CURRENT_ROOT / "analysis/figure_evidence_audit_20261008"
RECENT_CUTOFF = datetime(2026, 8, 8, tzinfo=timezone.utc).timestamp()
FIGURE_EXTENSIONS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tif", ".tiff"}
EVIDENCE_EXTENSIONS = {
    ".py", ".sh", ".slurm", ".csv", ".tsv", ".json", ".jsonl", ".md",
    ".tex", ".txt", ".xyz", ".extxyz", ".path", ".out", ".err", ".log",
}
SKIP_PARTS = {
    ".git", "__pycache__", ".venv", ".venv 2", "node_modules",
    "figure_evidence_audit_20261008",
}


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def is_skipped(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)


def status_tags(path: Path) -> list[str]:
    value = str(path).lower()
    tags: list[str] = []
    patterns = (
        ("placeholder", r"placeholder"),
        ("provisional", r"provisional"),
        ("exploratory", r"exploratory|diagnostic"),
        ("strict_replica", r"strict|replica"),
        ("publication", r"publication|paper|overleaf|final"),
        ("old_or_legacy", r"old_fig|legacy|historical"),
        ("audit", r"audit|residual|stability_gate"),
        ("comparison", r"comparison|profiles|parity|disagreement"),
    )
    for tag, pattern in patterns:
        if re.search(pattern, value):
            tags.append(tag)
    return tags or ["unclassified"]


def nearby_candidates(logical_path: Path) -> list[str]:
    """List nearby non-image files likely to participate in the evidence chain."""
    candidates: set[Path] = set()
    for parent in (logical_path.parent, logical_path.parent.parent):
        if not parent.exists():
            continue
        try:
            for path in parent.iterdir():
                if (
                    path.is_file()
                    and path.suffix.lower() in EVIDENCE_EXTENSIONS
                    and not is_skipped(path)
                ):
                    candidates.add(path)
        except OSError:
            continue
    return sorted(str(path) for path in candidates)


def collect(root: Path, root_id: str) -> list[dict]:
    groups: dict[Path, list[Path]] = defaultdict(list)
    if not root.exists():
        return []
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in FIGURE_EXTENSIONS
            and not is_skipped(path)
        ):
            groups[path.with_suffix("")].append(path)

    records: list[dict] = []
    for logical_path, outputs in sorted(groups.items(), key=lambda item: str(item[0])):
        output_records = []
        for output in sorted(outputs):
            stat = output.stat()
            output_records.append(
                {
                    "path": str(output),
                    "relative_to_root": str(output.relative_to(root)),
                    "extension": output.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                    "sha256": sha256(output),
                }
            )
        latest_mtime = max(item["mtime_utc"] for item in output_records)
        records.append(
            {
                "root_id": root_id,
                "logical_figure": str(logical_path.relative_to(root)),
                "logical_path": str(logical_path),
                "output_count": len(output_records),
                "formats": ",".join(item["extension"] for item in output_records),
                "latest_mtime_utc": latest_mtime,
                "recent_since_2026_08_08": any(
                    Path(item["path"]).stat().st_mtime >= RECENT_CUTOFF
                    for item in output_records
                ),
                "status_tags": ";".join(status_tags(logical_path)),
                "outputs": output_records,
                "nearby_candidates": nearby_candidates(logical_path),
            }
        )
    return records


def main() -> None:
    records = collect(CURRENT_ROOT, "wiley-migrationbench")
    records.extend(collect(ABLATION_WORKTREE_ROOT, "wiley-migrationbench-ablation-worktree"))
    records.extend(collect(PAPER_ROOT, "paper-repro"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    (OUT_ROOT / "figure_inventory.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fields = [
        "root_id", "logical_figure", "logical_path", "output_count", "formats",
        "latest_mtime_utc", "recent_since_2026_08_08", "status_tags",
        "output_paths", "output_sha256", "nearby_candidates",
    ]
    with (OUT_ROOT / "figure_inventory.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "root_id": record["root_id"],
                    "logical_figure": record["logical_figure"],
                    "logical_path": record["logical_path"],
                    "output_count": record["output_count"],
                    "formats": record["formats"],
                    "latest_mtime_utc": record["latest_mtime_utc"],
                    "recent_since_2026_08_08": record["recent_since_2026_08_08"],
                    "status_tags": record["status_tags"],
                    "output_paths": " | ".join(item["path"] for item in record["outputs"]),
                    "output_sha256": " | ".join(item["sha256"] for item in record["outputs"]),
                    "nearby_candidates": " | ".join(record["nearby_candidates"]),
                }
            )

    counts = defaultdict(int)
    for record in records:
        counts[record["root_id"]] += 1
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "recent_cutoff": "2026-08-08",
        "logical_figure_count": len(records),
        "counts_by_root": dict(counts),
        "recent_logical_figure_count": sum(
            bool(record["recent_since_2026_08_08"]) for record in records
        ),
        "format_export_count": sum(record["output_count"] for record in records),
    }
    (OUT_ROOT / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
