#!/usr/bin/env python3
"""Validate the Canva figure path catalog without requiring external data."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "CANVA_FIGURE_REGISTRY.tsv"
DOWNLOADS = ROOT / "DOWNLOAD_EVIDENCE.tsv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    registry = read_tsv(REGISTRY)
    downloads = read_tsv(DOWNLOADS)
    expected_ids = {f"CANVA-P{i:02d}" for i in range(1, 11)}
    actual_ids = {row["figure_id"] for row in registry}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise SystemExit(f"figure id mismatch: missing={missing}, extra={extra}")

    required = {
        "figure_id",
        "canva_page",
        "source_figure_path",
        "raw_data_path",
        "processed_data_path",
        "plot_script_path",
        "manifest_or_source_map",
    }
    for row in registry:
        missing = sorted(key for key in required if not row.get(key, "").strip())
        if missing:
            raise SystemExit(f"{row.get('figure_id')}: missing fields {missing}")

    unavailable = 0
    checked = 0
    for row in downloads:
        for field in ("download_file", "audited_source_file"):
            path = Path(row[field])
            expected = row["download_sha256"] if field == "download_file" else row["source_sha256"]
            if not path.exists():
                unavailable += 1
                continue
            checked += 1
            actual = sha256(path)
            if actual != expected:
                raise SystemExit(f"hash mismatch: {path}: expected {expected}, got {actual}")

    print(f"CATALOG_VALID figures={len(registry)} hash_files_checked={checked} external_files_unavailable={unavailable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
