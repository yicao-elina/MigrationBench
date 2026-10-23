#!/usr/bin/env python3
"""Write a file-level SHA-256 provenance manifest for this additive package."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tables"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role(path: Path) -> str:
    text = str(path.relative_to(ROOT))
    if text.startswith("raw/"):
        return "raw_evidence"
    if text.startswith("processed/"):
        return "processed_analysis"
    if text.startswith("figures/"):
        return "publication_figure"
    if text.startswith("scripts/"):
        return "reproduction_script"
    if text.startswith("tables/"):
        return "audit_or_table"
    return "documentation_or_registry"


def main() -> None:
    rows = []
    excluded = {"FILE_PROVENANCE_MANIFEST.tsv", "PACKAGE_PROVENANCE_MANIFEST.json"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in excluded or "__pycache__" in path.parts:
            continue
        rows.append(
            {
                "relative_path": str(path.relative_to(ROOT)),
                "role": role(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "FILE_PROVENANCE_MANIFEST.tsv", sep="\t", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(ROOT),
        "n_files": int(len(frame)),
        "n_raw_evidence_files": int(frame["role"].eq("raw_evidence").sum()),
        "n_processed_files": int(frame["role"].eq("processed_analysis").sum()),
        "n_figure_files": int(frame["role"].eq("publication_figure").sum()),
        "n_reproduction_scripts": int(frame["role"].eq("reproduction_script").sum()),
        "manifest_tsv": "tables/FILE_PROVENANCE_MANIFEST.tsv",
    }
    (OUT / "PACKAGE_PROVENANCE_MANIFEST.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
