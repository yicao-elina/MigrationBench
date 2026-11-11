#!/usr/bin/env python3
"""Materialize the unified sliding profile table required by the registry MD."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "analysis/foundation_matpes_pbe_omat_ft_campaign_20261024/sliding_extension"
PROFILES = CAMPAIGN / "results/tables/IMAGE_PROFILES.tsv"
RESULTS = CAMPAIGN / "results/tables/RESULT_MATRIX.tsv"
OUT = ROOT / "analysis/foundation_registry_20261024/SLIDING_PROFILES.tsv"
MANIFEST = ROOT / "analysis/foundation_registry_20261024/SLIDING_PROFILES_MANIFEST_R01.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def main() -> None:
    profiles = read(PROFILES)
    results = {row["record_id"]: row for row in read(RESULTS)}
    fields = [
        "system",
        "observable",
        "model_id",
        "model_class",
        "image_index",
        "s",
        "energy_eV",
        "relative_energy_eV",
        "barrier_eV",
        "status",
        "publication_status",
        "major_group",
        "model_family",
        "lineage",
        "is_fine_tuned",
        "source_record",
    ]
    out_rows = []
    for row in profiles:
        result = results[row["record_id"]]
        out_rows.append(
            {
                "system": row["system"],
                "observable": row["observable"],
                "model_id": row["model_id"],
                "model_class": row["model_family"],
                "image_index": row["image_index"],
                "s": row["image_fraction"],
                "energy_eV": row["energy_eV"],
                "relative_energy_eV": row["relative_energy_eV"],
                "barrier_eV": result["barrier_eV"],
                "status": result["scientific_status"],
                "publication_status": result["publication_status"],
                "major_group": row["major_group"],
                "model_family": row["model_family"],
                "lineage": row["lineage"],
                "is_fine_tuned": row["is_fine_tuned"],
                "source_record": row["record_id"],
            }
        )
    out_rows.sort(key=lambda r: (r["system"], r["model_id"], int(r["image_index"])))
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(out_rows)
    manifest = {
        "schema_version": "foundation_sliding_profiles_registry_r01_v1",
        "source_result_matrix": str(RESULTS),
        "source_result_matrix_sha256": sha256(RESULTS),
        "source_image_profiles": str(PROFILES),
        "source_image_profiles_sha256": sha256(PROFILES),
        "output": str(OUT),
        "output_sha256": sha256(OUT),
        "rows": len(out_rows),
        "model_count": len({r["model_id"] for r in out_rows}),
        "fixed_self_separation": True,
        "qe_status": "COMPLETED_BUT_SCF_WARNING_RETAINED",
        "old_release_untouched": True,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
