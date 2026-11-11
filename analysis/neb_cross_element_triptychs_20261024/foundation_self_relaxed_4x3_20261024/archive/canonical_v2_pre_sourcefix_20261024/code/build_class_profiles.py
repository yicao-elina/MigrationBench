#!/usr/bin/env python3
"""Aggregate accepted self-relaxed model profiles into class traces."""

from __future__ import annotations

import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    rows = read_tsv(data / "ROW_RESULTS.tsv")
    profiles = read_tsv(data / "IMAGE_PROFILES.tsv")
    accepted_ids = {
        row["record_id"]
        for row in rows
        if row.get("scientific_status") == "CONVERGED"
        and row.get("publication_status") == "PRIMARY_EVIDENCE"
    }
    accepted = [row for row in profiles if row["record_id"] in accepted_ids]
    class_rows: list[dict] = []
    summaries: list[dict] = []
    keys = sorted({(row["element"], row["path_id"], row["model_family"]) for row in profiles})
    for element, path_id, model_family in keys:
        subset = [
            row
            for row in accepted
            if row["element"] == element
            and row["path_id"] == path_id
            and row["model_family"] == model_family
        ]
        model_ids = sorted({row["model_id"] for row in subset})
        expected = EXPECTED[model_family]
        for image_index in range(10):
            points = [row for row in subset if int(row["image_index"]) == image_index]
            values = [float(row["rel_energy_eV"]) for row in points]
            if values and all(math.isfinite(value) for value in values):
                mean = statistics.fmean(values)
                sd = statistics.pstdev(values) if len(values) > 1 else 0.0
            else:
                mean = float("nan")
                sd = float("nan")
            class_rows.append(
                {
                    "element": element,
                    "path_id": path_id,
                    "path_class": points[0]["path_class"] if points else "",
                    "path_label": points[0]["path_label"] if points else "",
                    "model_family": model_family,
                    "image_index": image_index,
                    "s": image_index / 9.0,
                    "mean_rel_energy_eV": mean,
                    "sd_rel_energy_eV": sd,
                    "n_accepted": len(model_ids),
                    "n_expected": expected,
                    "complete_class": "PASS" if len(model_ids) == expected else "INCOMPLETE",
                }
            )
        summaries.append(
            {
                "element": element,
                "path_id": path_id,
                "path_class": subset[0]["path_class"] if subset else "",
                "path_label": subset[0]["path_label"] if subset else "",
                "model_family": model_family,
                "n_accepted": len(model_ids),
                "n_expected": expected,
                "accepted_model_ids": ",".join(model_ids),
                "status": "PASS" if len(model_ids) == expected else "INCOMPLETE",
            }
        )
    exclusions = []
    for row in rows:
        exclusions.append(
            {
                "record_id": row["record_id"],
                "element": row["element"],
                "path_id": row["path_id"],
                "model_id": row["model_id"],
                "model_family": row["model_family"],
                "scientific_status": row.get("scientific_status", ""),
                "publication_status": row.get("publication_status", ""),
                "included_in_class_profile": "YES" if row["record_id"] in accepted_ids else "NO",
                "failure_class": row.get("failure_class", ""),
                "failure_reason": row.get("failure_reason", ""),
            }
        )
    write_tsv(data / "CLASS_PROFILES.tsv", class_rows)
    write_tsv(data / "CLASS_SUMMARY.tsv", summaries)
    write_tsv(data / "PROFILE_INCLUSION_AUDIT.tsv", exclusions)
    release_ready = (
        len(rows) == 108
        and len(accepted_ids) == 108
        and bool(summaries)
        and all(item["status"] == "PASS" for item in summaries)
        and all(math.isfinite(float(item["mean_rel_energy_eV"])) for item in class_rows)
    )
    manifest = {
        "schema_version": "foundation_self_relaxed_class_profiles_v1",
        "campaign_id": "foundation_self_relaxed_4x3_20261024",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_rows": len(rows),
        "accepted_rows": len(accepted_ids),
        "class_summary_rows": len(summaries),
        "profile_rows": len(class_rows),
        "release_ready": release_ready,
        "release_gate": "all 108 rows converged and all class denominators complete",
        "outputs": {
            name: sha256(data / name)
            for name in ("CLASS_PROFILES.tsv", "CLASS_SUMMARY.tsv", "PROFILE_INCLUSION_AUDIT.tsv")
        },
    }
    (data / "DERIVED_PROFILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

