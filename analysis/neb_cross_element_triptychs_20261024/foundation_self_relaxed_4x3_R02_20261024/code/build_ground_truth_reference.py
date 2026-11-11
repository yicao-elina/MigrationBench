#!/usr/bin/env python3
"""Build the auditable QE/DFT reference layer for the R03 figure.

The reference layer is deliberately separate from the self-relaxed MLFF
profiles.  It collects the existing canonical fixed-path QE/DFT curves,
preserves their raw status, and records that they are SCF-warning diagnostics
rather than same-geometry DFT rescoring of the relaxed MLFF paths.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ELEMENTS = ["Cr", "Cu", "Li", "Na"]
PATHS = ["in_gap", "gap_to_ql", "within_ql"]
PATH_ALIASES = {"within_gap": "in_gap", "in_gap": "in_gap"}
REFERENCE_LABEL = "QE/DFT reference · SCF warning; diagnostic"
CAMPAIGN_ID = "foundation_self_relaxed_4x3_20261024_r02"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def raw_status_display(raw_status: str) -> str:
    if raw_status in {"COMPLETED_BUT_SCF_WARNING", "failed_scf_convergence"}:
        return "SCF_WARNING_DIAGNOSTIC"
    return "UNREVIEWED_REFERENCE_STATUS"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    campaign_root = root.parents[0]
    source_dir = campaign_root / "data"
    cr_source = source_dir / "Cr_fixed_path_representative_profiles.tsv"
    cross_source = source_dir / "foundation_model_qe_reference_profiles_3x3.tsv"
    path_registry_path = root / "PATH_REGISTRY.tsv"
    self_profiles_path = root / "data" / "CLASS_PROFILES.tsv"
    out_tsv = root / "data" / "GROUND_TRUTH_REFERENCE_PROFILES_R03.tsv"
    out_manifest = root / "data" / "GROUND_TRUTH_REFERENCE_MANIFEST_R03.json"

    sources = {
        "Cr": (cr_source, [row for row in read_tsv(cr_source) if row.get("kind") == "reference"]),
        "cross_element": (cross_source, read_tsv(cross_source)),
    }
    registry = read_tsv(path_registry_path)
    canonical_ids = {(row["element"], row["path_class"]): row["path_id"] for row in registry}
    if set(canonical_ids) != {(element, path_class) for element in ELEMENTS for path_class in PATHS}:
        raise SystemExit("PATH_REGISTRY does not contain the complete 4x3 canonical path matrix")

    output_rows: list[dict[str, str]] = []
    for source_name, (source_table, source_rows) in sources.items():
        source_hash = sha256(source_table)
        for row in source_rows:
            element = row["element"]
            source_path_class = row.get("path_key", "")
            path_class = PATH_ALIASES.get(source_path_class, source_path_class)
            if element not in ELEMENTS or path_class not in PATHS:
                raise SystemExit(f"unexpected reference identity: {element=} {source_path_class=}")
            source_image_index = int(row["image_index"])
            # The Cr legacy table is zero-based (0..9), whereas the
            # cross-element QE table is one-based (1..10).  Preserve the raw
            # index and normalize only the plotting/index contract.
            image_index = source_image_index if source_name == "Cr" else source_image_index - 1
            raw_source_path = Path(row["source_path"])
            output_rows.append(
                {
                    "element": element,
                    "path_class": path_class,
                    "canonical_path_id": canonical_ids[(element, path_class)],
                    "source_path_id": row.get("path_id", ""),
                    "image_index": str(image_index),
                    "source_image_index": str(source_image_index),
                    "source_s": row["s"],
                    "plot_s": f"{image_index / 9.0:.15g}",
                    "relative_energy_eV": row.get("relative_energy_eV", ""),
                    "raw_status": row.get("status", ""),
                    "display_status": raw_status_display(row.get("status", "")),
                    "reference_label": REFERENCE_LABEL,
                    "source_table": str(source_table),
                    "source_table_sha256": source_hash,
                    "raw_source_path": str(raw_source_path),
                    "raw_source_sha256": sha256(raw_source_path) if raw_source_path.is_file() else "",
                    "source_record": row.get("source_record", ""),
                }
            )

    output_rows.sort(key=lambda row: (ELEMENTS.index(row["element"]), PATHS.index(row["path_class"]), int(row["image_index"])))
    fields = list(output_rows[0])
    with out_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in output_rows:
        grouped[(row["element"], row["path_class"])].append(row)
    group_records = []
    for key in [(element, path_class) for element in ELEMENTS for path_class in PATHS]:
        group = grouped[key]
        group_records.append(
            {
                "element": key[0],
                "path_class": key[1],
                "canonical_path_id": group[0]["canonical_path_id"],
                "rows": len(group),
                "image_indices": [int(row["image_index"]) for row in group],
                "source_image_indices": [int(row["source_image_index"]) for row in group],
                "raw_statuses": dict(sorted(Counter(row["raw_status"] for row in group).items())),
                "source_records": sorted({row["source_record"] for row in group}),
                "source_table": group[0]["source_table"],
                "source_table_sha256": group[0]["source_table_sha256"],
                "source_s": [float(row["source_s"]) for row in group],
                "plot_s": [float(row["plot_s"]) for row in group],
            }
        )

    checks = {
        "complete_4x3_reference_matrix": len(output_rows) == 120 and len(group_records) == 12,
        "ten_images_per_group": all(record["rows"] == 10 and record["image_indices"] == list(range(10)) for record in group_records),
        "finite_values": all(
            finite(row["source_s"]) and finite(row["plot_s"]) and finite(row["relative_energy_eV"])
            for row in output_rows
        ),
        "normalized_plot_coordinate": all(
            abs(float(row["plot_s"]) - int(row["image_index"]) / 9.0) < 1e-12 for row in output_rows
        ),
        "zeroed_initial_energy": all(abs(float(group[0]["relative_energy_eV"])) < 1e-10 for group in grouped.values()),
        "source_tables_present": all(path.is_file() for path, _ in sources.values()),
        "raw_source_tables_present": all(bool(row["raw_source_sha256"]) for row in output_rows),
        "warning_statuses_retained": all(row["display_status"] == "SCF_WARNING_DIAGNOSTIC" for row in output_rows),
    }
    source_files = []
    for source_name, (source_table, source_rows) in sources.items():
        source_files.append(
            {
                "role": source_name,
                "path": str(source_table),
                "sha256": sha256(source_table),
                "bytes": source_table.stat().st_size,
                "rows_read": len(source_rows),
            }
        )
    raw_files = {}
    for row in output_rows:
        raw_files[row["raw_source_path"]] = row["raw_source_sha256"]
    raw_source_files = [{"path": path, "sha256": digest} for path, digest in sorted(raw_files.items())]

    manifest = {
        "schema_version": "foundation_self_relaxed_reference_manifest_v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_label": REFERENCE_LABEL,
        "scientific_status": "PROVISIONAL_DIAGNOSTIC_REFERENCE",
        "unqualified_ground_truth_claim": False,
        "observable": "QE_DFT_FIXED_PATH_SINGLE_POINT_REFERENCE",
        "profiles": {
            "path": str(out_tsv),
            "sha256": sha256(out_tsv),
            "rows": len(output_rows),
            "groups": len(group_records),
        },
        "source_files": source_files,
        "raw_source_files": raw_source_files,
        "groups": group_records,
        "alignment": {
            "reference_geometry": "canonical fixed-path NEB image tables",
            "self_relaxed_geometry": "Foundation MLFF endpoint-only self-relaxed profiles",
            "same_geometry_dft_rescoring": False,
            "plot_coordinate": "normalized_image_index / 9; raw source_image_index and source_s retained separately",
            "source_coordinate_policy": "Cr source_s is preserved but not used for overlay because it is nonuniform; all reference traces are aligned by image index.",
            "publication_boundary": "Display as a QE/DFT reference diagnostic; do not call it a validated DFT MEP, converged DFT barrier, or same-geometry ground truth for the self-relaxed curves.",
        },
        "checks": checks,
        "release_ready": all(checks.values()),
        "self_profile_input": {
            "path": str(self_profiles_path),
            "sha256": sha256(self_profiles_path),
        },
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"release_ready": manifest["release_ready"], "profiles": len(output_rows), "manifest": str(out_manifest)}, indent=2))
    if not manifest["release_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
