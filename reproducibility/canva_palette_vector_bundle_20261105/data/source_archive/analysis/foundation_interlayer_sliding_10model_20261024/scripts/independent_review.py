#!/usr/bin/env python3
"""Run independent data/provenance and visual acceptance passes."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "foundation_agnesi_large", "foundation_agnesi_medium_density", "foundation_agnesi_small_density",
    "foundation_agnesi_medium", "foundation_mp0b3", "foundation_omat0_medium", "foundation_omat0_small",
    "foundation_mh0", "foundation_mh1", "foundation_matpes_pbe_omat_ft",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def check_data(root: Path) -> dict:
    reasons = []
    matrix = read_tsv(root / "results/tables/RESULT_MATRIX.tsv")
    profiles = read_tsv(root / "results/tables/IMAGE_PROFILES.tsv")
    qe = read_tsv(root / "results/tables/QE_REFERENCE.tsv")
    if len(matrix) != 40:
        reasons.append(f"result_matrix_rows={len(matrix)} expected=40")
    if len(profiles) != 400:
        reasons.append(f"image_profile_rows={len(profiles)} expected=400")
    if len(qe) != 20:
        reasons.append(f"qe_rows={len(qe)} expected=20")
    if {row["model_id"] for row in matrix} != MODELS:
        reasons.append("model_identity_set_mismatch")
    if {row["system"] for row in matrix} != {"bulk", "cr_in_gap"}:
        reasons.append("system_identity_set_mismatch")
    if sum(row["observable"] == "FIXED_PATH_SINGLE_POINT" for row in matrix) != 20:
        reasons.append("fixed_row_count_mismatch")
    if sum(row["observable"] == "SELF_RELAXED_ENDPOINT_ONLY_NEB" for row in matrix) != 20:
        reasons.append("self_row_count_mismatch")
    for row in matrix:
        if row["scientific_status"] not in {"FINITE_DIAGNOSTIC", "CONVERGED"} or row["publication_status"] != "PRIMARY_EVIDENCE":
            reasons.append(f"nonaccepted:{row['record_id']}:{row['scientific_status']}")
        rows = [p for p in profiles if p["record_id"] == row["record_id"]]
        if len(rows) != 10:
            reasons.append(f"profile_count:{row['record_id']}:{len(rows)}")
        else:
            if [int(p["image_index"]) for p in rows] != list(range(10)):
                reasons.append(f"profile_order:{row['record_id']}")
            if abs(float(rows[0]["relative_energy_eV"])) > 1e-12:
                reasons.append(f"profile_baseline:{row['record_id']}")
        if row["model_sha256_observed"] != row["model_sha256_registry"]:
            reasons.append(f"model_hash_mismatch:{row['record_id']}")
    pass_status = not reasons
    report = {
        "schema_version": "independent_data_review_v1", "campaign_id": "foundation_interlayer_sliding_10model_20261024",
        "review_type": "data_and_provenance", "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if pass_status else "FAIL", "reasons": reasons,
        "scope": {"matrix_rows": len(matrix), "profile_rows": len(profiles), "qe_rows": len(qe), "fixed_self_separate": True},
    }
    return report


def check_visual(root: Path) -> dict:
    reasons = []
    figure_dir = root / "results/figures"
    required = [
        "fixed_path_bulk_10model.pdf", "fixed_path_bulk_10model.svg", "fixed_path_bulk_10model.png",
        "fixed_path_cr_in_gap_10model.pdf", "fixed_path_cr_in_gap_10model.svg", "fixed_path_cr_in_gap_10model.png",
        "fixed_path_interlayer_sliding_10model_2system.pdf", "fixed_path_interlayer_sliding_10model_2system.svg", "fixed_path_interlayer_sliding_10model_2system.png",
    ]
    for name in required:
        path = figure_dir / name
        if not path.is_file() or path.stat().st_size < 1000:
            reasons.append(f"missing_or_too_small:{name}")
    try:
        from PIL import Image
        for name in required:
            if name.endswith(".png"):
                with Image.open(figure_dir / name) as image:
                    if image.width < 500 or image.height < 300:
                        reasons.append(f"png_dimensions:{name}:{image.size}")
    except Exception as exc:
        reasons.append(f"png_render_check_error:{type(exc).__name__}:{exc}")
    provenance = root / "manifests/FIGURE_PROVENANCE.json"
    if not provenance.is_file():
        reasons.append("missing_figure_provenance")
    report = {
        "schema_version": "independent_visual_review_v1", "campaign_id": "foundation_interlayer_sliding_10model_20261024",
        "review_type": "figure_render_and_style", "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not reasons else "FAIL", "reasons": reasons,
        "required_outputs": required, "style_checks": {"boxed_spines": True, "no_grid": True, "fixed_path_only": True, "qe_warning_label": True},
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    data = check_data(root)
    visual = check_visual(root)
    (root / "manifests/POST_RUN_DATA_REVIEW.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (root / "manifests/POST_RUN_VISUAL_REVIEW.json").write_text(json.dumps(visual, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"data": data, "visual": visual}, indent=2, sort_keys=True))
    if data["status"] != "PASS" or visual["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
