#!/usr/bin/env python3
"""Collect fixed/self row manifests into auditable campaign tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_manifests(root: Path, branch: str, filename: str) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for path in (root / "runs" / branch).glob("*/*/attempt_*/" + filename):
        payload = json.loads(path.read_text())
        record_id = payload["record_id"]
        found.setdefault(record_id, []).append(path)
    return found


def fixed_profile_rows(manifest_path: Path, manifest: dict) -> list[dict]:
    profile_path = manifest_path.parent / "fixed_profile.csv"
    if not profile_path.is_file():
        profile_path = Path(manifest["fixed_profile_csv"])
    rows = []
    with profile_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "record_id": manifest["record_id"],
                "observable": manifest["observable"],
                "system": manifest["system"],
                "plot_column": manifest["plot_column"],
                "model_id": manifest["model_id"],
                "model_label": manifest["model_label"],
                "image_index": int(row["image_index"]),
                "image_fraction": float(row["image_fraction"]),
                "energy_eV": float(row["predicted_energy_eV"]),
                "relative_energy_eV": float(row["relative_energy_eV"]),
                "source": "fixed_path_native_mace",
            })
    return rows


def self_profile_rows(manifest: dict) -> list[dict]:
    engine = manifest["engine"]
    profile = engine["profile"]
    n_images = len(profile)
    rows = []
    for row in profile:
        index = int(row["image_index"])
        rows.append({
            "record_id": manifest["record_id"],
            "observable": manifest["observable"],
            "system": manifest["system"],
            "plot_column": manifest["plot_column"],
            "model_id": manifest["model_id"],
            "model_label": manifest["model_label"],
            "image_index": index,
            "image_fraction": index / (n_images - 1),
            "energy_eV": float(row["energy_eV"]),
            "relative_energy_eV": float(row["rel_energy_eV"]),
            "source": "self_relaxed_endpoint_only_neb",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    tables = root / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)

    fixed_expected = {row["record_id"]: row for row in read_tsv(root / "jobs/JOBS_FIXED.tsv")}
    self_expected = {row["record_id"]: row for row in read_tsv(root / "jobs/JOBS_SELF.tsv")}
    expected = {**fixed_expected, **self_expected}
    fixed_paths = collect_manifests(root, "fixed", "fixed_path_manifest.json")
    self_paths = collect_manifests(root, "self", "endpoint_only_manifest.json")
    all_paths = {record_id: fixed_paths.get(record_id, []) + self_paths.get(record_id, []) for record_id in expected}

    matrix_fields = [
        "record_id", "observable", "system", "plot_column", "model_id", "model_label", "atoms", "n_images",
        "scientific_status", "publication_status", "barrier_eV", "input_sha256", "model_sha256_observed",
        "model_sha256_registry", "model_registry_sha256", "runner_sha256", "wrapper_sha256", "job_table_sha256", "job_id", "environment_sha256",
        "output_sha256", "failure_class", "failure_reason",
    ]
    matrix = []
    profiles = []
    failures = []
    duplicate_records = []
    for record_id, expected_row in expected.items():
        paths = all_paths[record_id]
        if len(paths) != 1:
            duplicate_records.append(record_id)
            matrix.append({
                "record_id": record_id,
                "observable": expected_row["observable"],
                "system": expected_row["system"],
                "plot_column": expected_row["plot_column"],
                "model_id": expected_row["model_id"],
                "model_label": expected_row["model_label"],
                "atoms": expected_row["atoms"],
                "n_images": expected_row["n_images"],
                "scientific_status": "DUPLICATE_OR_MISSING_ATTEMPTS",
                "publication_status": "DIAGNOSTIC_ONLY",
                "failure_class": "DUPLICATE_OR_MISSING_ATTEMPTS",
                "failure_reason": f"expected one manifest, found {len(paths)}",
            })
            failures.append({"record_id": record_id, "failure_class": "DUPLICATE_OR_MISSING_ATTEMPTS", "reason": f"found {len(paths)} manifests"})
            continue
        manifest_path = paths[0]
        manifest = json.loads(manifest_path.read_text())
        observable = manifest["observable"]
        if observable == "FIXED_PATH_SINGLE_POINT":
            profile_rows = fixed_profile_rows(manifest_path, manifest) if manifest.get("prediction_gate") == "PASS" else []
            barrier = max((float(row["relative_energy_eV"]) for row in profile_rows), default=None)
            output_sha = manifest.get("fixed_profile_csv_sha256")
            input_sha = manifest.get("input_xyz_sha256")
        else:
            profile_rows = self_profile_rows(manifest) if manifest.get("scientific_gate") == "PASS" else []
            barrier = max((float(row["relative_energy_eV"]) for row in profile_rows), default=None)
            engine = manifest.get("engine") or {}
            output_sha = (engine.get("outputs") or {}).get("images_sha256")
            input_sha = ";".join(filter(None, [manifest.get("initial_endpoint_sha256"), manifest.get("final_endpoint_sha256")]))
        profiles.extend(profile_rows)
        status = manifest.get("scientific_status", "UNKNOWN")
        failure_class = manifest.get("failure_class", "NA")
        failure_reason = manifest.get("failure_reason", "")
        matrix.append({
            "record_id": record_id,
            "observable": observable,
            "system": manifest.get("system", expected_row["system"]),
            "plot_column": manifest.get("plot_column", expected_row["plot_column"]),
            "model_id": manifest.get("model_id", expected_row["model_id"]),
            "model_label": manifest.get("model_label", expected_row["model_label"]),
            "atoms": manifest.get("atoms", expected_row["atoms"]),
            "n_images": manifest.get("n_images_target", expected_row["n_images"]),
            "scientific_status": status,
            "publication_status": manifest.get("publication_status", "DIAGNOSTIC_ONLY"),
            "barrier_eV": barrier,
            "input_sha256": input_sha,
            "model_sha256_observed": manifest.get("model_sha256_observed"),
            "model_sha256_registry": manifest.get("model_sha256_registry"),
            "model_registry_sha256": manifest.get("model_registry_sha256"),
            "runner_sha256": manifest.get("runner_sha256"),
            "wrapper_sha256": manifest.get("wrapper_sha256"),
            "job_table_sha256": manifest.get("job_table_sha256"),
            "job_id": manifest.get("slurm_job_id"),
            "environment_sha256": manifest.get("environment_sha256"),
            "output_sha256": output_sha,
            "failure_class": failure_class,
            "failure_reason": failure_reason,
        })
        if status not in ("FINITE_DIAGNOSTIC", "CONVERGED") or not profile_rows:
            failures.append({"record_id": record_id, "failure_class": failure_class, "reason": failure_reason})

    qe_rows = []
    with (root / "inputs/dft_bpath2/combined_sb2te3_energy.csv").open(newline="") as handle:
        raw_qe_rows = list(csv.DictReader(handle))
        baselines = {"bulk": float(raw_qe_rows[0]["1.bulk"]), "cr_in_gap": float(raw_qe_rows[0]["2.cr-in-gap"])}
        for index, row in enumerate(raw_qe_rows):
            x = float(row["Reaction_Coord"])
            for system, column in (("bulk", "1.bulk"), ("cr_in_gap", "2.cr-in-gap")):
                raw = float(row[column])
                qe_rows.append({
                    "system": system,
                    "plot_column": "Bulk" if system == "bulk" else "Cr-in-gap",
                    "image_index": index,
                    "image_fraction": index / 9.0,
                    "reaction_coord_raw": x,
                    "raw_energy_eV": raw,
                    "relative_energy_eV": raw - baselines[system],
                    "dft_status": "COMPLETED_BUT_SCF_WARNING",
                    "source": "combined_sb2te3_energy.csv",
                })

    write_tsv(tables / "RESULT_MATRIX.tsv", matrix, matrix_fields)
    write_tsv(tables / "IMAGE_PROFILES.tsv", profiles, [
        "record_id", "observable", "system", "plot_column", "model_id", "model_label", "image_index", "image_fraction", "energy_eV", "relative_energy_eV", "source",
    ])
    write_tsv(tables / "QE_REFERENCE.tsv", qe_rows, [
        "system", "plot_column", "image_index", "image_fraction", "reaction_coord_raw", "raw_energy_eV", "relative_energy_eV", "dft_status", "source",
    ])
    write_tsv(tables / "FAILURE_REGISTRY.tsv", failures, ["record_id", "failure_class", "reason"])

    pass_rows = sum(row["scientific_status"] in ("FINITE_DIAGNOSTIC", "CONVERGED") for row in matrix)
    all_pass = len(matrix) == 12 and pass_rows == 12 and not duplicate_records and all(len([r for r in profiles if r["record_id"] == row["record_id"]]) == 10 for row in matrix)
    collection = {
        "schema_version": "foundation_interlayer_sliding_2x2_collection_v1",
        "campaign_id": "foundation_interlayer_sliding_2x2_20261024",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_rows": 12,
        "observed_rows": len(matrix),
        "scientifically_passing_rows": pass_rows,
        "profile_rows": len(profiles),
        "duplicate_or_missing_records": duplicate_records,
        "collection_status": "COMPLETE" if all_pass else "INCOMPLETE",
        "acceptance_freeze": str(root / "ACCEPTANCE_FREEZE_R13.json"),
        "acceptance_freeze_sha256": sha256(root / "ACCEPTANCE_FREEZE_R13.json"),
        "model_registry_sha256": sha256(root / "MODEL_REGISTRY.tsv"),
        "result_matrix_sha256": sha256(tables / "RESULT_MATRIX.tsv"),
        "image_profiles_sha256": sha256(tables / "IMAGE_PROFILES.tsv"),
        "qe_reference_sha256": sha256(tables / "QE_REFERENCE.tsv"),
        "failure_registry_sha256": sha256(tables / "FAILURE_REGISTRY.tsv"),
    }
    (root / "manifests/COLLECTION_MANIFEST.json").write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n")
    print(json.dumps(collection, indent=2, sort_keys=True))
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
