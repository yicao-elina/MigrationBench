#!/usr/bin/env python3
"""Collect and gate the complete 10-model sliding campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_STATUS = {"FINITE_DIAGNOSTIC", "CONVERGED"}
EXPECTED_MODELS = {
    "foundation_agnesi_large", "foundation_agnesi_medium_density", "foundation_agnesi_small_density",
    "foundation_agnesi_medium", "foundation_mp0b3", "foundation_omat0_medium", "foundation_omat0_small",
    "foundation_mh0", "foundation_mh1", "foundation_matpes_pbe_omat_ft",
}
EXPECTED_SYSTEMS = {"bulk", "cr_in_gap"}


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


def discover(root: Path, branch: str, filename: str) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    for path in (root / "runs" / branch).glob("*/*/attempt_*/" + filename):
        try:
            payload = json.loads(path.read_text())
            found[payload["record_id"]].append(path)
        except Exception:
            continue
    return found


def choose(paths: list[Path]) -> tuple[Path | None, str]:
    if len(paths) == 1:
        return paths[0], "ONE_ATTEMPT"
    accepted = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
            if payload.get("scientific_status") in EXPECTED_STATUS:
                accepted.append(path)
        except Exception:
            pass
    if len(accepted) == 1 and len(paths) > 1:
        # Failed attempts remain in ATTEMPT_LEDGER; one accepted terminal
        # attempt is a recoverable retry, not an unresolved duplicate.
        return accepted[0], "ONE_ACCEPTED_RETRY_PLUS_FAILURES"
    return None, "DUPLICATE_OR_MISSING_ATTEMPTS"


def load_profile(manifest_path: Path, manifest: dict) -> list[dict]:
    if manifest["observable"] == "FIXED_PATH_SINGLE_POINT":
        profile_path = manifest_path.parent / "fixed_profile.csv"
        with profile_path.open(newline="") as handle:
            raw = list(csv.DictReader(handle))
        if len(raw) != int(manifest["n_images_target"]):
            raise ValueError(f"fixed profile length {len(raw)}")
        return [{
            "record_id": manifest["record_id"], "observable": manifest["observable"], "system": manifest["system"],
            "plot_column": manifest["plot_column"], "model_id": manifest["model_id"], "model_label": manifest["model_label"],
            "image_index": int(row["image_index"]), "image_fraction": float(row["image_fraction"]),
            "energy_eV": float(row["predicted_energy_eV"]), "relative_energy_eV": float(row["relative_energy_eV"]),
            "source": "fixed_path_ase_mace",
        } for row in raw]
    engine = manifest.get("engine") or {}
    raw = engine.get("profile") or []
    if len(raw) != int(manifest["n_images_target"]):
        raise ValueError(f"self profile length {len(raw)}")
    return [{
        "record_id": manifest["record_id"], "observable": manifest["observable"], "system": manifest["system"],
        "plot_column": manifest["plot_column"], "model_id": manifest["model_id"], "model_label": manifest["model_label"],
        "image_index": int(row["image_index"]), "image_fraction": int(row["image_index"]) / (len(raw) - 1),
        "energy_eV": float(row["energy_eV"]), "relative_energy_eV": float(row["rel_energy_eV"]),
        "source": "self_relaxed_endpoint_only_neb",
    } for row in raw]


def matrix_row(manifest: dict, profile: list[dict], manifest_path: Path) -> dict:
    barrier = max(float(row["relative_energy_eV"]) for row in profile) if profile else None
    engine = manifest.get("engine") or {}
    if manifest["observable"] == "FIXED_PATH_SINGLE_POINT":
        output_sha = manifest.get("fixed_profile_csv_sha256")
        input_sha = manifest.get("input_xyz_sha256")
    else:
        output_sha = (engine.get("outputs") or {}).get("images_sha256")
        input_sha = ";".join(filter(None, [manifest.get("initial_endpoint_sha256"), manifest.get("final_endpoint_sha256")]))
    return {
        "record_id": manifest["record_id"], "observable": manifest["observable"], "system": manifest["system"],
        "plot_column": manifest["plot_column"], "model_id": manifest["model_id"], "model_label": manifest["model_label"],
        "model_head": manifest.get("model_head", ""), "atoms": manifest["atoms"], "n_images": manifest["n_images_target"],
        "scientific_status": manifest.get("scientific_status", "UNKNOWN"), "publication_status": manifest.get("publication_status", "DIAGNOSTIC_ONLY"),
        "barrier_eV": "" if barrier is None else f"{barrier:.16g}", "input_sha256": input_sha or "",
        "model_sha256_observed": manifest.get("model_sha256_observed", ""), "model_sha256_registry": manifest.get("model_sha256_registry", ""),
        "model_registry_sha256": manifest.get("model_registry_sha256", ""), "runner_sha256": manifest.get("runner_sha256", ""),
        "wrapper_sha256": manifest.get("wrapper_sha256", ""), "job_table_sha256": manifest.get("job_table_sha256", ""),
        "job_id": manifest.get("job_id", ""), "environment_sha256": manifest.get("environment_sha256", ""),
        "output_sha256": output_sha or "", "manifest_path": str(manifest_path),
        "failure_class": manifest.get("failure_class", "NA"), "failure_reason": manifest.get("failure_reason", ""),
    }


def read_qe(root: Path) -> list[dict]:
    path = root / "inputs/dft_bpath2/combined_sb2te3_energy.csv"
    with path.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 10:
        raise ValueError(f"expected ten QE rows, found {len(raw)}")
    output = []
    for system, column, label in (("bulk", "1.bulk", "Bulk"), ("cr_in_gap", "2.cr-in-gap", "Cr-in-gap")):
        values = [float(row[column]) for row in raw]
        baseline = values[0]
        for index, (source, value) in enumerate(zip(raw, values)):
            output.append({
                "system": system, "plot_column": label, "image_index": index, "image_fraction": index / 9.0,
                "reaction_coord_raw": source["Reaction_Coord"], "raw_energy_eV": value,
                "relative_energy_eV": value - baseline, "dft_status": "COMPLETED_BUT_SCF_WARNING",
                "source": "combined_sb2te3_energy.csv",
            })
    return output


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
    fixed_found = discover(root, "fixed", "fixed_path_manifest.json")
    self_found = discover(root, "self", "endpoint_only_manifest.json")
    all_found = {record_id: fixed_found.get(record_id, []) + self_found.get(record_id, []) for record_id in expected}

    fields = [
        "record_id", "observable", "system", "plot_column", "model_id", "model_label", "model_head", "atoms", "n_images",
        "scientific_status", "publication_status", "barrier_eV", "input_sha256", "model_sha256_observed", "model_sha256_registry",
        "model_registry_sha256", "runner_sha256", "wrapper_sha256", "job_table_sha256", "job_id", "environment_sha256", "output_sha256",
        "manifest_path", "failure_class", "failure_reason",
    ]
    matrix: list[dict] = []
    profiles: list[dict] = []
    failures: list[dict] = []
    attempts: list[dict] = []
    duplicate_or_missing: list[str] = []
    for record_id, expected_row in expected.items():
        paths = all_found.get(record_id, [])
        for path in paths:
            try:
                payload = json.loads(path.read_text())
                attempts.append({"record_id": record_id, "manifest_path": str(path), "scientific_status": payload.get("scientific_status", "UNKNOWN"), "failure_class": payload.get("failure_class", "NA"), "failure_reason": payload.get("failure_reason", "")})
            except Exception as exc:
                attempts.append({"record_id": record_id, "manifest_path": str(path), "scientific_status": "UNREADABLE", "failure_class": "MANIFEST_PARSE", "failure_reason": str(exc)})
        chosen, choice = choose(paths)
        if chosen is None:
            duplicate_or_missing.append(record_id)
            row = {"record_id": record_id, "observable": expected_row["observable"], "system": expected_row["system"], "plot_column": expected_row["plot_column"], "model_id": expected_row["model_id"], "model_label": expected_row["model_label"], "model_head": expected_row.get("model_head", ""), "atoms": expected_row["atoms"], "n_images": expected_row["n_images"], "scientific_status": "DUPLICATE_OR_MISSING_ATTEMPTS", "publication_status": "DIAGNOSTIC_ONLY", "barrier_eV": "", "input_sha256": "", "model_sha256_observed": "", "model_sha256_registry": "", "model_registry_sha256": "", "runner_sha256": "", "wrapper_sha256": "", "job_table_sha256": "", "job_id": "", "environment_sha256": "", "output_sha256": "", "manifest_path": ";".join(map(str, paths)), "failure_class": "DUPLICATE_OR_MISSING_ATTEMPTS", "failure_reason": f"expected one accepted terminal manifest; found {len(paths)} attempts"}
            matrix.append(row)
            failures.append({"record_id": record_id, "failure_class": row["failure_class"], "reason": row["failure_reason"]})
            continue
        payload = json.loads(chosen.read_text())
        try:
            profile = load_profile(chosen, payload) if payload.get("scientific_status") in EXPECTED_STATUS else []
        except Exception as exc:
            profile = []
            payload["failure_class"] = "PROFILE_PARSE"
            payload["failure_reason"] = f"{type(exc).__name__}:{exc}"
        row = matrix_row(payload, profile, chosen)
        matrix.append(row)
        profiles.extend(profile)
        if row["scientific_status"] not in EXPECTED_STATUS or len(profile) != int(expected_row["n_images"]):
            failures.append({"record_id": record_id, "failure_class": row["failure_class"], "reason": row["failure_reason"] or "scientific/profile gate failed"})

    qe = read_qe(root)
    write_tsv(tables / "RESULT_MATRIX.tsv", matrix, fields)
    write_tsv(tables / "IMAGE_PROFILES.tsv", profiles, ["record_id", "observable", "system", "plot_column", "model_id", "model_label", "image_index", "image_fraction", "energy_eV", "relative_energy_eV", "source"])
    write_tsv(tables / "QE_REFERENCE.tsv", qe, ["system", "plot_column", "image_index", "image_fraction", "reaction_coord_raw", "raw_energy_eV", "relative_energy_eV", "dft_status", "source"])
    write_tsv(tables / "FAILURE_REGISTRY.tsv", failures, ["record_id", "failure_class", "reason"])
    write_tsv(tables / "ATTEMPT_LEDGER.tsv", attempts, ["record_id", "manifest_path", "scientific_status", "failure_class", "failure_reason"])
    fixed_summary = [row for row in matrix if row["observable"] == "FIXED_PATH_SINGLE_POINT"]
    self_summary = [row for row in matrix if row["observable"] == "SELF_RELAXED_ENDPOINT_ONLY_NEB"]
    write_tsv(tables / "FIXED_PATH_SUMMARY.tsv", fixed_summary, fields)
    write_tsv(tables / "SELF_RELAXED_SUMMARY.tsv", self_summary, fields)

    by_key = {(row["model_id"], row["system"]): row for row in matrix}
    sliding = []
    for model_id in sorted(EXPECTED_MODELS):
        for system in ("bulk", "cr_in_gap"):
            fixed = by_key.get((model_id, system)) if False else None
            fixed = next((row for row in matrix if row["model_id"] == model_id and row["system"] == system and row["observable"] == "FIXED_PATH_SINGLE_POINT"), None)
            self = next((row for row in matrix if row["model_id"] == model_id and row["system"] == system and row["observable"] == "SELF_RELAXED_ENDPOINT_ONLY_NEB"), None)
            sliding.append({"model_id": model_id, "system": system, "fixed_status": fixed["scientific_status"] if fixed else "MISSING", "fixed_barrier_eV": fixed["barrier_eV"] if fixed else "", "fixed_record_id": fixed["record_id"] if fixed else "", "self_status": self["scientific_status"] if self else "MISSING", "self_barrier_proxy_eV": self["barrier_eV"] if self else "", "self_record_id": self["record_id"] if self else ""})
    write_tsv(tables / "SLIDING_MATRIX.tsv", sliding, ["model_id", "system", "fixed_status", "fixed_barrier_eV", "fixed_record_id", "self_status", "self_barrier_proxy_eV", "self_record_id"])

    accepted = [row for row in matrix if row["scientific_status"] in EXPECTED_STATUS]
    expected_complete = (
        len(matrix) == 40 and len(accepted) == 40 and not duplicate_or_missing and len(profiles) == 400 and
        {row["model_id"] for row in matrix} == EXPECTED_MODELS and {row["system"] for row in matrix} == EXPECTED_SYSTEMS and
        sum(row["observable"] == "FIXED_PATH_SINGLE_POINT" for row in matrix) == 20 and sum(row["observable"] == "SELF_RELAXED_ENDPOINT_ONLY_NEB" for row in matrix) == 20 and
        all(sum(p["record_id"] == row["record_id"] for p in profiles) == 10 for row in accepted)
    )
    collection = {
        "schema_version": "foundation_interlayer_sliding_10model_collection_v1",
        "campaign_id": "foundation_interlayer_sliding_10model_20261024",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_rows": 40, "observed_rows": len(matrix), "accepted_rows": len(accepted), "profile_rows": len(profiles),
        "fixed_rows": sum(row["observable"] == "FIXED_PATH_SINGLE_POINT" for row in matrix), "self_rows": sum(row["observable"] == "SELF_RELAXED_ENDPOINT_ONLY_NEB" for row in matrix),
        "duplicate_or_missing_records": duplicate_or_missing, "collection_status": "COMPLETE" if expected_complete else "INCOMPLETE",
        "acceptance_freeze": str(root / "ACCEPTANCE_FREEZE_R01.json"), "acceptance_freeze_sha256": sha256(root / "ACCEPTANCE_FREEZE_R01.json"),
        "model_registry_sha256": sha256(root / "MODEL_REGISTRY.tsv"), "result_matrix_sha256": sha256(tables / "RESULT_MATRIX.tsv"),
        "image_profiles_sha256": sha256(tables / "IMAGE_PROFILES.tsv"), "qe_reference_sha256": sha256(tables / "QE_REFERENCE.tsv"),
        "failure_registry_sha256": sha256(tables / "FAILURE_REGISTRY.tsv"), "attempt_ledger_sha256": sha256(tables / "ATTEMPT_LEDGER.tsv"),
    }
    manifest_path = root / "manifests/COLLECTION_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n")
    print(json.dumps(collection, indent=2, sort_keys=True))
    if not expected_complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
