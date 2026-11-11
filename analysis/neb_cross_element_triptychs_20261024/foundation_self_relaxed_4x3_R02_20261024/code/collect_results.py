#!/usr/bin/env python3
"""Collect row manifests and self-relaxed image profiles without changing raw runs."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def manifest_retry_round(manifest: dict[str, object]) -> str:
    retry_round = str(manifest.get("retry_round", ""))
    if retry_round:
        return retry_round
    role = str(manifest.get("manifest_role", ""))
    return role.removeprefix("RETRY_ARRAY_") if role.startswith("RETRY_ARRAY_") else ""


def load_retry_registry(root: Path) -> dict[str, dict[str, str]]:
    registry: dict[str, dict[str, str]] = {}
    for path in sorted(root.glob("JOB_TABLE_IDPP_RETRY_*.tsv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                registry[row["record_id"]] = {
                    "manifest_role": row.get("manifest_role", ""),
                    "retry_round": row.get("retry_round", ""),
                    "retry_reason": row.get("retry_reason", ""),
                }
    return registry


def choose_attempt(root: Path, row: dict[str, str]) -> tuple[Path | None, list[dict[str, object]]]:
    base = root / "runs" / row["initialization_method"].lower() / row["model_id"] / row["path_id"]
    attempts = sorted(item for item in base.glob("attempt_*") if item.is_dir())
    bookkeeping_path = root / "EXECUTION_BOOKKEEPING_R02.json"
    bookkeeping = json.loads(bookkeeping_path.read_text()) if bookkeeping_path.is_file() else {}
    smoke_job_id = str(bookkeeping.get("smoke_job_id", "")).split("_")[0]
    formal_job_id = str(bookkeeping.get("formal_array_job_id", ""))
    ledger: list[dict[str, object]] = []
    formal_candidates: list[Path] = []
    for attempt in attempts:
        manifest_path = attempt / "endpoint_only_manifest.json"
        manifest = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = {}
        attempt_id = attempt.name
        role = manifest.get("execution_role") or row.get("execution_role", "UNKNOWN")
        role_source = "manifest_or_job_table"
        observed_job_id = str(manifest.get("slurm_job_id", ""))
        if smoke_job_id and observed_job_id == smoke_job_id:
            role = "SMOKE"
            role_source = "execution_bookkeeping.smoke_job_id"
        if formal_job_id and observed_job_id == formal_job_id:
            role = "FORMAL_ARRAY"
            role_source = "execution_bookkeeping.formal_array_job_id"
        if role == "FORMAL_ARRAY":
            formal_candidates.append(attempt)
        ledger.append(
            {
                "record_id": row["record_id"],
                "attempt_id": attempt_id,
                "attempt_dir": str(attempt),
                "run_role": role,
                "role_source": role_source,
                "manifest_role": manifest.get("manifest_role", ""),
                "retry_round": manifest_retry_round(manifest),
                "retry_reason": manifest.get("retry_reason", ""),
                "parent_array_job_id": formal_job_id,
                "slurm_job_id": manifest.get("slurm_job_id", ""),
                "slurm_array_task_id": manifest.get("slurm_array_task_id", ""),
                "manifest_path": str(manifest_path) if manifest_path.is_file() else "",
                "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else "",
                "scientific_status": manifest.get("scientific_status", "PENDING_OR_MISSING"),
                "failure_class": manifest.get("failure_class", ""),
                "selected_attempt": "NO",
                "selection_reason": "not selected; retained in attempt ledger",
            }
        )
    selected = (formal_candidates[-1] if formal_candidates else attempts[-1]) if attempts else None
    for item in ledger:
        if selected and item["attempt_id"] == selected.name:
            item["selected_attempt"] = "YES"
            item["selection_reason"] = "FORMAL_ARRAY_PRIMARY" if item["run_role"] == "FORMAL_ARRAY" else "latest available attempt"
    return selected, ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="mark no-manifest rows as final missing-output failures")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    campaign_manifest = json.loads((root / "CAMPAIGN_MANIFEST.json").read_text())
    campaign_id = campaign_manifest.get("campaign_id", root.name)
    rows = list(csv.DictReader((root / "JOB_TABLE_IDPP.tsv").open(), delimiter="\t"))
    retry_registry = load_retry_registry(root)
    collected: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    attempt_ledger: list[dict[str, object]] = []
    for row in rows:
        attempt, attempt_rows = choose_attempt(root, row)
        retry_meta = retry_registry.get(row["record_id"], {})
        for attempt_row in attempt_rows:
            if attempt_row.get("manifest_role", "").startswith("RETRY_ARRAY_") or attempt_row.get("retry_round"):
                attempt_row["retry_round"] = attempt_row.get("retry_round") or retry_meta.get("retry_round", "")
                attempt_row["retry_reason"] = attempt_row.get("retry_reason") or retry_meta.get("retry_reason", "")
        attempt_ledger.extend(attempt_rows)
        manifest_path = attempt / "endpoint_only_manifest.json" if attempt else None
        record: dict[str, object] = {
            **row,
            "attempt_dir": str(attempt) if attempt else "",
            "manifest_path": str(manifest_path) if manifest_path and manifest_path.exists() else "",
            "manifest_sha256": sha256(manifest_path) if manifest_path and manifest_path.exists() else "",
            "scientific_status": "MISSING_OUTPUT" if args.final else "PENDING",
            "publication_status": "DIAGNOSTIC_ONLY",
            "failure_class": "MISSING_OUTPUT" if args.final else "PENDING",
            "failure_reason": "no endpoint_only_manifest.json found",
            "optimizer_converged": "false",
            "target_model_convergence": "false",
            "n_profile_images": 0,
            "attempt_count": len(attempt_rows),
            "selected_attempt_id": attempt.name if attempt else "",
            "selected_attempt_role": next((item["run_role"] for item in attempt_rows if item["attempt_id"] == (attempt.name if attempt else "")), ""),
            "selection_reason": next((item["selection_reason"] for item in attempt_rows if item["attempt_id"] == (attempt.name if attempt else "")), ""),
        }
        manifest = None
        if manifest_path and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
                record.update(
                    {
                        "scientific_status": manifest.get("scientific_status", "UNKNOWN"),
                        "publication_status": manifest.get("publication_status", "DIAGNOSTIC_ONLY"),
                        "failure_class": manifest.get("failure_class", "UNKNOWN"),
                        "failure_reason": manifest.get("failure_reason", ""),
                        "manifest_role": manifest.get("manifest_role", record.get("manifest_role", "")),
                        "selected_attempt_manifest_role": manifest.get("manifest_role", ""),
                        "retry_round": manifest_retry_round(manifest) or retry_meta.get("retry_round", ""),
                        "retry_reason": manifest.get("retry_reason", "") or retry_meta.get("retry_reason", ""),
                        "selected_attempt_execution_role": record.get("selected_attempt_role", ""),
                        "optimizer_converged": str(bool(manifest.get("optimizer_converged", False))).lower(),
                        "target_model_convergence": str(bool(manifest.get("target_model_convergence", False))).lower(),
                        "engine_manifest": manifest.get("engine_manifest", ""),
                        "engine_manifest_sha256": manifest.get("engine_manifest_sha256", ""),
                        "trajectory_sha256": manifest.get("trajectory_sha256", ""),
                        "final_max_internal_neb_force_eV_A": manifest.get("final_max_internal_neb_force_eV_A", ""),
                        "final_max_neb_residual_eV_A": manifest.get("final_max_neb_residual_eV_A", ""),
                        "forward_barrier_eV": manifest.get("forward_barrier_eV", ""),
                        "slurm_job_id": manifest.get("slurm_job_id", ""),
                        "slurm_array_task_id": manifest.get("slurm_array_task_id", ""),
                    }
                )
            except Exception as exc:
                record["scientific_status"] = "FAILURE_MODE"
                record["failure_class"] = "MANIFEST_PARSE"
                record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        engine_path = attempt / "mlff_neb_manifest.json" if attempt else None
        if engine_path and engine_path.is_file():
            try:
                engine = json.loads(engine_path.read_text())
                profile = engine.get("profile", [])
                valid = len(profile) == int(row["n_images"])
                indexes = []
                for point in profile:
                    indexes.append(int(point["image_index"]))
                    valid = valid and all(
                        finite(point.get(key))
                        for key in ("energy_eV", "rel_energy_eV", "fmax_eV_A")
                    )
                valid = valid and indexes == list(range(int(row["n_images"])))
                record["n_profile_images"] = len(profile)
                if not valid:
                    record["scientific_status"] = "FAILURE_MODE"
                    record["publication_status"] = "DIAGNOSTIC_ONLY"
                    record["failure_class"] = "PROFILE_INTEGRITY"
                    record["failure_reason"] = "profile is not finite or not exactly ordered 10 images"
                else:
                    for point in profile:
                        profiles.append(
                            {
                                "record_id": row["record_id"],
                                "element": row["element"],
                                "path_id": row["path_id"],
                                "path_class": row["path_class"],
                                "path_label": row["path_label"],
                                "model_id": row["model_id"],
                                "model_label": row["model_label"],
                                "model_family": row["model_family"],
                                "image_index": int(point["image_index"]),
                                "s": int(point["image_index"]) / (int(row["n_images"]) - 1),
                                "energy_eV": float(point["energy_eV"]),
                                "rel_energy_eV": float(point["rel_energy_eV"]),
                                "fmax_eV_A": float(point["fmax_eV_A"]),
                                "scientific_status": record["scientific_status"],
                                "publication_status": record["publication_status"],
                                "attempt_dir": str(attempt),
                            }
                        )
            except Exception as exc:
                record["scientific_status"] = "FAILURE_MODE"
                record["publication_status"] = "DIAGNOSTIC_ONLY"
                record["failure_class"] = "ENGINE_MANIFEST_PARSE"
                record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        if record["scientific_status"] not in ("CONVERGED", "PENDING"):
            failures.append(
                {
                    "record_id": row["record_id"],
                    "element": row["element"],
                    "path_id": row["path_id"],
                    "model_id": row["model_id"],
                    "scientific_status": record["scientific_status"],
                    "failure_class": record["failure_class"],
                    "failure_reason": record["failure_reason"],
                    "attempt_dir": record["attempt_dir"],
                }
            )
        collected.append(record)

    data = root / "data"
    data.mkdir(exist_ok=True)

    def write_tsv(name: str, items: list[dict[str, object]]) -> None:
        if not items:
            (data / name).write_text("")
            return
        fields = list(items[0])
        with (data / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(items)

    write_tsv("ROW_RESULTS.tsv", collected)
    write_tsv("IMAGE_PROFILES.tsv", profiles)
    write_tsv("FAILURE_REGISTRY.tsv", failures)
    if not failures:
        (data / "FAILURE_REGISTRY.tsv").write_text(
            "record_id\telement\tpath_id\tmodel_id\tscientific_status\tfailure_class\tfailure_reason\tattempt_dir\n"
        )
    write_tsv("ATTEMPT_LEDGER.tsv", attempt_ledger)
    manifest = {
        "schema_version": "foundation_self_relaxed_collected_v1",
        "campaign_id": campaign_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_rows": len(rows),
        "rows_with_manifests": sum(bool(item["manifest_path"]) for item in collected),
        "converged_rows": sum(item["scientific_status"] == "CONVERGED" for item in collected),
        "profile_rows": len(profiles),
        "failure_rows": len(failures),
        "attempt_rows": len(attempt_ledger),
        "accepted_profile_rows_only": True,
        "finalized": bool(args.final),
        "source": "remote-or-local runs/idpp manifests; raw run directories are not rewritten",
        "outputs": {
            name: sha256(data / name)
            for name in ("ROW_RESULTS.tsv", "IMAGE_PROFILES.tsv", "FAILURE_REGISTRY.tsv", "ATTEMPT_LEDGER.tsv")
        },
    }
    (data / "COLLECTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

