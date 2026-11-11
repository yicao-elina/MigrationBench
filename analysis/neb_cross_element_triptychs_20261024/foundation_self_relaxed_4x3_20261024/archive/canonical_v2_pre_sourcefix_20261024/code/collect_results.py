#!/usr/bin/env python3
"""Collect row manifests and self-relaxed image profiles without changing raw runs."""

from __future__ import annotations

import csv
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


def choose_attempt(root: Path, row: dict[str, str]) -> Path | None:
    base = root / "runs" / row["initialization_method"].lower() / row["model_id"] / row["path_id"]
    attempts = sorted(item for item in base.glob("attempt_*") if item.is_dir())
    return attempts[-1] if attempts else None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader((root / "JOB_TABLE_IDPP.tsv").open(), delimiter="\t"))
    collected: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row in rows:
        attempt = choose_attempt(root, row)
        manifest_path = attempt / "endpoint_only_manifest.json" if attempt else None
        record: dict[str, object] = {
            **row,
            "attempt_dir": str(attempt) if attempt else "",
            "manifest_path": str(manifest_path) if manifest_path and manifest_path.exists() else "",
            "manifest_sha256": sha256(manifest_path) if manifest_path and manifest_path.exists() else "",
            "scientific_status": "MISSING_OUTPUT",
            "publication_status": "DIAGNOSTIC_ONLY",
            "failure_class": "MISSING_OUTPUT",
            "failure_reason": "no endpoint_only_manifest.json found",
            "optimizer_converged": "false",
            "target_model_convergence": "false",
            "n_profile_images": 0,
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
        if record["scientific_status"] != "CONVERGED":
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
    manifest = {
        "schema_version": "foundation_self_relaxed_collected_v1",
        "campaign_id": "foundation_self_relaxed_4x3_20261024",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_rows": len(rows),
        "rows_with_manifests": sum(bool(item["manifest_path"]) for item in collected),
        "converged_rows": sum(item["scientific_status"] == "CONVERGED" for item in collected),
        "profile_rows": len(profiles),
        "failure_rows": len(failures),
        "accepted_profile_rows_only": True,
        "source": "remote-or-local runs/idpp manifests; raw run directories are not rewritten",
        "outputs": {
            name: sha256(data / name)
            for name in ("ROW_RESULTS.tsv", "IMAGE_PROFILES.tsv", "FAILURE_REGISTRY.tsv")
        },
    }
    (data / "COLLECTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

