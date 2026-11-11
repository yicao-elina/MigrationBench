#!/usr/bin/env python3
"""Collect fixed-path runs, compute auditable metrics, and enforce stage gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, model, read_tsv, root_path, sha256_file, system, write_json, write_tsv


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def dft_profile(root: Path, config: dict, system_id: str) -> tuple[np.ndarray, np.ndarray]:
    spec = system(config, system_id)
    rows = read_csv_rows(root_path(root, spec["dft_csv"]))
    return (
        np.asarray([float(row["Reaction_Coord"]) for row in rows], dtype=float),
        np.asarray([float(row[spec["dft_column"]]) for row in rows], dtype=float),
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def latest_manifest(root: Path, stage: str, model_id: str, system_id: str, name: str) -> tuple[Path | None, list[Path]]:
    matches = sorted((root / "runs" / stage / model_id / system_id).glob(f"attempt_*/{name}"))
    return (matches[-1] if matches else None, matches)


def read_profile(path: Path, relative_column: str = "relative_energy_eV") -> list[dict[str, str]]:
    return read_csv_rows(path)


def fixed_metric(root: Path, config: dict, manifest_path: Path, profile_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    model_id = manifest["model_id"]
    system_id = manifest["system_id"]
    dft_x, dft_y = dft_profile(root, config, system_id)
    rows = read_profile(profile_path)
    values = np.asarray([float(row["relative_energy_eV"]) for row in rows], dtype=float) if rows and all(row["relative_energy_eV"] != "" for row in rows) else np.asarray([], dtype=float)
    finite = len(values) == len(dft_y) and np.isfinite(values).all()
    if finite:
        delta = values - dft_y
        barrier = float(values.max())
        peak_index = int(values.argmax())
        endpoint_drift = float(values[-1])
        rmse = float(np.sqrt(np.mean(delta**2)))
        mae = float(np.mean(np.abs(delta)))
        rho = spearman(values, dft_y)
    else:
        barrier = peak_index = endpoint_drift = rmse = mae = rho = float("nan")
    spec = model(config, model_id)
    return {
        "model_id": model_id,
        "model_label": spec["label"],
        "system_id": system_id,
        "model_family": spec["family"],
        "scratch_subfamily": spec.get("scratch_subfamily", "none"),
        "training_seed": spec.get("training_seed", "NA"),
        "seed_semantics": spec.get("seed_semantics", "not_applicable"),
        "model_sha256": manifest.get("actual_model_sha256", ""),
        "expected_model_sha256": spec["model_sha256"],
        "model_hash_match": manifest.get("model_hash_match", False),
        "scientific_status": manifest.get("scientific_status", "UNKNOWN"),
        "publication_status": manifest.get("publication_status", "UNKNOWN"),
        "run_status": manifest.get("status", "UNKNOWN"),
        "observed_images": manifest.get("observed_images", len(rows)),
        "observed_atoms": ";".join(str(value) for value in manifest.get("observed_atoms", [])),
        "finite_profile": finite,
        "max_relative_energy_eV": barrier,
        "peak_image_index": peak_index,
        "endpoint_drift_eV": endpoint_drift,
        "rmse_dft_eV": rmse,
        "mae_dft_eV": mae,
        "spearman_dft": rho,
        "dft_peak_eV": float(dft_y.max()),
        "dft_peak_image_index": int(dft_y.argmax()),
        "dft_endpoint_drift_eV": float(dft_y[-1]),
        "profile_path": str(profile_path),
        "manifest_path": str(manifest_path),
        "profile_sha256": sha256_file(profile_path) if profile_path.is_file() else "",
    }


def collect_stage_rows(root: Path, config: dict, stage: str, model_ids: list[str], profile_name: str, manifest_name: str) -> tuple[list[dict], list[dict]]:
    metrics = []
    attempts = []
    for model_id in model_ids:
        for system_id in config["systems"]:
            manifest_path, all_manifests = latest_manifest(root, stage, model_id, system_id, manifest_name)
            for candidate in all_manifests:
                attempts.append({
                    "stage": stage,
                    "model_id": model_id,
                    "system_id": system_id,
                    "manifest_path": str(candidate),
                    "selected": candidate == manifest_path,
                })
            if manifest_path is None:
                metrics.append({"model_id": model_id, "system_id": system_id, "run_status": "MISSING_MANIFEST", "finite_profile": False})
                continue
            profile_path = manifest_path.parent / profile_name
            if not profile_path.is_file():
                metrics.append({"model_id": model_id, "system_id": system_id, "run_status": "MISSING_PROFILE", "finite_profile": False, "manifest_path": str(manifest_path)})
                continue
            metrics.append(fixed_metric(root, config, manifest_path, profile_path) if stage == "fixed" else {"model_id": model_id, "system_id": system_id, "run_status": json.loads(manifest_path.read_text()).get("status", "UNKNOWN"), "finite_profile": False, "manifest_path": str(manifest_path)})
    return metrics, attempts


def old_relative(path: Path, model_name: str) -> np.ndarray:
    rows = [row for row in read_csv_rows(path) if row["model_name"] == model_name]
    rows.sort(key=lambda row: int(row["image_index"]))
    energies = np.asarray([float(row["predicted_energy_eV"]) for row in rows], dtype=float)
    return energies - energies[0]


def collect_a0(root: Path, config: dict, a0_ids: list[str]) -> list[dict]:
    results = []
    legacy_paths = {
        "pristine": root_path(root, config["legacy_inputs"]["pristine_predictions"]),
        "cr_in_gap": root_path(root, config["legacy_inputs"]["cr_in_gap_predictions"]),
    }
    for model_id in a0_ids:
        spec = model(config, model_id)
        for system_id, legacy_path in legacy_paths.items():
            manifest_path, all_manifests = latest_manifest(root, "fixed", model_id, system_id, "fixed_manifest.json")
            if manifest_path is None:
                results.append({"model_id": model_id, "system_id": system_id, "status": "MISSING_MANIFEST", "max_abs_profile_difference_eV": float("nan"), "attempt_count": 0})
                continue
            profile_path = manifest_path.parent / "image_profile.csv"
            rows = read_profile(profile_path) if profile_path.is_file() else []
            new = np.asarray([float(row["relative_energy_eV"]) for row in rows if row["relative_energy_eV"] != ""], dtype=float)
            old = old_relative(legacy_path, spec["old_model_name"])
            difference = float(np.max(np.abs(new - old))) if len(new) == len(old) and len(new) else float("inf")
            passed = bool(difference < config["protocol"]["m0_max_abs_profile_difference_eV"] and len(new) == config["protocol"]["fixed_images"])
            results.append({
                "model_id": model_id,
                "model_label": spec["label"],
                "legacy_model_name": spec["old_model_name"],
                "system_id": system_id,
                "status": "PASS" if passed else "FAIL",
                "max_abs_profile_difference_eV": difference,
                "new_barrier_eV": float(new.max()) if len(new) else float("nan"),
                "old_barrier_eV": float(old.max()) if len(old) else float("nan"),
                "attempt_count": len(all_manifests),
                "new_profile_path": str(profile_path),
                "old_profile_path": str(legacy_path),
            })
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_profile_table(root: Path, config: dict, metrics: list[dict]) -> None:
    rows = []
    for metric in metrics:
        profile_path = Path(metric.get("profile_path", ""))
        if not profile_path.is_file():
            continue
        for row in read_profile(profile_path):
            merged = dict(row)
            merged.update({key: metric.get(key, "") for key in ("model_label", "model_family", "scratch_subfamily", "training_seed", "seed_semantics", "model_sha256", "run_status", "publication_status")})
            rows.append(merged)
    write_csv(root / "results/tables/fixed_profiles.csv", rows)


def family_summary(config: dict, metrics: list[dict]) -> list[dict]:
    groups = {}
    for metric in metrics:
        if metric.get("run_status") != "PASS" or not metric.get("finite_profile"):
            continue
        key = (metric["system_id"], metric["model_family"], metric.get("scratch_subfamily", "none"))
        groups.setdefault(key, []).append(metric)
    rows = []
    for (system_id, family, subfamily), items in sorted(groups.items()):
        if family == "foundation":
            for item in items:
                for measure in ("max_relative_energy_eV", "endpoint_drift_eV", "rmse_dft_eV", "mae_dft_eV", "spearman_dft"):
                    rows.append({
                        "system_id": system_id,
                        "model_family": family,
                        "scratch_subfamily": subfamily,
                        "model_id": item["model_id"],
                        "metric": measure,
                        "summary_type": "individual_checkpoint",
                        "n": 1,
                        "mean": float(item[measure]),
                        "std": "",
                        "min": float(item[measure]),
                        "max": float(item[measure]),
                        "seed_semantics": "checkpoint_variant_not_seed",
                    })
            continue
        for measure in ("max_relative_energy_eV", "endpoint_drift_eV", "rmse_dft_eV", "mae_dft_eV", "spearman_dft"):
            values = np.asarray([float(item[measure]) for item in items], dtype=float)
            rows.append({
                "system_id": system_id,
                "model_family": family,
                "scratch_subfamily": subfamily,
                "model_id": "",
                "metric": measure,
                "summary_type": "training_seed_mean_sd",
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "seed_semantics": "checkpoint_variant_not_seed" if family == "foundation" else "registered_training_seed",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["a0", "fixed", "all"], default="all")
    args = parser.parse_args()
    config, root = load_config(args.config)
    current_ids = [mid for mid, item in config["models"].items() if item["role"].startswith(("A1_", "A2_", "A3_", "A4_"))]
    a0_ids = [mid for mid, item in config["models"].items() if item["role"] == "A0_pipeline_self_check"]
    validation_path = root / "manifests/M0_INPUT_VALIDATION.json"
    validation = json.loads(validation_path.read_text()) if validation_path.is_file() else {"status": "MISSING"}
    a0_results = collect_a0(root, config, a0_ids)
    write_tsv(root / "results/tables/A0_SELF_CHECK.tsv", a0_results, list(a0_results[0]) if a0_results else ["model_id", "system_id", "status"])
    a0_pass = bool(a0_results) and all(row["status"] == "PASS" for row in a0_results)
    metrics, attempts = collect_stage_rows(root, config, "fixed", current_ids, "image_profile.csv", "fixed_manifest.json")
    write_csv(root / "results/tables/fixed_metrics_per_model.csv", metrics)
    write_profile_table(root, config, metrics)
    summary = family_summary(config, metrics)
    write_csv(root / "results/tables/fixed_family_summary.csv", summary)
    write_tsv(root / "manifests/ATTEMPT_AUDIT.tsv", attempts, ["stage", "model_id", "system_id", "manifest_path", "selected"])
    expected_rows = []
    for label, stage, model_ids, expected_count in (
        ("A0 pipeline self-check", "fixed", a0_ids, len(a0_ids) * len(config["systems"])),
        ("Tier 1 current matrix", "fixed", current_ids, len(current_ids) * len(config["systems"])),
        ("Tier 2 dense matrix", "dense", current_ids, len(current_ids) * len(config["systems"])),
    ):
        actual = 0
        actual_pass = 0
        manifest_name = "fixed_manifest.json" if stage == "fixed" else "dense_manifest.json"
        for model_id in model_ids:
            for system_id in config["systems"]:
                manifest_path, _ = latest_manifest(root, stage, model_id, system_id, manifest_name)
                if manifest_path is not None:
                    actual += 1
                    if json.loads(manifest_path.read_text()).get("status") == "PASS":
                        actual_pass += 1
        expected_rows.append({"stage": stage, "scope": label, "expected_rows": expected_count, "actual_manifests": actual, "actual_pass": actual_pass, "status": "PASS" if actual_pass == expected_count else "INCOMPLETE"})
    write_tsv(root / "results/tables/EXPECTED_vs_ACTUAL.tsv", expected_rows, ["stage", "scope", "expected_rows", "actual_manifests", "actual_pass", "status"])
    fixed_complete = len(metrics) == len(current_ids) * len(config["systems"]) and all(item.get("run_status") == "PASS" and item.get("finite_profile") and item.get("model_hash_match") for item in metrics)
    dense_complete = all(row["status"] == "PASS" for row in expected_rows if row["scope"] == "Tier 2 dense matrix")
    gate = {
        "campaign_id": config["campaign_id"],
        "m0_status": validation.get("status"),
        "a0_status": "PASS" if a0_pass else "FAIL",
        "fixed_status": "PASS" if fixed_complete else "INCOMPLETE",
        "dense_status": "PASS" if dense_complete else "INCOMPLETE",
        "stage_requested": args.stage,
        "attempt_duplicates": sum(1 for row in attempts if not row["selected"]),
    }
    if args.stage in ("fixed", "all") and not a0_pass:
        gate["blocking_reason"] = "A0 pipeline self-check did not pass"
    if args.stage in ("fixed", "all") and not fixed_complete:
        gate["blocking_reason"] = gate.get("blocking_reason", "Tier 1 current matrix incomplete or failed")
    if args.stage == "all" and not dense_complete:
        gate["blocking_reason"] = gate.get("blocking_reason", "Tier 2 dense matrix incomplete")
    write_json(root / "manifests/STAGE1_GATE.json", gate)
    print(json.dumps(gate, indent=2))
    if args.stage == "a0" and not a0_pass:
        raise SystemExit(2)
    if args.stage in ("fixed", "all") and (not a0_pass or not fixed_complete):
        raise SystemExit(3)
    if args.stage == "all" and not dense_complete:
        raise SystemExit(4)


if __name__ == "__main__":
    main()

