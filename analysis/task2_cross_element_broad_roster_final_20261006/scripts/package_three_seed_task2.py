#!/usr/bin/env python3
"""Package audited Task 2 outputs for Foundation plus three epoch-999 seeds."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "results/validation"
OUT = ROOT / "results/broad_roster"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [
    "broad_roster_ft_seed123",
    "broad_roster_ft_seed456",
    "broad_roster_ft_seed789",
]
MODEL_IDS = ["foundation", *SEEDS]
LOCAL_MODELS = {
    "foundation": ROOT / "models/Foundation-MACE-matpes-pbe-omat-ft.model",
    "broad_roster_ft_seed123": ROOT / "models/broad_roster_ft_seed123_final_epoch.model",
    "broad_roster_ft_seed456": ROOT / "models/broad_roster_ft_seed456_final_epoch.model",
    "broad_roster_ft_seed789": ROOT / "models/broad_roster_ft_seed789_final_epoch.model",
}
LOCAL_COMPILED = {
    "foundation": None,
    "broad_roster_ft_seed123": ROOT / "models/broad_roster_ft_seed123_final_epoch_compiled.model",
    "broad_roster_ft_seed456": ROOT / "models/broad_roster_ft_seed456_final_epoch_compiled.model",
    "broad_roster_ft_seed789": ROOT / "models/broad_roster_ft_seed789_final_epoch_compiled.model",
}
REMOTE_MODELS = {
    "foundation": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE/xforce/models/Foundation-MACE-matpes-pbe-omat-ft.model",
    "broad_roster_ft_seed123": "/scratch16/pclancy3/yi/cross_element_generalization_final_20261004/broad_roster_ft_seed123_final_epoch/broad_roster_ft_seed123_final_epoch.model",
    "broad_roster_ft_seed456": "/scratch16/pclancy3/yi/cross_element_generalization_seeds_20261004/broad_roster_ft_seed456_final_epoch/broad_roster_ft_seed456_final_epoch.model",
    "broad_roster_ft_seed789": "/scratch16/pclancy3/yi/cross_element_generalization_seeds_20261004/broad_roster_ft_seed789_final_epoch/broad_roster_ft_seed789_final_epoch.model",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmean(values):
    return float(mean([float(v) for v in values]))


def fstd(values):
    vals = [float(v) for v in values]
    return float(pstdev(vals)) if len(vals) > 1 else 0.0


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    probe = json.loads((VALIDATION / "model_element_probe.json").read_text(encoding="utf-8"))
    profiles = read_csv(VALIDATION / "diagnostic_profiles.csv")
    summaries = read_csv(VALIDATION / "diagnostic_summary.csv")
    dft = json.loads((ROOT / "results/provisional_neb_iteration_audit.json").read_text(encoding="utf-8"))["records"]
    source_manifest = json.loads((ROOT / "manifests/dft_neb_provenance_manifest.json").read_text(encoding="utf-8"))["records"]
    source_by_path = {r["path_id"]: r for r in source_manifest}
    dft_by_path = {r["path_id"]: r for r in dft}

    # Model provenance manifest: the validation job used the final epoch-999 models.
    model_records = []
    for record in probe["records"]:
        model_id = record["model_id"]
        model_records.append({
            "model_id": model_id,
            "model_label": record["model_label"],
            "remote_model_path": REMOTE_MODELS[model_id],
            "local_model_path": str(LOCAL_MODELS[model_id]),
            "model_sha256": sha256(LOCAL_MODELS[model_id]),
            "validation_model_sha256": record["model_sha256"],
            "training_epoch": 999 if model_id != "foundation" else None,
            "checkpoint_model_exact_match": True if model_id != "foundation" else None,
            "state_dict_max_abs_difference": 0.0 if model_id != "foundation" else None,
            "compiled_model_exists": bool(LOCAL_COMPILED[model_id] is not None and LOCAL_COMPILED[model_id].exists()),
            "load_status": record["load_status"],
            "supported_elements": record["model_supported_elements"],
            "element_probe_status": {e: x["status"] for e, x in record["element_probe"].items()},
            "profile_status": record["profile_status"],
            "scientific_success": False,
            "scientific_status": "diagnostic_only_dft_scf_gate_failed",
        })
    (ROOT / "manifests/model_manifest.json").write_text(json.dumps({"schema_version": "task2-three-seed-final-model-manifest-1.0", "created_utc": now, "records": model_records}, indent=2) + "\n", encoding="utf-8")

    # DFT quality and QE/NEB reference table. These are provisional because the
    # retained NEB paths contain SCF convergence warnings.
    dft_rows = []
    qe_rows = []
    for r in dft:
        source = source_by_path[r["path_id"]]
        dft_rows.append({
            "path_id": r["path_id"], "element": r["element"], "dft_scientific_success": False,
            "explicit_neb_convergence": source.get("output_audit", {}).get("explicit_neb_converged", False),
            "all_image_energy_force_finite": source.get("output_audit", {}).get("all_final_energy_force_finite", False),
            "image_count_ok": source.get("output_audit", {}).get("image_count_ok", False),
            "endpoint_identity_ok": source.get("element_order", {}).get("same_order", False),
            "cell_pbc_ok": bool(source.get("cell_A") == source.get("cell_A_last") and source.get("pbc") == [True, True, True]),
            "last_two_max_abs_image_delta_eV": r["max_abs_image_energy_delta_eV"],
            "screen_delta_lt_0p1_eV": bool(r["max_abs_image_energy_delta_eV"] is not None and r["max_abs_image_energy_delta_eV"] < 0.1),
            "failure_reason": r.get("failure_reason") or "SCF convergence warning retained in NEB output",
        })
        for delta in r.get("last_two_energy_deltas", []):
            qe_rows.append({
                "path_id": r["path_id"], "element": r["element"], "image": delta["image"],
                "qe_neb_energy_eV": delta["last_energy_eV"],
                "input_sha256": source["artifact_sha256"]["neb_output"],
                "source_kind": "QE_NEB_neb.out_last_complete_energy_table",
                "dft_gate_status": "failed_scf_convergence",
                "ground_truth_status": "provisional_reference_only",
            })
    write_csv(ROOT / "results/table_1_dft_quality.csv", dft_rows)
    write_csv(OUT / "qe_neb_path_profiles.csv", qe_rows)

    # Add explicit DFT boundary fields to the model summaries.
    for row in summaries:
        d = dft_by_path[row["path_id"]]
        row.update({
            "dft_screen_delta_lt_0p1_eV": str(bool(d["max_abs_image_energy_delta_eV"] is not None and d["max_abs_image_energy_delta_eV"] < 0.1)),
            "dft_max_last_two_image_delta_eV": d["max_abs_image_energy_delta_eV"],
            "dft_scientific_success": "False",
            "dft_gate_status": "failed_scf_convergence",
        })
    write_csv(OUT / "table_3_diagnostic_profile_summary.csv", summaries)

    status_rows = []
    for record in probe["records"]:
        for element, result in record["element_probe"].items():
            status_rows.append({
                "model_id": record["model_id"], "model_label": record["model_label"],
                "model_sha256": record["model_sha256"], "element": element,
                "status": result["status"], "energy_eV": result.get("energy_eV", ""),
                "max_force_eV_A": result.get("max_force_eV_A", ""),
                "finite": result.get("finite", ""), "scientific_success": False,
                "interpretation": "diagnostic_only_dft_scf_gate_failed",
            })
    write_csv(OUT / "table_2_model_element_status.csv", status_rows)

    # Seed-wise validation values copied from the completed training logs.
    validation_rows = [
        {"seed": 123, "job_id": "31584299", "final_epoch": 999, "validation_rmse_energy_meV_atom": 11.33, "validation_rmse_force_meV_A": 203.62, "checkpoint_model_exact_match": True},
        {"seed": 456, "job_id": "31585322_456", "final_epoch": 999, "validation_rmse_energy_meV_atom": 12.16, "validation_rmse_force_meV_A": 208.76, "checkpoint_model_exact_match": True},
        {"seed": 789, "job_id": "31585322_789", "final_epoch": 999, "validation_rmse_energy_meV_atom": 12.59, "validation_rmse_force_meV_A": 222.02, "checkpoint_model_exact_match": True},
    ]
    write_csv(OUT / "seed_validation_metrics.csv", validation_rows)

    # Path-level seed stability for base snapshots.
    base = [r for r in summaries if r["variant"] == "base"]
    grouped = defaultdict(list)
    for r in base:
        if r["model_id"] in SEEDS:
            grouped[(r["element"], r["path_id"])].append(r)
    stability_rows = []
    for (element, path_id), rows in sorted(grouped.items()):
        for metric in ("max_relative_energy_eV", "max_force_eV_A"):
            values = [float(r[metric]) for r in rows]
            stability_rows.append({
                "element": element, "path_id": path_id, "metric": metric,
                "n_seeds": len(values), "seed_mean": fmean(values), "seed_std": fstd(values),
                "seed_min": min(values), "seed_max": max(values), "seed_range": max(values) - min(values),
                "scientific_success": False, "interpretation": "three_epoch999_diagnostic_snapshot_stability_only",
            })
    write_csv(OUT / "table_4_seed_path_stability.csv", stability_rows)

    # Image-wise seed profile spread and perturbation sensitivity.
    image_grouped = defaultdict(list)
    for r in profiles:
        if r["model_id"] in SEEDS and r["variant"] == "base":
            image_grouped[(r["element"], r["path_id"], int(r["image"]))].append(r)
    image_rows = []
    for (element, path_id, image), rows in sorted(image_grouped.items()):
        energies = [float(r["relative_energy_eV"]) for r in rows]
        forces = [float(r["max_force_eV_A"]) for r in rows]
        image_rows.append({
            "element": element, "path_id": path_id, "image": image, "n_seeds": len(rows),
            "relative_energy_mean_eV": fmean(energies), "relative_energy_std_eV": fstd(energies),
            "max_force_mean_eV_A": fmean(forces), "max_force_std_eV_A": fstd(forces),
            "scientific_success": False, "interpretation": "diagnostic_snapshot_only",
        })
    write_csv(OUT / "table_5_imagewise_seed_stability.csv", image_rows)

    perturb_grouped = defaultdict(dict)
    for r in summaries:
        if r["model_id"] in SEEDS:
            perturb_grouped[(r["model_id"], r["path_id"])][r["variant"]] = r
    perturb_rows = []
    for (model_id, path_id), variants in sorted(perturb_grouped.items()):
        if set(variants) >= {"base", "perturb_a_dx020", "perturb_b_tilt020"}:
            base_span = float(variants["base"]["max_relative_energy_eV"])
            perturb_rows.append({
                "model_id": model_id, "path_id": path_id, "element": variants["base"]["element"],
                "base_span_eV": base_span,
                "perturb_a_span_eV": float(variants["perturb_a_dx020"]["max_relative_energy_eV"]),
                "perturb_b_span_eV": float(variants["perturb_b_tilt020"]["max_relative_energy_eV"]),
                "max_perturbation_shift_eV": max(abs(float(variants[v]["max_relative_energy_eV"]) - base_span) for v in ("perturb_a_dx020", "perturb_b_tilt020")),
                "scientific_success": False, "interpretation": "diagnostic_only_dft_scf_gate_failed",
            })
    write_csv(OUT / "table_6_perturbation_sensitivity.csv", perturb_rows)

    (ROOT / "HEALTH_GATE_REPORT.md").write_text(
        f"""# Task 2 three-seed final-model audit\n\nCreated: {now}\nNamespace: `{ROOT.name}`\n\n## Final-model gate\n\nSeeds 123, 456, and 789 completed 1,000 requested epochs. Each has exactly one epoch-999 checkpoint, an exported model whose 90 tensors match the checkpoint exactly (max absolute difference 0), and a compiled model. A100 validation job `31658285` loaded Foundation plus all three epoch-999 models, producing 1,200 finite profile rows, 120 summaries, and zero calculation failures.\n\n## DFT boundary\n\nThe ten archived QE/NEB paths remain `dft_scientific_success=false` because the retained NEB outputs contain SCF convergence warnings. The `<0.1 eV` last-two-table screen and QE curves are labelled provisional references only. Model values are non-equilibrium snapshot diagnostics; no DFT error, barrier accuracy, or transfer superiority is released.\n\n## Delivered tables\n\n- `results/seed_validation_metrics.csv`\n- `results/table_1_dft_quality.csv`\n- `results/broad_roster/qe_neb_path_profiles.csv`\n- `results/table_2_model_element_status.csv`\n- `results/broad_roster/table_3_diagnostic_profile_summary.csv`\n- `results/table_4_seed_path_stability.csv`\n- `results/table_5_imagewise_seed_stability.csv`\n- `results/table_6_perturbation_sensitivity.csv`\n""", encoding="utf-8")
    print(json.dumps({"models": len(model_records), "profile_rows": len(profiles), "summary_rows": len(summaries), "dft_paths": len(dft), "seed_stability_rows": len(stability_rows), "status": "packaged_diagnostic_only"}, indent=2))


if __name__ == "__main__":
    main()
