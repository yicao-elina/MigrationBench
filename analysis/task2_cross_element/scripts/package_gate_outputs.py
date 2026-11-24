#!/usr/bin/env python3
"""Package explicit model/DFT gate outcomes and empty scientific table schemas."""

from __future__ import annotations

import csv
import json
import platform
import os
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MIGRATIONBENCH_MODEL_DIR", os.environ.get("MIGRATIONBENCH_MODEL_ARCHIVE", "/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_git_large_assets_20261124/models")))
REMOTE_MODELS = {
    "foundation": str(MODEL_DIR / "Foundation-MACE-matpes-pbe-omat-ft.model"),
    "cr_ft": str(MODEL_DIR / "FT-600K.model"),
    "cr_scratch": str(MODEL_DIR / "Scratch_Multi_T.model"),
    "multi_element_ft": str(MODEL_DIR / "FT_Multi_T.model"),
}
SEEDS = {"foundation": 0, "cr_ft": 123, "cr_scratch": 123, "multi_element_ft": 123}
FAMILIES = {"foundation": "Foundation", "cr_ft": "Cr-specific FT", "cr_scratch": "Cr-specific Scratch", "multi_element_ft": "Multi-element FT"}
TRAINING = {
    "foundation": {"values": ["broad/foundation"], "status": "roster_not_present_in_local_manifest"},
    "cr_ft": {"values": ["Cr", "Sb", "Te"], "status": "declared_by_Cr-specific_inventory"},
    "cr_scratch": {"values": ["Cr", "Sb", "Te"], "status": "declared_by_Cr-specific_inventory"},
    "multi_element_ft": {"values": ["multi-element"], "status": "exact_roster_not_present_in_local_manifest"},
}


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    smoke = json.loads((ROOT / "smoke/model_element_support_and_profile.json").read_text())
    smoke_by_id = {r["model_id"]: r for r in smoke["records"]}
    records = []
    for model_id, family in FAMILIES.items():
        s = smoke_by_id[model_id]
        records.append({
            "model_id": model_id,
            "model_label": s["model_label"],
            "family": family,
            "training_elements": TRAINING[model_id]["values"],
            "training_elements_status": TRAINING[model_id]["status"],
            "seed": SEEDS[model_id],
            "remote_model_path": REMOTE_MODELS[model_id],
            "local_model_path": s["model_path"],
            "model_sha256": s["model_sha256"],
            "environment": {"python": platform.python_version(), "torch": s["software"]["torch"], "device": s["device"], "mace_calculator": "mace.calculators.MACECalculator"},
            "load_status": s["load_status"],
            "model_supported_elements": s.get("model_supported_elements", []),
            "element_support_status": {e: x["status"] for e, x in s["element_support"].items()},
            "profile_smoke_status": s["representative_path"].get("profile_status"),
            "scientific_success": False,
            "failure_reason": s.get("failure_reason") or ("DFT gate failed; model smoke is diagnostic only" if s["representative_path"].get("profile_status") == "complete" else "health gate incomplete"),
        })
    out = ROOT / "manifests/model_manifest.json"
    out.write_text(json.dumps({"schema_version": "task2-model-manifest-1.0", "created_utc": now, "namespace": "task2_cross_element", "records": records}, indent=2) + "\n")

    table_schemas = {
        "table_1_dft_quality.csv": ["path_id", "element", "dft_scientific_success", "explicit_neb_convergence", "all_image_energy_force_finite", "image_count_ok", "endpoint_identity_ok", "cell_pbc_ok", "failure_reason"],
        "table_2_success.csv": ["model_id", "element", "task_type", "scientific_success", "failure_reason", "input_sha256", "model_sha256"],
        "table_3_barrier_force.csv": ["model_id", "element", "path_id", "barrier_eV", "barrier_error_eV", "force_rmse_eV_A", "force_max_eV_A", "scientific_success", "failure_reason"],
        "table_4_transfer_gain.csv": ["element", "metric", "foundation_error", "ft_or_scratch_error", "transfer_gain", "training_elements_evidence", "embedding_evidence", "distribution_evidence", "interpretation_status"],
        "table_5_basin_seed_uncertainty.csv": ["element", "model_family", "seed", "path_id", "basin_id", "seed_spread", "agreement_status", "scientific_success", "failure_reason"],
    }
    schema_dir = ROOT / "results/table_schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for name, fields in table_schemas.items():
        with (schema_dir / name).open("w", newline="") as fh:
            csv.writer(fh).writerow(fields)
    (schema_dir / "README.md").write_text("# Task 2 table schemas\n\nThese files intentionally contain headers only. The DFT gate accepted 0/10 Cu/Li/Na paths, so no barrier, force, transfer-gain, basin, or seed values are scientifically admissible yet.\n")
    (ROOT / "results/figure_schema.json").write_text(json.dumps({"schema_version": "task2-figure-schema-1.0", "created_utc": now, "status": "placeholder_only_no_dft_gated_paths", "figures": ["model_element_error_heatmap", "foundation_vs_cr_ft_vs_scratch", "barrier_grouped_boxplot", "force_violin", "transfer_gain_matrix", "seed_spread", "basin_agreement"], "required_inputs": {"dft_quality": "results/table_schemas/table_1_dft_quality.csv", "success": "results/table_schemas/table_2_success.csv", "barrier_force": "results/table_schemas/table_3_barrier_force.csv", "transfer_gain": "results/table_schemas/table_4_transfer_gain.csv", "basin_seed": "results/table_schemas/table_5_basin_seed_uncertainty.csv"}, "rule": "Do not render scientific values until at least one path passes explicit DFT convergence, finite all-image energy/force, image/identity/cell/PBC, and retry audit gates."}, indent=2) + "\n")
    provisional = json.loads((ROOT / "results/provisional_neb_iteration_audit.json").read_text()) if (ROOT / "results/provisional_neb_iteration_audit.json").exists() else None
    report = f"""# Task 2 health-gate report: Cr-trained / Scratch cross-element generalization

Created: {now}\nNamespace: `task2_cross_element`\n
## Gate result

The read-only audit found 10 NEB directories: 4 Cu, 3 Li, and 3 Na. All have 10 images, 81 atoms, matching endpoint element order, and the same periodic cell. However, all 10 outputs contain `WARNING: scf convergence NOT achieved` for at least one image. Therefore DFT scientific success is **0/10**; `JOB DONE` is retained as runtime evidence only.

No cross-element error, barrier, force, basin, seed, or transfer-gain claim is released. Table files are header-only schemas, and figures remain placeholder-only by design.

## Provisional non-equilibrium snapshot run

At the user's requested exploratory level, the archived NEB coordinates were also treated as non-equilibrium snapshots. The `sb2te3.path` coordinates were converted from bohr to Å before MACE evaluation. Foundation-MACE completed 300 finite single-point evaluations: 10 paths × 10 images × 3 variants (`base`, `+0.20 Å x`, and `[-0.10, +0.14, +0.10] Å` on the dopant). All 300 rows remain `scientific_success=false` because the DFT source gate failed. The `<0.1 eV` last-two-completed-image-table screen passes for 8/10 base paths; it does not override SCF incompleteness or large residual image force errors.

The exact outer-iteration and last-two-table audit is in `results/provisional_neb_iteration_audit.csv`; Foundation profiles are in `results/provisional_mace_profiles.csv` and summarized in `results/provisional_mace_summary.csv`.

Electronic SCF `iteration #` counts were recovered from retained per-image `PW.out` files for 6/10 paths and are summarized in `results/pw_scf_iteration_inventory.csv`. For the other 4 paths those per-image files are absent; the outer NEB log still records the number of image-level self-consistency calls in the final block, but it does not expose the internal electronic iteration count.

## Model health gate

Foundation loaded on CPU and evaluated finite energy/forces for Cr/Cu/Li/Na/Sb/Te; its Cu diagnostic path profile completed for all 10 images. The Cr-specific FT model loaded and exposes Cr/Sb/Te, but its compiled forward schema is incompatible with the installed MACE calculator (`compute_edge_forces` keyword); Cu/Li/Na are element-unsupported by model metadata. Cr-specific Scratch and Multi-element FT remain load-blocked by a CUDA-backend serialization/runtime error in the current local environment. These are environment/model-interface statuses, not scientific performance failures.

## Evidence boundary

The Foundation profile is a diagnostic smoke only because its source DFT path failed the DFT gate. Relative energies are therefore not released as a model-comparison result. Transfer Gain and overfitting/degradation language are intentionally withheld until DFT-gated, identity-matched metrics plus training-element, embedding, and distribution evidence exist.
"""
    (ROOT / "HEALTH_GATE_REPORT.md").write_text(report)
    print(json.dumps({"model_manifest": str(out), "table_schema_count": len(table_schemas), "dft_gated_paths": 0}, indent=2))


if __name__ == "__main__":
    main()
