from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REMOTE = ROOT / "remote_results"
MANIFEST = pd.read_csv(ROOT / "candidate_manifest.csv")
RESULTS = pd.read_csv(REMOTE / "mace_single_point_results.csv")
FAIL = pd.read_csv(REMOTE / "failure_ledger.csv")
PROF = pd.read_csv("/Users/alina/Documents/Codex/2026-09-21/new-chat/work/mace_path_campaign/plots/mace_neb_profiles_36.csv")

for fn in ["mace_single_point_results.csv", "model_comparison_summary.csv", "runtime_manifest.json"]:
    shutil.copy2(REMOTE / fn, ROOT / fn)

# Candidate inventory is the manifest plus the immutable input hashes; keep it as a separate, readable table.
MANIFEST.to_csv(ROOT / "candidate_inventory.csv", index=False)

# Source coverage audit: missing final MACE image files are explicit omissions, never replaced by IDPP.
observed = set(MANIFEST.loc[MANIFEST.category.isin(["neb_intermediate", "near_saddle"]), "cov_id"].dropna().astype(str))
all_cov = sorted(PROF.cov_id.astype(str).unique())
cov_audit = pd.DataFrame({"cov_id": all_cov})
cov_audit["profile_rows"] = cov_audit.cov_id.map(PROF.cov_id.value_counts())
cov_audit["final_mace_image_available"] = cov_audit.cov_id.isin(observed)
cov_audit["coverage_status"] = cov_audit.final_mace_image_available.map({True: "candidate_source_available", False: "not_executed_missing_final_mace_image"})
cov_audit.to_csv(ROOT / "neb_source_coverage_audit.csv", index=False)

# Geometry/force comparison only. Absolute MACE energies are retained in the raw table but never differenced across models.
ok = RESULTS[RESULTS.status.eq("success")].copy()
summary = ok.groupby(["candidate_id", "category"], as_index=False).agg(
    n_models_success=("model_id", "nunique"),
    max_force_median_eV_A=("max_force_eV_A", "median"),
    max_force_max_eV_A=("max_force_eV_A", "max"),
    cr_z_spread_A=("cr_z_A", lambda x: float(x.max() - x.min())),
    coordination_min=("cr_coordination_3p2_A", "min"),
    coordination_max=("cr_coordination_3p2_A", "max"),
    min_distance_min_A=("min_interatomic_distance_A", "min"),
)
summary["model_anomaly"] = (summary.cr_z_spread_A > 0.25) | ((summary.coordination_max - summary.coordination_min) >= 2) | (summary.max_force_max_eV_A > 2.0)
summary["absolute_energy_comparison"] = "not_performed_model_zero_points_not_comparable"
summary.to_csv(ROOT / "model_comparison_summary.csv", index=False)

# Preserve the original runtime failures and add explicit not-executed records for absent sources and MD.
extra = []
for cov in cov_audit.loc[~cov_audit.final_mace_image_available, "cov_id"]:
    extra.append({"candidate_id": f"not_executed_{cov}", "category": "neb_intermediate", "model_label": "all_models", "failure_class": "not_executed_missing_final_mace_image", "runtime_error": "No local provenance-backed final MACE NEB image file; IDPP initialization was not substituted."})
extra.append({"candidate_id": "not_executed_md_missing", "category": "md_high_energy_or_large_displacement", "model_label": "all_models", "failure_class": "not_executed_md_missing", "runtime_error": "No auditable real MD trajectory/snapshot was available in the current workspace."})
extra_df = pd.DataFrame(extra)
FAIL = pd.concat([FAIL, extra_df], ignore_index=True, sort=False)
FAIL.to_csv(ROOT / "failure_ledger.csv", index=False)

counts = {"candidate_counts": MANIFEST.category.value_counts().to_dict(), "single_point_rows": int(len(RESULTS)), "single_point_success_rows": int((RESULTS.status == "success").sum()), "runtime_failure_rows": int((FAIL.failure_class == "runtime_failure").sum()), "not_executed_rows": int((FAIL.failure_class.astype(str).str.startswith("not_executed")).sum()), "md_missing": True, "neb_paths_total": int(len(all_cov)), "neb_paths_with_local_final_mace_images": int(cov_audit.final_mace_image_available.sum())}
(ROOT / "coverage_summary.json").write_text(json.dumps(counts, indent=2))

lines = [
    "# MLIP candidate energy-surface validation (2026-10-01)", "",
    "## Scope and evidence boundary", "",
    "This is an independent candidate single-point campaign. Existing endpoint relaxations and NEB campaigns were read-only inputs; no candidate was relaxed and no existing campaign directory was overwritten. Absolute MACE energies are retained per model for traceability but are not compared across model families because their zero points and energy conventions differ.", "",
    "## Coverage", "",
]
for cat, n in MANIFEST.category.value_counts().items(): lines.append(f"- **{cat}**: {n} candidates.")
lines += [f"- **NEB source coverage**: {int(cov_audit.final_mace_image_available.sum())}/{len(cov_audit)} paths had a local provenance-backed final MACE image; {int((~cov_audit.final_mace_image_available).sum())} were not executed and were not replaced by IDPP.", "- **MD high-energy/large-displacement**: `md_missing`; no auditable real MD snapshots were available.", "", "## Single-point result boundary", ""]
for model, g in RESULTS.groupby("model_label"):
    lines.append(f"- **{model}**: {int((g.status == 'success').sum())}/{len(g)} successful single points; failures remain in `failure_ledger.csv`.")
lines += ["", "Successful model comparisons use max force, Cr position spread, coordination range, and finite-geometry checks. `model_anomaly` is a screening flag, not a claim of model accuracy or OOD proof.", "", "## Scientific interpretation", "", "Stable endpoints are the most complete category: the FT-600K, Foundation, and MPA0 CPU-compatible weights evaluated all 24 stable candidates. The near-saddle and NEB-intermediate sets are intentionally single-point-only; their forces are diagnostic of local model response and must not be interpreted as relaxed barriers.", "", "The current CPU-compatible results show a strong regime distinction: stable endpoints have low FT-600K residual forces, while intermediates and near-saddle images have larger forces, as expected for non-stationary structures. Foundation and MPA0 also produce finite values but generally larger forces on these non-endpoint candidates. This is evidence for force-surface sensitivity, not evidence that either model is more accurate.", "", "The 13 historical/seed weights that failed during model deserialization are classified as runtime failures, with original tracebacks retained. They are not counted as scientific failures, OOD cases, or successful evaluations. GPU/QoS access is still required before claiming complete seed-ensemble coverage.", "", "## DFT single-point priority", "", "1. Near-saddle candidates from `cov_020`, `cov_035`, `cov_002`, `cov_027`, and `cov_031`, especially those marked `model_anomaly` or with high successful-model max force.", "2. NEB intermediates on the highest-barrier paths and all source-complete paths with force/coordination disagreement.", "3. Stable endpoints `unique_044`, `unique_065`, `unique_067`, `unique_080`, `unique_096`, `unique_097`, `unique_109`, and `unique_151`, preserving the prior consensus/disagreement rationale.", "", "## Reproduction", "", "The local manifest/build script is `build_candidates.py`; remote execution used `run_rockfish.slurm` with `run_single_points.py` under `/scratch16/pclancy3/yi/mace_candidate_eval_20261001`. The successful remote job was Rockfish job 31475643 (CPU-compatible subset); job 31475439 records the earlier full CPU attempt with the original legacy/seed weights and its runtime failures.", ""]
(ROOT / "publication_ready_analysis.md").write_text("\n".join(lines))
print(json.dumps(counts, indent=2))
