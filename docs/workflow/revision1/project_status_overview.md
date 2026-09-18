# Project Status Overview

Date: 2026-09-09

## Current Project State

The revision is in a controlled partial state. The local audit infrastructure is much stronger than the original manuscript state, but the final manuscript still cannot use several key NEB and multi-seed values.

## Completed Or Strong Local Assets

| Area | Current state | Evidence |
|---|---|---|
| Claim audit | Major risky claims mapped to reviewer items and edit refs | `Revision1/audit_reports/claim_map.md` |
| Zero-force/data leakage | No evidence fabricated zero-force NEB frames entered reported FT-600K/FT-MultiT training | `Revision1/audit_reports/zero_force_verdict.md` |
| NEB path inventory | Rockfish roots identified; `/data` read-only, `/scratch16` for new jobs | `Revision1/audit_reports/cluster_inventory.md` |
| 1-4 DFT candidate | Current best in-gap candidate barrier is 0.336050 eV, but formal convergence caveat remains | `Revision1/audit_reports/dft_reference_note.md` |
| Fixed-path model errors | 1-4 fixed-path errors are reproducible; Foundation overestimates by about 0.69 eV | `Revision1/audit_reports/rockfish_neb_pipeline_audit.md` |
| 81-atom path classification | `neb_1`-`neb_5` are deep-penetration candidates by continuous descriptors | `Revision1/audit_reports/81_atom_path_classification.md` |
| Latent robustness | t-SNE/PHATE claims softened; original-space caveats documented | `Revision1/audit_reports/latent_robustness_note.md` |
| SHAP provenance | Local provenance and surrogate-R2 checks improved; mechanistic claims must be surrogate-scoped | `Revision1/audit_reports/shap_provenance_note.md` |
| Dataset schema | MigrationBench/Hugging Face style schema drafted | `Revision1/configs/migrationbench_hf_schema.yaml` |
| Restartable NEB design | QE `max_seconds` and Slurm walltime chaining designed | `work/revision1_neb_restart_design/` |

## Major Gaps

| Gap | Why it matters | Current blocker | Required closeout |
|---|---|---|---|
| G1: converged DFT NEB for 1-7 | Needed to repair Fig. 3d vs Fig. S3 protocol conflict and validate second in-gap path | QE preflights timed out after reaching image 3; long walltime job not yet submitted by Codex due SSH sandbox issue | Submit restartable 24-48 h QE chain from MACE-relaxed 1-7 |
| G2: manuscript-grade deep-penetration DFT | Needed before claiming deep-penetration model ranking | Historical 1-2/1-3/81-atom paths are unconverged or high-error | Pick priority path, MACE precondition, QE refine to gate |
| G3: multi-seed retraining | Needed for reviewer A3 and Scratch-5% control A5 | Cluster jobs/results not finalized in local audit | Run grouped-split seeds {123,234,345}, aggregate mean +/- std |
| G4: train/test overlap SOAP/RMSD | Needed for reviewer A2 | SOAP nearest-neighbor audit still placeholder | Compute min distance from in-gap NEB images to FT-600K training frames |
| G5: MD protocol correction | Needed for reviewer C9 | FT-600K rerun under corrected 100000-step protocol pending | Rerun and replace quarantined transport values |
| G6: SHAP perturbation test | Needed to support/soften feature-level interpretation | Cluster-side perturbation test pending | Run finite perturbation or explicitly limit claim to surrogate sensitivity |
| G7: final figure rebuild | Reviewer response depends on updated figures/tables | Several upstream values are placeholders | Rebuild only after upstream data gates pass |

## Acceptance Gates

| Output type | Gate |
|---|---|
| Manuscript numeric barrier | Raw artifact exists locally, path id recorded, parser re-derives value, status is `converged` or explicitly accepted as `numerically_stationary_unconverged` |
| QE NEB production | Slurm `COMPLETED 0:0`; QE clean end or restart-compatible stop; `max_image_error_eV_A <= 0.03`; barrier drift last iterations < 0.02 eV |
| MACE pre-NEB initializer | `mlff_neb_manifest.json`; `mlff_neb_images.extxyz`; seed and input SHA recorded; energy delta table present; no unphysical spike |
| Reviewer response text | Every placeholder either filled with audited value or explicitly stated as not included |
| Figure/table | Built from local `data_processed` output by script, not manually copied from screenshots |
| Dataset release | Each record has split/protocol/calculator/convergence/provenance SHA/source artifact path |

## Immediate Next Critical Path

1. Submit long restartable QE chain for `1-7` from MACE-relaxed images.
2. Parse the resulting `neb.out` after each segment.
3. Continue from previous run directory until convergence gate passes or the path is scientifically rejected.
4. Only then update Fig. 3/SI and response A1 with final DFT-consistent values.
