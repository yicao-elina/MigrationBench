# Project Status Overview

Date: 2026-09-13

## Current Project State

The revision is in a controlled partial state. The local audit infrastructure is much stronger than the original manuscript state, but the final manuscript still cannot use several key NEB and multi-seed values.

## Completed Or Strong Local Assets

| Area | Current state | Evidence |
|---|---|---|
| Claim audit | Major risky claims mapped to reviewer items and edit refs | `Revision1/audit_reports/claim_map.md` |
| Zero-force/data leakage | No evidence fabricated zero-force NEB frames entered reported FT-600K/FT-MultiT training | `Revision1/audit_reports/zero_force_verdict.md` |
| NEB path inventory | Rockfish roots identified; `/data` read-only, `/scratch16` for new jobs | `Revision1/audit_reports/cluster_inventory.md` |
| 1-4 DFT candidate | Historical 0.336050 eV is preserved only as an unconverged candidate (`0.482407 eV/A` maximum NEB error), not a final reference | `docs/current_neb_barrier_results_zh.md` |
| Fixed-path model errors | 1-4 fixed-path errors are reproducible; Foundation overestimates by about 0.69 eV | `Revision1/audit_reports/rockfish_neb_pipeline_audit.md` |
| Historical Fig. S3 | The 82-atom Foundation self-NEB mixed DFT endpoint and MACE internal-image energies; 0.41 eV is rejected | `docs/historical_figS3_foundation_audit.md` |
| 81-atom path classification | `neb_1`-`neb_5` are deep-penetration candidates by continuous descriptors | `Revision1/audit_reports/81_atom_path_classification.md` |
| Latent robustness | t-SNE/PHATE claims softened; original-space caveats documented | `Revision1/audit_reports/latent_robustness_note.md` |
| SHAP provenance | Local provenance and surrogate-R2 checks improved; mechanistic claims must be surrogate-scoped | `Revision1/audit_reports/shap_provenance_note.md` |
| Dataset schema | Five normalized audit tables validate; final rows require path, endpoint, calculator, and runtime provenance hashes | `configs/migrationbench_hf_schema.yaml` |
| Restartable NEB design | QE `max_seconds` and Slurm walltime chaining designed | `work/revision1_neb_restart_design/` |
| Runtime reproducibility | Future QE/MACE wrappers record package/module versions, input hashes, and scientific executable hashes; real Slurm smoke passed | `docs/environment_reproducibility.md` |
| Checkpoint-scoped leakage | 1,110 QE frames have zero registered SOAP/RMSD/exact matches to declared train/valid/test files; unknown Foundation pretraining remains unauditable | `docs/reviewer_response_A2_data_leakage.md` |

## Major Gaps

| Gap | Why it matters | Current blocker | Required closeout |
|---|---|---|---|
| G1: converged DFT NEB for 1-7 | Needed to repair Fig. 3d vs Fig. S3 protocol conflict and validate second in-gap path | Restartable long-walltime chain is running; latest accepted prefix remains unconverged | Continue only from monitor-classified clean parents until movable-image force and barrier-drift gates pass |
| G2: manuscript-grade deep-penetration DFT | Needed before claiming deep-penetration model ranking | Historical 1-2/1-3/81-atom paths are unconverged or high-error | Pick priority path, MACE precondition, QE refine to gate |
| G3: multi-seed retraining | Needed for reviewer A3 and Scratch-5% control A5 | Cluster jobs/results not finalized in local audit | Run grouped-split seeds {123,234,345}, aggregate mean +/- std |
| G4: train/test overlap SOAP/RMSD | Needed for reviewer A2 | Evidence complete for the declared fine-tuned checkpoint; Foundation pretraining corpus unavailable | Re-run automatically if frames, grouped splits, training files, or checkpoint change |
| G5: MD protocol correction | Needed for reviewer C9 | FT-600K rerun under corrected 100000-step protocol pending | Rerun and replace quarantined transport values |
| G6: SHAP perturbation test | Needed to support/soften feature-level interpretation | Cluster-side perturbation test pending | Run finite perturbation or explicitly limit claim to surrogate sensitivity |
| G7: final figure rebuild | Reviewer response depends on updated figures/tables | Several upstream values are placeholders | Rebuild only after upstream data gates pass |
| G8: QE calculator identity | Current SI says universal 100/400 Ry, 4x4x1, SOC, while 377 audited inputs form 12 identities and none explicitly enables SOC | Cutoff contrast passed; Gamma-versus-2x2x1 jobs are running; spin/SOC remain unresolved | Resolve k-point, spin, and SOC branches; then retain or recompute each result under one accepted identity |

## Acceptance Gates

| Output type | Gate |
|---|---|
| Manuscript numeric barrier | Raw artifact exists locally; parser re-derives it; DFT NEB is converged; path, endpoint, calculator, and runtime-provenance acceptance hashes all match |
| QE NEB production | Slurm `COMPLETED 0:0`; QE clean end or restart-compatible stop; `max_movable_image_error_eV_A <= 0.03`; barrier drift last iterations < 0.02 eV; calculator identity accepted by N24 |
| Geometry-transform speedup | Direct and transformed structures both converge to the same basin under the same N24-accepted calculator; report ionic-step and SCF-iteration ratios, not unconverged prefix ratios |
| MACE pre-NEB initializer | `mlff_neb_manifest.json`; `mlff_neb_images.extxyz`; seed and input SHA recorded; energy delta table present; no unphysical spike |
| Reviewer response text | Every placeholder either filled with audited value or explicitly stated as not included |
| Figure/table | Built from local `data_processed` output by script, not manually copied from screenshots |
| Dataset release | Each record has split/protocol/calculator/convergence/source hashes; new production rows also bind runtime provenance URI/SHA/environment identity |

## Immediate Next Critical Path

1. Let the active restartable `1-6`/`1-7` chains, endpoint A/B relaxations, and k-point jobs run without high-frequency polling.
2. Classify each NEB segment from raw output and continue only clean, hash-verified parents.
3. Resolve the cutoff gate, then run k-point and spin/SOC branches before accepting a production calculator identity.
4. Compare direct and transformed relaxations only after same-basin convergence; rerun both variants if the accepted calculator changes.
5. Only then update Fig. 3/SI and response A1 with final DFT-consistent values.
