# Completeness matrix — reviewer point → status

Status codes: DONE-local / PENDING-cluster (+ JB ref).

| Reviewer point | Linked task | Status | Evidence |
|---|---|---|---|
| I6 (t-SNE separability robustness / descriptor-space check) | LP-2 | DONE-local | `scripts/latent_robustness.py`, `data_processed/latent_silhouette_all_spaces.csv`, `data_processed/latent_embedding_sensitivity.csv`, `tables/silhouette_descriptor_space.tex`, `figures/figS_latent_robustness.pdf`, `audit_reports/latent_robustness_note.md` |
| R4 (latent analysis reproducibility: seeds, block order, zero-stub) | LP-2 | DONE-local | same as above; block order verified vs `mace_probe_blackbox0818.py` L56-62; stub asserted all-zero then dropped; all seeds printed |
| R5 (embedding sensitivity: perplexity x seed, PHATE, trustworthiness) | LP-2 | DONE-local | `data_processed/latent_embedding_sensitivity.csv` + Fig. S-latent |

| R2 (MD transport uncertainty; analysis side) | LP-3 | DONE-local (Foundation/FT-MultiT verification PENDING-cluster: slurm logs came from truncated 41/140/162-frame trajectories — see note) | `scripts/md_transport_stats.py`, `data_processed/md_transport_metrics.csv`, `data_processed/md_transport_run_report.json`, `tables/md_transport_metrics.tex`, `figures/fig_md_msd_kappa_revised.pdf`, `audit_reports/md_reanalysis_note.md` |
| R2 (FT-600K naive-FT 2000-step run invalid) | LP-3 | DONE-local quarantine; rerun = JB-6 | all FT-600K transport values tagged `[INVALID-pending-JB-6]`; rationale: `MB/results/MD/2-naive-fine-tuning/mace_md_eval_0814.py:359-360` |

| I6 (SHAP surrogate validity + Fig 6 provenance) | LP-4 | DONE-local | `scripts/shap_audit.py`, `scripts/shap_fig6_regenerate.py`, `data_processed/shap_surrogate_r2.csv`, `data_processed/shap_surrogate_r2_revised.csv`, `tables/shap_surrogate_r2.tex`, `figures/fig6_shap_revised.pdf`, `audit_reports/shap_provenance_note.md`; Fig 6 provenance MD5-proven; in-sample R² debunked, CV R² 0.88-0.98 delivered |
| R6 (SHAP data side: value arrays / importance CSVs) | LP-4 | DONE-local | `data_processed/shap_top_features.csv` (published-label + corrected-label top-50 per model), `data_processed/shap_claimed_feature_check.csv`, `data_processed/shap_top_features_revised.csv`; published feature names shown mislabeled (386/390) and consensus/non-consensus claims unsupported |

| C1 (Fig 3d vs Fig S3 conflict; NEB PES asset audit, local side) | LP-6 | PARTIAL-local / PENDING-cluster (JB-5) | `scripts/lp6_neb_pes_audit.py`, `audit_reports/neb_pes_local_audit.md`, `data_processed/neb_pes_models_local.csv`, `data_processed/neb_selfneb_scratch_per_step.csv`, `data_processed/neb_pes_panel_1-4_digitized_approx.csv`, `figures/figS_neb_pes_local_audit.pdf`; C1 "+0.7 eV" reproduced by validated digitization (+0.71 eV); Scratch self-NEB shown diverged locally; Foundation 0.41 eV MEP is PNG-only locally |
