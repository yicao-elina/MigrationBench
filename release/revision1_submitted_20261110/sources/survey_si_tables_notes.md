# SI figures + SI tables + response tables — provenance notes (read-only audit, 2026-11-10)

Submitted source: codex/revision1-camera-ready-evidence@334260c (now also in integrate/final-paper-20261106 HEAD 991a419).
Paths in the TSV are repo-relative to the integrate checkout unless absolute. C = /Users/alina/Documents/ChatGPT/Wiley-MigrationBench.
Submitted asset hashes: prov/submitted_hashes.txt.

## Number mapping (confirmed from compiled SI order = submission captions)
- S1 fig:training_design (fig_training_design_routes.pdf) · S2 fig:grouped_overlap (grouped_retraining_overlap.pdf)
- S3 fig:other_dopants (fig_NEB-fixed-path_Cu-Li-Na-Cr.pdf) · S4 fig:sliding_all (fig_NEB_interlayer-sliding.pdf)
- S5 fig:current_feature_audit (figS_current_feature_audit.pdf)
- Tables S1 current_pointwise · S2 a2_grouped_audit_summary (inline) · S3 profile_coverage (inline) · S4 current_neb_errors
  · S5 dft_neb_barriers · S6 path_registry · S7 scratch5_path_resolved_descriptive · S8 current_md_horizons
  · S9 current_original_space · S10 current_tsne_sensitivity · S11 current_shap_fidelity
- Response tables R1–R11: all inline, hand-typed in response_to_reviewers.tex (labels tab:resp-*).

## Status summary
- FULL: S2-fig (after import), S3-fig (after import), S5-fig (after import); Tables S1, S2, S4, S6, S7, S8, S9, S10, S11; R1–R11.
- PARTIAL: Fig. S1 (newer .drawio only in ChatGPT folder; export manual), Fig. S4 (final PDF is a hand-edited cairo export of CANVA-P09; intermediate not found),
  Table S3 (matches R02 coverage audit, not newer R05), Table S5 (legacy 61-atom neb.out only in /Users/alina/Project/26MIgrationBench).
- MISSING: none in the sense of "no data"; every number spot-checked against a backing CSV/TSV/JSON in the repo.

## Main finding: the central table generator is outside the repo
`C/output/revision1_camera_ready/audit_current_figures.py` writes current_pointwise / current_md_horizons / current_neb_errors /
current_shap_fidelity / current_original_space / current_tsne_sensitivity .tex, fig5_family_statistics.csv, current_analysis_audit.json,
source_ledger.json AND renders figS_current_feature_audit.pdf. It is listed in release/revision1_camera_ready/DELIVERY_MANIFEST.json
but the file is not present in the repo release dir. It hardcodes ROOT=C and BACKUP=~/Downloads/MigrationBench_ALL_CANVA_and_publication_figures_backup_20261102.
Also outside repo: finish_package.py, apply_to_original.py, replacement_manifest.json (same dir).

## Hand-edit layer (Overleaf, commit 334260c)
Generators emit 2–3 decimals; the submitted tables were rounded/reformatted by hand (e.g. 40.68±4.88 → 41±5; 0.0627 → 6.3×10⁻²).
Other manual edits: dft_neb_barriers.tex dropped the Status (UNCONVERGED/CONVERGED) column; path_registry.tex header Audit ID→Path ID;
scratch5 table caption reworded (generated sha e41666.. vs submitted 9e2a43a8). Regenerating will not byte-match; values do.
Recommendation: add a `round_for_manuscript` step or freeze the submitted .tex with a values-check script.

## Inconsistencies found (worth fixing before next revision)
1. Fig. S2 caption ≠ figure. Caption describes legacy G1 panel (FT-600K/Scratch/Scratch-5 %, single MPA0, dashed 0.16 eV);
   the submitted PDF panel a is "Canonical in-gap fixed path" with 4 trained groups × 3 seeds + 3 foundation checkpoints.
   In-repo paper/revision1/scripts/plot_grouped_retraining_overlap.py is stale (renders the old G1 version).
2. Table S3 counts follow cr_three_path_coverage_audit_R02.tsv (scratch_legacy column). R05 (also in repo, used for plotted_profiles_R05)
   has scratch_600k / scratch_multit with different counts (self in-gap scratch_600k = 2). Check which release main Fig. 4 shows.
3. Table S8 says "Target reached" for foundation models; MD15_STATUS_AUDIT_R02 marks them TARGET_REACHED_PROVISIONAL.
   R7 uses "Scientific stop" where S8 uses "Threshold reached" (same runs).
4. Response A4 / R4 cites SI-sec:endpoint_perturbation_campaign, but the SI no longer \input s the appendix (line 263 commented);
   resolves only via the stale cached sn-article-SI.aux.
5. source_ledger.json lists an older sha (dcfb7f45) for fig_NEB-fixed-path_Cu-Li-Na-Cr.pdf; submitted is 72263413.
6. shap_reproduction_check.json (R11) — the comparison script that produced it was not found in repo or C/output/revision1_camera_ready.

## Files to import into the repo (abs path, size)
- C/output/revision1_camera_ready/audit_current_figures.py (12K) — required (S1,S4,S5-fig,S8–S11 tex writer)
- C/output/revision1_camera_ready/{finish_package.py 8K, apply_to_original.py 4K, replacement_manifest.json 8K} — packaging lineage
- C/analysis/grouped_retraining_overlap_palette_20261109/ (320K) — Fig. S2 generator, hash-matched
- C/analysis/four_dopant_fixed_path_palette_20261109/ (1.7M) — Fig. S3 generator + source TSVs, hash-matched
- C/analysis/task2_cross_element_broad_roster_final_20261006/results/broad_roster/qe_neb_path_profiles.csv (20K) — S3 QE upstream
- C/output/revision1_camera_ready/paper/revision1/figures/main/SI/method-P02_palette_updated_pre_vector_export.pdf.drawio (36K) — Fig. S1 source (newer than repo)
- C/analysis/migrationbench_training_design_figure_20261024/ (264K) — Fig. S1 drafts (optional)
- /Users/alina/Project/26MIgrationBench/data_raw/historical_neb/legacy_61/ (2.4M) — S5/S6/S7/R1/R3 legacy 61-atom QE neb.out (G2,Q1–Q3,M1,L1)
- C/analysis/sliding_four_model_campaign_20261024/ (1.1M) — S4 upstream not mirrored in repo source_archive
- Optional large upstream (keep as pointers or archive): C/analysis/rmse_train_test_20261026 (91M; S1/R2 per-frame),
  C/analysis/md_1ns_updated_publication_figure_20261025 (24M; S8), C/analysis/shap_full_campaign_20261101 (11M; S11),
  C/analysis/neb_cross_element_triptychs_20261024 (23M; S3, partly mirrored).
- Not found anywhere: the edited SVG/intermediate behind fig_NEB_interlayer-sliding.pdf (sha 6f45a330, cairo, 2026-11-03 22:56).

## Already in repo (no import)
data_processed/current_release/* (parity_summary, MD15_*, fig5_*, fig6_current_*, shap_*, zero_force_*, coverage audits),
data_processed/{dft_neb_barriers.csv,path_registry.csv}, paper/revision1/scripts/build_path_registry.py,
paper/response_to_reviewers/reviewer_a2_grouped_retraining/{build_a5_path_resolved_table.py,data/*},
release/revision1_camera_ready/reproducibility/{recompute_feature_metrics.py,run_shap_full_campaign.py,fig6_current_input.npz (array-identical
to C/analysis/legacy_fig6_phate_reproduction_20261101/.../fig6_full_campaign_features.npz),source_data/common_test_267.xyz,source_tables/,
build_zero_force_training_manifest.py}, reproducibility/canva_palette_vector_bundle_20261105 (P09/P10 renderer + source_archive),
data/raw_compact/revision1/{dft_raw/1-4, 81_atom_2D_421_Diffusion_traj/neb_1..5}, paper/revision1/appendix/data/endpoint_level_summary.csv (846 rows).

## Rockfish raw locations (verified to exist 2026-11-10, read-only ls) — flag for skipjack migration
- /scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/data_grouped/{train,valid,test}.xyz + grouped_split_summary.json (S1, S11, R2, R8)
- /scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/data_grouped_scratch5/ (S2, S7, R2)
- /scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/retrain_scratch16_20260926/{MACE-multihead_600K_grouped_A2_OMATFT,_MPA0} (grouped checkpoints)
- /data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/data_600K/{train,valid,test}.xyz (R6 legacy training data)
Referenced, not individually verified: /scratch16/pclancy3/yi/{grouped_A2_ft_1ns_md_20261025, scratch_lmax1_g1_md_20261025_r01,
foundation_md_backlog_20261024, foundation_self_relaxed_4x3_20261024_r02/runs, scratch_lmax1_interlayer_sliding_20261025_r01,
scratch_lmax1_g4_self_20261025_r02}. All model checkpoint paths + sha256: data_processed/current_release/parity_model_registry.tsv,
MD15_STATUS_AUDIT_R02.tsv (model_sha256/input_sha256). Foundation checkpoints referenced only at /Users/alina/Downloads/mace-foundation-models (local, not in repo).
