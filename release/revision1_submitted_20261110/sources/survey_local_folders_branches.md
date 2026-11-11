# Sources survey (read-only), 2026-11-10

Consolidation base = `integrate/final-paper-20261106` + `334260c` (camera-ready). "In base" below = git blob identical somewhere in base (path-independent), unless stated.
Key finding: the Overleaf export `~/Downloads/MigrationBench_overleaf (2).zip` main/SI/response .tex are byte-identical to `334260c:paper/revision1/*` (sha256 a2252edc…, 2e65eb40…, 02d9e8ff…). This zip is the final submitted source.

## 1. ChatGPT repo `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench` (11 GB, branch master @ffa168c)
- Git: 1,987 tracked files; **12,796 untracked** (gitignore only covers caches, so everything else is simply uncommitted). No remote is configured; the stale tracking ref `github/main`=817acf2 is a MigrationBench commit. master is 10 commits ahead of it (SHAP full campaign, Fig6 UMAP finalization, Canva palette bundle, Scratch-600K R05 three-path, 1 ns figures). These commits are **not** in paper-repro. Local branches `codex/figure-repro-archive-20261101` (2f4d4ac), `codex/from-scratch-ablation-md-20261005` (worktree bf6e), `paper/cr-neb-evidence-workflow-20261016` (d96b4a8) are not in paper-repro. `codex/multimodel-639-evidence-pr` (476ca16) is in paper-repro (as origin branch).
- Worktrees `~/.codex/worktrees/5876` (detached 6553dcd, clean, 220M) and `bf6e` (e9f1d1a, clean, 201M) are duplicates of commits above → delete the worktrees after the branches are preserved.
- `rockfish_sources/` is empty (only an empty `victor/`). `hpc_repairs/` (68K, 12 files: rockfish_20261019, rockfish_20261024, omat0_zero_shot, 1ns backlog) holds Slurm repair scripts → import (small; records Rockfish job history).
- Models: 38 `*.model/*.pt` (484 MB). Paper-relevant: `task2_cross_element*/models/{FT-600K,FT_Multi_T,Scratch_Multi_T,Foundation-MACE-matpes-pbe-omat-ft}.model`, broad-roster FT seeds 123/456/789; `foundation_model_1ns_md_backlog_20261024/model_backup_rockfish_20261024/` (5 foundation checkpoints, copies of public MACE). `elementa_20261015` = smoke-run checkpoints (obsolete).
- Exact copies of final figure PDFs present here: `figS_current_feature_audit.pdf`, `fig_NEB_interlayer-sliding.pdf` (in output/overleaf_current_20261107 and output/revision1_camera_ready). Final main Fig1–4, 6 and 7 `*_new.pdf` have **no byte-identical copy anywhere outside the repo**. Only Overleaf/repo has them; Fig5 `_new` is also in Downloads.

| dir (size; text files in base / total) | unique content | class | action |
|---|---|---|---|
| analysis/four_dopant_fixed_path_palette_20261109 (1.7M; 2/5) | renderer + data for **SI Fig S3** (fig_NEB-fixed-path_Cu-Li-Na-Cr) | import | → repo figure source for S3 |
| analysis/grouped_retraining_overlap_palette_20261109 (320K; 0/3) | `plot_grouped_retraining_overlap.py` + `source_data/*.json` for **SI Fig S2** | import | → repo |
| analysis/shap_full_campaign_20261101 (11M; 24/32) | Fig7 renderer `plot_shap_top20_bar_transposed_seed_heatmap_20261102.py` + variants | import remaining 8 text files | → repo (Fig 7) |
| analysis/legacy_fig6_phate_reproduction_20261101 (335M; 1/18) | `updated_campaign_20261102_umap_density/` = **Fig 6** precursor (fig6_umap_density_full_campaign_15models_1x4) | import scripts/tsv; archive rest | Fig 6 |
| analysis/fig6_full_campaign_update_20261101 (21M; 0/17) | 15-model feature matrices/embeddings for Fig 6 | import small, archive npz | Fig 6 |
| analysis/fig6_neb_model_roster_update_20261101 (6.3M; 2/17) | NEB roster / NEB_main_figure_reference (Fig 5 lineage) | import | Fig 5 |
| analysis/canva_palette_vector_bundle_20261105 (76M; 4/10) | palette renderers (md_six_panel → **Fig 3**, Fig 2 parity) | already mostly in repo `reproducibility/canva_palette_vector_bundle_20261105` (912 path hits); import the 6 missing text files | Fig 2/3 |
| analysis/rmse_train_test_20261026 (91M; 78/100) | **Fig 2** train/test parity data | import 22 missing | Fig 2 |
| analysis/rmse_train_test_20261025 (2.5M; 0/60) | earlier version | obsolete | delete after archive |
| analysis/md_1ns_updated_publication_figure_20261025 (24M; 133/304) | **Fig 3** MD data | import missing | Fig 3 |
| analysis/migrationbench_training_design_figure_20261024 (264K; 1/11) | **SI Fig S1** training-design (method-P02) source | import | S1 |
| analysis/figure1_production_kit_v2_audit_20261101 (86M; 0/30) | **Fig 1** production kit (Blender/SOAP-PCA) | import text; archive renders | Fig 1 |
| analysis/migrationbench_universe_publication_20261026 (156M; 1/42), migrationbench_data_universe_20261026 (40M) | Fig 1b configuration-universe SOAP/PCA data | import text, archive large | Fig 1 |
| analysis/cr_interlayer_sliding_grouped_20261025 (6.4M; 58/335), foundation_interlayer_sliding_2x2_20261024 (103M; 0/154), sliding_four_model_campaign_20261024 (1.1M; 24/184), foundation_interlayer_sliding_10model_20261024 (all in base) | **SI Fig S4** / Fig 5e sliding profiles | import text; archive traj | S4 |
| analysis/neb_fixed_path_trajectory_figure_20261023 (113M; 1/43), neb_cross_element_triptychs_20261024 (23M; 40/254) | Fig 4 / S3 NEB profile sources | import text; archive traj | Fig 4, S3 |
| analysis/dft_mace_47_site_error_attribution_20261023 (9.4M; 0/651) | same as origin branch dft-mace47-evidence-package | preserve via that branch | archive |
| analysis/multimodel_639_screen_20261014 (66M; 0/64) | = origin/codex/multimodel-639-evidence-pr | preserve via branch | archive |
| analysis/task3_endpoint_only_neb (318M; 4/2049) | endpoint-only NEB campaign; part in origin/paper/cr-neb-evidence-workflow | archive-to-storage | unused-calc archive |
| analysis/from_scratch_failure_attribution_20261011 (220M; 0/132) | from-scratch MD failure audit (C9 support) | archive (import summary tsv/md) | archive |
| analysis/foundation_matpes_pbe_omat_ft_campaign_20261024 (148M; 17/359) | foundation MD/NEB campaign | archive; import summaries | archive |
| analysis/mace_1ns_merged_20261010/20261016, mace_1ns_publication_figure_20261005, mace_1ns_seed_aware_20261011, mace_md_time_convergence_20261001, mace_candidate_eval_20261001 (~70M) | earlier MD figure iterations | obsolete (superseded by md_1ns_updated_…25) | archive then delete |
| analysis/task2_cross_element*, cross_element_generalization_repair, task1_foundation_path, victor_dft_reference_benchmark, dft_single_point_validation, dft_validation_phase3, discovery_dft_* (~380M) | cross-element / DFT validation campaigns (+ models) | archive-to-storage | unused-calc archive |
| analysis/canonical_model_matrix_20261016, model_registry_unified_20261017, foundation_registry_20261024, migrationbench_asset_registry_20261025, figure_evidence_audit_20261008, paper_provenance_audit_20261013 (~13M) | registries/audits referenced by figure_registry `evidence_metadata_source` | import (small, needed for trace) | → repo registry/legacy |
| analysis/ft_5pct_grouped, ft_600k_random_reproduction_*, ft_grouped_A2_1ns_md, scratch_lmax1_grouped, strict_lmax2_control, mace_multit_grouped_retrain, mace_endpoint_* , mace_seed_completion, mace_foundation_*_1ns (<1M each) | training/control job specs (A2 grouped retraining, controls) | import (small) | → repo runs |
| analysis/cr_sb2te3_nafsi_style_20261026 (4.2G), cr_sb2te3_three_migration_types_blender (344M), migrationbench_typical10_render (227M), cr_sb2te3_five_independent_panels_preview_v1–3 (88M), dft47_representative_crystal_preview (44M), neb_blender_visual_refinement (19M), cr_sb2te3_repro_bundles (99M), cs_style_figure_reconstruction, figure2a_md_scope_reconstruction, frozen_visual_acceptance_criteria | Blender/visual design iterations (3.6 GB .blend, 1.1 GB png) | archive-to-storage (keep selected_hd_final + .blend of chosen render); previews = junk | archive / delete previews |
| analysis/elementa_20261015 (117M) | smoke-test models | obsolete | delete |
| analysis/manuscript_comment_sweep_20261106 (4.9M) | comment sweep for final text | import | → paper/revision1/history |
| analysis/phase1_*, phase2_*, shap_fig6_reproduction_20261101 | mostly in base (30/44, 56/72, 155/156, 6/12) | import remainder | — |
| artifacts/neb_blender_migration_20261024 (1.7G), sb2te3_bulk_3d_pair (299M), sb2te3_vmd_publication_figure (53M), others (<20M) | Blender/VMD render assets | archive-to-storage | archive |
| output/revision1_camera_ready (103M; 227/301 files in base), output/overleaf_current_20261107 (38M) | earlier copies of camera-ready package | obsolete (superseded by 334260c) | delete after confirm |
| output/MigrationBench_revision1_revised_package.{zip,tar.xz} (112M) | revised package tarball | obsolete duplicate | archive one, delete other |
| md_continuation_20260930 (1.6M; 0/42) | MD continuation drivers (continue_md.py, seedreplica_continue_1ns, rockfish/skipjack slurm) — **Fig 3 / Table S8 MD protocol** | import | → workflow/md_continuation |
| scripts/ (128K; 0/13), data/ (228K; 0/18), paper_evidence/ (36K), manuscript_inputs/ (8K) | helpers, collaborator DFT-NEB table tex | import | small |
| results/ (33M; 44/110) | plots | import missing text; pngs junk | — |
| tmp/ (38M), generated_images/, opaque_rgb/, transparent_rgba/ (~12M), .ruff_cache | scratch/ChatGPT images | junk | delete |

## 2. `/Users/alina/Project/migrationbench-integrate` (1.2 G)
Worktree of `integrate/final-paper-20261106` @65fa4fc. It is clean, and only .pytest_cache/.ruff_cache are ignored. There is no unique content → **remove the worktree** (`git worktree remove`) after merging base; the branch stays.

## 3. `/Users/alina/Project/migrationbench-remotion` (1.1 G; 963M node_modules, 34M build, 41M out)
Separate local-only git repo (no remote, 2 commits). `out/migration_story.mp4` is blob-identical to `video/migration_story/release/migration_story.mp4` on branch codex/migration-story-video-20261103 (in base). Unique items: **uncommitted** `figures/nature_schematics/` (figkit + fig1–6 schematic generator, BLUEPRINT.md, 13 changed/new files, Nov 3; exploratory schematics, not the submitted figures), `figures/schematics/fig_process_schematics_row.*` (also in Downloads), `out/cr_migration.mp4` (1 MB), stills. Class: archive the small unique parts (import `figures/nature_schematics` + `cr_migration.mp4` to repo `video/`/`archive/exploratory_schematics`). node_modules/build = junk. Then delete the folder.

## 4. Codex worktrees
- `~/.codex/worktrees/revision1-camera-ready-evidence` (244M): clean @334260c → remove worktree after merge.
- `~/.codex/worktrees/neb-seed-pairing-heatmap` (697M): 18 untracked raw NEB files (528 MB: ftmultit/omat0/scratch600k `*_self_extension_20261101` traj/extxyz). All 18 are **byte-identical in local branch `backup/neb-seed-pairing-heatmap-wip-20261106`** (not pushed). → archive raw to storage, then remove the worktree.

## 5. paper-repro working tree
- 126 untracked raw files (`analysis/neb_protocol_matrix_20261018/raw/{self_current,self_gpu_retry_pending}` 1.7 GB; `data/raw_compact/sliding_recovered_20261024/*`): **all 126 byte-identical in local `backup/main-checkout-discovery-wip-20261106`**. Raw NEB traj → archive-to-storage (+ git-ignore), not into git main history.
- Untracked build products: root `response_to_reviewers.*` (pdf/xdv/aux…, root also has sn-article-SI.aux/fls/log) and `paper/revision1/*.pdf/.xdv` → junk (rebuildable); delete. `paper/revision1/figure_pages/assets`, `figure_pages.{pdf,toc,xdv}`, `figure_book.{pdf,xdv}` (65 MB) → generated, ignore.
- `paper/revision1/figures/_audit_originals/` (63M, untracked): pre-correction originals (e.g. figS_dft_neb_mep_profiles, fig_md_msd_kappa_revised_with_status_labels) → archive (evidence of what was corrected), not main tree.
- `tmp/` (97M = tmp/pdfs reviewer_20261023 build dirs; also in backup branch) → junk. `work/` (63M: uv-bootstrap 60M = junk; `rockfish_shap_candidates` 2.7M → import/inspect; latex logs junk). Root `revision1/tables/` is an empty dir → delete.
- Local backup branches: `backup/canva-figure-provenance-wip` — all added files present in integrate → delete. `backup/main-checkout-discovery-wip` — 586 added files not in integrate (raw NEB, sliding_recovered raw, tmp pdfs, a figure_registry/figure_pages snapshot) → archive raw then delete/keep as tag. `backup/neb-seed-pairing-heatmap-wip` — 17 raw_input files not in integrate → archive raw. None of the backups are pushed.

## 6. Remote-only origin branches (vs base)
| branch | holds | recommendation |
|---|---|---|
| origin/codex/dft-mace47-evidence-package-20261023 (+2) | 62 new files: analysis/dft_mace_47_site_error_attribution_20261023 (DFT vs MACE 47-site endpoint evidence) | not superseded; unused in final paper → **tag `archive/dft-mace47-evidence-20261023`**, delete branch (or merge into archive/ area) |
| origin/codex/document-mlip-639-selection-provenance (+3) | 44 new: 639-candidate MLIP selection provenance, workflow/from_scratch_mlip_candidates, from-scratch failure attribution text | tag + delete (or merge docs/workflow parts) |
| origin/codex/multimodel-639-evidence-pr (+7) | analysis/multimodel_639_screen_20261014 (58 files) | tag + delete |
| origin/paper/cr-neb-evidence-workflow-20261016 (+3) | 136 new: task3_endpoint_only_neb evidence + paper_evidence release workflow | tag + delete |
| origin/revert-16-codex/paper-figures-repro-release (+1) | revert commit only, no new files | delete |
| origin/dev, origin/archive/main-before-paper-repro-2026-09-17, origin/chore/skipjack-production-environment, origin/fix/qe-mlff-monitoring-and-neb-controls | 0 commits ahead (fully merged) | delete branch (keep archive/* as tag if desired) |
Also: origin/main is stale (2026-09-29); `334260c` (the submitted source) and its parent 8e9947e are **NOT on origin** (origin/codex/revision1-camera-ready-evidence = 3578142); backups are unpushed too.

## 7. Overleaf git remote `overleaf` (693b515e…)
Stale/different project. Its `revision1/sn-article-final.tex` and `sn-article-r1.tex` carry the old title "Non-Equilibrium Properties Reveal the Limits of Specialist and Foundation Machine…". The 2026-11-11 commit only adds the author Yile Wu to `sn-article-r1.tex`. The submitted title ("When Does a Foundation Model Help?") exists only in the Overleaf project exported as `MigrationBench_overleaf (2).zip` (= 334260c). → Mark the remote as legacy (rename `overleaf-legacy`) and record the true project URL.

## 8. ~/Downloads (MigrationBench-related)
| item | size/date | status |
|---|---|---|
| d1567691-…pdf | 22M, Nov 10 | **submitted PDF** → copy into repo `paper/revision1/submitted/` (+sha256) |
| MigrationBench_overleaf (2).zip | 21M, Nov 9 | final source = 334260c → keep one archived copy |
| MigrationBench_1109/{main,SI,R2R}.pdf | 10M, Nov 9 | pre-submission compiled PDFs → superseded by submitted PDF |
| fig5_NEB_main_figure_new.pdf, SI-fig1.pdf, grouped_retraining_overlap_palette_updated.pdf, fig_NEB-fixed-path_Cu-Li-Na-Cr_palette_updated.pdf, fig1_campaign_overview.pdf, sn-article-SI.tex (Nov 8) | <3M | byte-identical to 334260c figures → superseded duplicates |
| MigrationBench_ALL_CANVA_and_publication_figures_backup_20261102 2/ | 76M, 137 files; 99 in base | 38 unique (Canva page exports) → archive |
| NEB_Dotmatrix_v3/ | 3.2M; 1/26 in base | Fig 5 dot-matrix v3 bundle (plotted_values.tsv, audit.json) → import |
| Figure1_Production_Kit_v2.zip (37M), Figure1_source_package.zip, Figure1_Production_Guide.pdf, Figure1_Ideal_Visual.png, Figure1_Elements_Contact.png, MACE_Interface_Vector.pdf | Nov 1 | Fig 1 design kit; overlaps ChatGPT figure1_production_kit_v2_audit → archive one copy |
| NEB_Nature_v2_complete.zip, NEB_main_figure.pdf, fig6_neb_roster_aligned_v1.pdf, fig6_panel_c_umap_density…pdf, fig_shap_full_campaign*.pdf, variant04_…pdf, fig_process_schematics_row.pdf | Nov 1–2 | intermediate panel exports → superseded |
| Revision_Package_Complete.zip (7.6M), Cr_migration_reproduction_source.zip (16M), Cr_migration_in_Sb2Te3_final_rgba_4500x3000.png (24M) | Nov 1–2 | earlier packages / render → archive |
| MigrationBench_Design_and_Captions.pdf, _Figure_Blueprint, _Main_Figures, _Main_Figure_Proofs_v3, _Figure_Design_v3, wiley series*.pdf/pptx, WileyAIS…submission (2).pdf, 5Global_Visual_Standard, ChatGPT Image *.png, *slab*.png | Nov 1 | design drafts → superseded (wiley series/WileyAIS = first-submission context; keep one) |
| sn-article-SI-clean.pdf (Nov 6), response_to_reviewers.pdf (Nov 4) | | superseded |
| 26_Migration_as_a_Probe_A_Gene.pdf (Feb), MRS2026/MRS2025 pptx | | original submission/talks → keep outside repo |
| mace-models/ (2.6 GB, Jun) | public MACE foundation checkpoints + 2023-12 early models + replay xyz | archive-to-storage (models used: matpes-pbe-omat-ft, omat-0-medium, mp-0b3, mh-1 — record sha256 in registry) |
| neb_extxyz_outputs/ (Jun) | early NEB extxyz | legacy → archive |
