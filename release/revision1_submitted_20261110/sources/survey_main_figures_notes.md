# Main-text figure provenance — Revision 1 submission (2026-11-10)

Source of truth: `codex/revision1-camera-ready-evidence@334260c`, `paper/revision1/sn-article-final.tex`
(text match to submitted PDF 0.9996). Paths in `main_figures.tsv` are relative to the
`integrate/final-paper-20261106` checkout (`/Users/alina/Project/migrationbench-integrate`) unless absolute.
`rockfish:` = `/scratch16/pclancy3/yi/...` or `/data/pclancy3/yi/...` on Rockfish (scratch16 = purge risk → migrate to skipjack).

## Status count (25 rows)
FULL 14 · PARTIAL 10 · MISSING 1 (graphical abstract)

## Common finding: every final main figure is a Canva export
All seven `*_new.pdf` / `fig6_..._v2.pdf` have Creator/Producer = Canva, title `wiley-figure-clean`.
Script outputs → (PDF palette recolor for Fig 2) → Canva relabel/compose. No Canva page export, page ID,
or editable source is archived for the final `wiley-figure-clean` design. Known designs: `DAHZXpLtN6U`
(current, 2026-11-01, connector REAUTH_REQUIRED) and `DAHZabm1B1g` (previous 10-page catalog, PR #14).
Action: export each Canva page (PDF+SVG) and record design ID + page number per figure.

## Per-figure summary
- **Fig 1** (PARTIAL): a,c = Blender renders, all `.blend` + PNG + sha in
  `analysis/cr_sb2te3_blender_render_set_20261105`. b = SOAP-PCA (`build_figure01a_clean_panel.py`,
  `data/figure01/comprehensive_task_map_points.csv`). Upstream xyz `raw_cache/` (51 MB) is git-ignored.
- **Fig 2** (FULL): `build_slide3_energy_force_train_test_separated.py` on 24 parity NPZ in
  `docs/.../data/figure02/raw_rockfish_parity`. Counts in figure/caption verified against the manifest
  (5,358/801/7,977/2,301 energy; 1,336,032/206,658/1,983,330/584,658 force). The render manifest
  cites dead paths (`~/.codex/worktrees/paper-figures-repro-release/...`, `~/Downloads/... (1).pdf`); same-sha copies are in git.
- **Fig 3** (FULL, raw on rockfish): Canva bundle `render_md_six_panel_palette_20261102.py` → R06G MD15 tables.
  15 MD runs: FT `rockfish:/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025/runs`, Scratch
  `.../scratch_lmax1_g1_md_20261025_r01/runs`, Foundation `.../foundation_md_backlog_20261024`.
  MD15_STATUS_AUDIT: only FT600K s234 + 3 foundation reached 1 ns; 11 are SCIENTIFIC_FAILURE (horizons censored).
- **Fig 4** (PARTIAL): `render_neb_palette_20261102.py` + R05 tables (in git). **Gap:** the DFT-NEB reference
  curves are hard-coded arrays in `make_cr_three_path_2x3_R02/R04.py::dft_reference()`, copied from the vector line
  of `~/Downloads/foundation_self_relaxed_4x3_R03.pdf`. I verified the arrays equal
  `ChatGPT/.../neb_fixed_path_trajectory_figure_20261023/python/data/dft_profiles.tsv`, which is parsed from three
  QE `neb.out` files in `.../rockfish_cr_neb_20261013/raw/2L-Cu__neb-{gap-diffuse,gap2QL,QLs}/`.
  The directories are Cu-named, but the content is Cr-substituted (81 atoms: Cr 1, Sb 32, Te 48).
  Rockfish origin: `/scratch16/pclancy3/yi/MigrationBench_rockfish_cr_neb_20261013`. None of these files are in git (476 KB total).
  DFT status COMPLETED_BUT_SCF_WARNING. The final asset carries the Canva typo "Self-relaxted".
- **Fig 5** (FULL b–e; a PARTIAL): `release/revision1_camera_ready/reproducibility/render_fig5_corrected.py`
  + `fig5_corrected_input.tsv` (sha matches audit). 120 records; 5 NONCONVERGED flagged; correction audit
  documents the Scratch-600K s123 within-QL self fix. Panel a was redrawn in Canva.
- **Fig 6** (PARTIAL): the feature matrix is in git (`fig6_current_input.npz`, content-identical to the
  ChatGPT NPZ, different zip bytes). The **prepare/render/palette scripts, embeddings NPZ and source
  manifests exist only in** `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/legacy_fig6_phate_reproduction_20261101`
  (336 MB; trajectory_samples 324 MB). The prepare script pulls frames from rockfish over ssh.
- **Fig 7** (FULL): SHAP scripts, common_test_267.xyz and 12 source tables are in git; outputs are claimed
  byte-reproducible in the pinned env. 189/190 final text tokens match the script PDF.
- **Graphical abstract** (MISSING): no source file anywhere local (mdfind, all repos, ChatGPT, Downloads).
- **Hi-res uploads pp.56–62**: identical to the seven embedded assets.

## Models (all sha256-recorded in parity_model_registry.tsv / MD15_STATUS_AUDIT_R02.tsv)
- FT-600K s123/234/345: `rockfish:/scratch16/pclancy3/yi/revision1_migrationbench_runs/ft600k_grouped_A2/{retrain_scratch16_20260926,retrain_matrix_scratch16_20260927}/...stagetwo_compiled.model`
- FT-MultiT: `.../revision1_migrationbench_runs/multit_grouped_A2_20261001/retrain_multit_scratch16_20261001/multit_omatft_grouped_seed*/`
- Scratch-600K/MultiT Lmax1: `rockfish:/scratch16/pclancy3/yi/scratch_lmax1_grouped_20261019_repair_r01/models/`
- Foundation checkpoints: found locally and sha-verified at `~/Downloads/mace-mpa-0-medium.model` (75428afe),
  `~/Downloads/mace-models/mace-mp-0b3-medium.model` (2f2be696) and `.../mace-omat-0-medium.model` (d4b14be9).
  The bundle pointer `PTR-FOUNDATION-CHECKPOINTS` → `/Users/alina/Downloads/mace-foundation-models` does NOT exist (stale pointer).
- Training/test data: `rockfish:.../ft600k_grouped_A2/data_grouped/{train,test}.xyz` (sha 04b1369d… / d49cf4b2…) and the MultiT equivalents.

## Files to import to close the trace (absolute path, size)
1. `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013/` (476 KB; 3× neb.out 62/105/126 KB + inputs + QE summary)
2. `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_fixed_path_trajectory_figure_20261023/python/` (8.7 MB; dft_profiles.tsv, fixed_path_long.tsv, builders) + figure_data_manifest_R03.json
3. `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/legacy_fig6_phate_reproduction_20261101/{prepare,render,update_palette}_fig6_*_20261102.py` (~53 KB), `updated_campaign_20261102_umap_density/{fig6_full_campaign_manifest.json,qa/,fig6_full_campaign_embeddings.npz}`, `updated_campaign_20261102/source/{preparation_manifest.json (16 KB),feature_metadata.tsv,sample_manifest.tsv}`; trajectory_samples (324 MB) → archive/LFS, not plain git
4. `/Users/alina/Project/migrationbench-paper-repro/analysis/task_conditioned_distribution_atlas_20261018/raw_cache/` (51 MB, git-ignored) → LFS or external archive + sha manifest
5. `/Users/alina/Downloads/foundation_self_relaxed_4x3_R03.pdf` (69 KB): the source cited by the hard-coded DFT arrays
6. `/Users/alina/Downloads/variant04_temperature_plus_scratch_ft_shared_legend_square (1).pdf` (1.58 MB): the Fig 2 recolor input, not in git (the slide3 input is in git by sha)
7. Canva exports for the final `wiley-figure-clean` pages (Fig 1–7 + graphical abstract): not on disk
8. Foundation checkpoints (3 files, ~80–250 MB each) in ~/Downloads: archive with sha, not git

## Other notes (outside scope)
- `analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite` (in git, integrate) indexes these assets. It is likely the "SQL database" the user asked about.
- Rockfish raw roots referenced (all `/scratch16/pclancy3/yi/`): revision1_migrationbench_runs, scratch_lmax1_grouped_20261019_repair_r01,
  grouped_A2_ft_1ns_md_20261025, scratch_lmax1_g1_md_20261025_r01, foundation_md_backlog_20261024,
  neb_fixed_vs_self_matrix_20261019, foundation_self_relaxed_4x3_20261024_r02, scratch_600k_migration_20261101,
  scratch_lmax1_g4_self_20261025_r02, ft_interlayer_sliding_20261025_r0{1,2}, foundation_interlayer_sliding_10model_20261024,
  MigrationBench_rockfish_cr_neb_20261013; plus `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3` (MD start config, AIMD source).
- Confidence: high for Fig 2/3/5/7 (manifest sha + number checks); medium for Fig 4 (DFT values verified but sourced from a PDF);
  medium for Fig 1/6 (manual Canva relayout; panel content matched by legend/token checks, not bytes).
