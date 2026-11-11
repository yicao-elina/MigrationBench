# Rockfish inventory — MigrationBench (read-only survey, 2026-11-10)

Host `rockfish` (login03, user ycao73). All commands were read-only. Skipjack
was not reachable non-interactively (keyboard-interactive auth), so nothing
there was verified.

## Where things live

| Root | Filesystem | Role |
|---|---|---|
| `/scratch16/pclancy3/yi/` | scratch16 (2.0 PB, 86% full) | **All Revision-1 campaigns**: 205 top-level dirs |
| `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/` | data (5.2 PB, 71%) | Original project: AIMD, training pools `data_600K`/`data_Multi_T`, legacy G1 QE NEB, original-submission models |
| `/scratch4/pclancy3/yi` | scratch4 | group scratch; not referenced by any repo ledger |
| `/home/ycao73` | home (83% full) | logs (~320 .err/.out), `~/.cache/mace` (2023 models) — nothing paper-critical |

No per-user quota is reported (`quota`: none). The scratch16 purge policy was
**not verified** from the cluster. Treat scratch16 as at-risk: every paper-critical
Revision-1 artifact lives there **except** the training pools and legacy G1 references.

## Footprint

- scratch16/yi: about 5.3 TiB by `du`. `revision1_migrationbench_runs` alone is
  3.1 TiB; the other 203 dirs total 2.2 TiB.
- By apparent bytes (find): QE wavefunction/save/mix scratch 4.7 TB,
  per-epoch `*_epoch-N.pt` checkpoints 0.89 TB, trajectories 0.12 TB, and
  everything else (inputs, outputs, logs, final models, tables) only 0.06 TB.
- The **migration payload is small**. Dropping QE `wfc*/.save/.mix` and per-epoch
  `.pt` files (keeping final `.model`, `_compiled`, `_stagetwo*`, `_swa.pt`, `run-*.log`):
  - paper-used (`yes`): about 35 GB (30 GB of MD/NEB trajectories)
  - partial: about 7 GB
  - unused-but-keep: about 110 GB
  - Total is about 150 GB plus the `/data` training pools and AIMD (about 40 GB).

## Verified paper checkpoints (sha256 recomputed on Rockfish = recorded)

All 12 trained models in `parity_model_registry.tsv` exist and match:
- FT-600K s123/234/345, FT-MultiT s123/234/345: `revision1_migrationbench_runs/{ft600k_grouped_A2,multit_grouped_A2_20261001}/...stagetwo_compiled.model`
- Scratch-600K / Scratch-MultiT Lmax1 s123/234/345: `scratch_lmax1_grouped_20261019_repair_r01/models/*`

The 4 grouped train/test xyz files also match (`data_grouped/{train,test}.xyz`,
`data_grouped_multiT/{train,test}.xyz`). The MD15 foundation models match
MD15_STATUS_AUDIT_R02 hashes: OMAT-0 medium `d4b14be9…`, MP-0b3 `2f2be696…`,
MPA-0 `75428afe…`, located in `foundation_grouped_benchmark_20261001/models/` and
`flare-data/.../X-FORCE/xforce/models/`. MatPES-PBE-OMAT-FT is `e618ad58…`
(matches the asset registry). Training hyperparameters are in each model dir's
`*_run-<seed>.log` (MACE logs full args). The submit scripts are listed in
`rockfish_models.tsv`.

## Must-migrate (used_in_paper = yes) → proposed `migrationbench_archive/`

- models/finetune_grouped_A2, models/scratch_lmax1_grouped, models/foundation
- training_data/grouped_A2 (4 xyz) and training_data/source_{600K,MultiT} (/data)
- dft/qe_neb/cr_canonical_81atom_20261013: `MigrationBench_rockfish_cr_neb_20261013`
  (550 GB, of which 98% is wfc). This is the canonical 1.52 eV in-gap reference
  (Fig. 4/5, Table S4). Keep `*.in/*.out/*.xml/*.xyz`.
  NOTE: `canonical_dft_barriers.tsv` flags all three paths `scf_warning=TRUE`,
  `EXCLUDE_FROM_PUBLICATION`. The status wording should be reconciled with the paper's use.
- dft/qe_neb/legacy_G1_61atom (`flare-data/.../3.neb/1.vdw_corr_DFT_D3/1-2..1-7`): Table S5/S6
- neb/: `neb_fixed_vs_self_matrix_20261019` (derived_r01), `_20261101`,
  `foundation_self_relaxed_4x3_20261024_r02`, `foundation_matpes_pbe_omat_ft_fixed_path_4x3_20261024_r01`,
  `scratch_600k_migration_20261101`
- md/md15_1ns: `grouped_A2_ft_1ns_md_20261025`, `scratch_lmax1_g1_md_20261025_r01`,
  `mace_foundation_representatives_1ns_20261006`, `mace_foundation_omat_true_1ns_20261013`
- sliding/: `ft_interlayer_sliding_20261025_r02`, `_r03_retry`, plus foundation/four-model sliding (partial)
- analysis/: `md15_global_rdf_r06_run01`, `md15_partial_rdf_r05_run01`,
  `task_conditioned_distribution_atlas_20261018`, `scratch_lmax1_g5_{shap,embedding,parity}_*`, `rmse_train_test_*`
- aimd/ (`flare-data/.../2.aimd`, including 2L_o3_t3_600K_aimd_1, the SHAP common test trajectory)

## Optional (unused-but-keep; harmonization differs from paper)

Broad-roster FT seeds 123/456/789 (`cross_element_generalization_*`: 830 GB, of
which 97% is epoch .pt; keep 3 final models + logs). Also:
- canonical_model_matrix_20261016, ft_5pct_grouped_v1
- foundation3_ft_seed123, omat0_ft_seedmatrix
- ft_600k_random_reproduction_*, strict_lmax2_control
- from_scratch_* ablations, scratch_lmax1_grouped_20261016 (superseded by repair_r01)
- earlier MACE NEB/MD campaigns
- DFT restart/reconvergence attempts (`cr_81atom_neb_path_restart*`, 470 GB wfc; `phase3_dft_recovery_*`)
- original-submission models in `flare-data/.../3.fine-tuning` and `3.neb/mlff/models`

## Not MigrationBench / junk

Not MigrationBench: Gen-COMPAS, 3d-polymer-topology, polymer_campaign*,
xingfei_*, campaign3_protocol_calibration_*, qe-jamie-sandbox, defects_scripts,
archives (phonon/raman), WSe2/PbS under flare-data, and `1.2_bands`.

Junk: smoke/prep_failed/retry scaffolding, venvs, `__pycache__`, `=0.4.0`/`=3.1.0`
pip artifacts, about 320 slurm logs in $HOME, and `~/.cache/mace` 2023 models.

## Caveats

- Classification is rule-based on directory names plus repo ledger references. 39
  `partial` rows need a human check (especially sliding and g5 analysis dirs).
- Checkpoints of unused runs were not hashed.
- `du` timed out on `revision1_migrationbench_runs`. Its size comes from a longer
  run (3.1 TiB); n_files comes from `find`.
- Local copies: `data/raw_compact/{revision1,rockfish}` holds compact extracts of
  ~100 job dirs; `analysis/` holds several same-named campaign dirs (marked `yes:` in local_copy).
