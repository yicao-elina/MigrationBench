# Cluster Inventory

Date: 2026-09-09

## Scope

This inventory covers the Rockfish paths needed for the NEB/data-provenance revision work. The files under `/data/pclancy3/yi/...` are treated as historical/read-only inputs because the group quota on `/data` is full. New active calculations should run under `/scratch16/pclancy3/yi/revision1_migrationbench_runs`.

## Access

- SSH alias: `rockfish`
- Direct login: `ycao73@login.rockfish.jhu.edu`
- User observed: `ycao73`
- Login nodes observed: `login02`, `login03`
- Slurm: available through `sbatch`, `squeue`, and `sacct`

## Storage

- Historical results root: `/data/pclancy3/yi/`
- Historical project root: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3`
- Active run root: `/scratch16/pclancy3/yi/revision1_migrationbench_runs`
- Local sync target: `Revision1/data_processed/cluster/<job>/`

Rockfish quota output showed `/data` group `pclancy3` at 136% of quota. Use `/scratch16` for new runs.

## Key NEB Paths

| Purpose | Path |
|---|---|
| NEB root | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb` |
| DFT-D3 fixed-path root | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3` |
| 1-4 DFT path | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-4` |
| 1-7 DFT path | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-7` |
| DFT-D3 summary XYZ | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/Em-xyz` |
| DFT-D3 summary DAT | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/Em-dat` |
| 81-atom 2D diffusion | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/2.2-layer-diffusion/2D_421_Diffusion_traj` |
| Gap-center 82-atom run | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/2.2-layer-diffusion/1.Gap-Center` |

## Model And Training Paths

| Purpose | Path |
|---|---|
| Fine-tuning root | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning` |
| Two-layer root | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer` |
| Foundation/OMAT checkpoint | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model` |
| Reported FT-600K logs/models | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K` |
| Reported FT-MultiT logs/models | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T` |
| Reported scratch seed-123 logs/models | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802` |
| X-FORCE model copies | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE/xforce/models` |
| FT-600K training data | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/data_600K` |
| FT-MultiT training data | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/data_Multi_T` |
| Fixed-geometry benchmark | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/neb_benchmark` |

## Fig. 3d vs Fig. S3 Source Paths

| Figure/Protocol | Path | Current Audit Status |
|---|---|---|
| Fig. 3d fixed-geometry benchmark | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/neb_benchmark` | Recomputable; 1-4 is usable, 1-7 and deep paths quarantined |
| Fig. S3 self-NEB-like runs | `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806-NEB` | Not currently auditable for the reported 0.41 eV Foundation barrier |

## Victor Paths

| Path | Notes |
|---|---|
| `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/Victor` | Contains Victor-related directories |
| `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/Victor/nebProject` | Found as a directory; no finished NEB outputs verified yet |
| `/data/pclancy3/yi/flare-data/victor` | Separate Victor root |
| `/data/pclancy3/yi/flare-data/victor-yi-test` | Contains at least `1.in` |
| `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/md_path_search/assignments/victor_sites.csv` | Site assignment metadata |
| `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/md_path_search/assignments/slurm_victor.sh` | Victor assignment Slurm script |

## Local Evidence Files

- `data_processed/cluster/inventory/key_dirs.txt`
- `data_processed/cluster/inventory/finetuning_files.txt`
- `data_processed/cluster/inventory/neb_files.txt`
- `data_processed/cluster/inventory/victor_paths.txt`
- `data_processed/cluster/inventory/rockfish_neb_out_summary.csv`
- `data_processed/cluster/training_provenance/zero_force_training_audit.csv`
- `data_processed/cluster/training_provenance/isolated_atom_zero_force_frames.txt`

## Missing Or Weak Evidence

- No reproducible raw-energy source has been found for the Fig. S3 Foundation self-NEB value of about 0.41 eV.
- Victor directories have not yet yielded finished NEB barriers.
- Existing deep-penetration DFT NEB outputs are not solid enough for manuscript use.
- `/data` is full, so all new production jobs must be designed for `/scratch16`.
