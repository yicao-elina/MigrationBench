# Rockfish NEB Pipeline Audit

Date: 2026-09-09

## Executive Verdict

The Rockfish connection and Slurm execution path are working, but the current manuscript data must be split into two classes:

1. Defensible now: the fixed-geometry 1-4 in-gap benchmark with a DFT reference barrier of 0.3360499862 eV.
2. Quarantine until rerun: Fig. S3 self-NEB claims, 1-2/1-3/deep-penetration references, and the current 1-7 benchmark CSVs.

The reviewer concern about internal inconsistency is real. The auditable fixed-geometry data supports the main-text statement that the Foundation/OMAT model overestimates the 1-4 in-gap barrier by about 0.69 eV. I did not find a reproducible raw basis for saying the same Foundation self-NEB is exceptional at about 0.41 eV.

## Confirmed Rockfish Access

- SSH targets tested:
  - `ssh rockfish`
  - `ssh ycao73@login.rockfish.jhu.edu`
- Login nodes observed: `login02`, `login03`.
- User: `ycao73`.
- Slurm available via `sbatch`, `squeue`, and `sacct`.

## Confirmed Remote Paths

- NEB root: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb`
- Fine-tuning root: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning`
- Two-layer fine-tuning root: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer`
- Fixed-geometry benchmark root: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/neb_benchmark`
- Fig. S3/self-NEB-like source directory: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806-NEB`
- Victor-related material:
  - `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/Victor`
  - `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/Victor/nebProject`
  - `/data/pclancy3/yi/flare-data/victor`
  - `/data/pclancy3/yi/flare-data/victor-yi-test`

Inventory snapshots are stored locally under `data_processed/cluster/inventory/`.

## Quota Risk And Run Location

Rockfish reported:

- Home: 48.77 GB used of 50.00 GB, 97.55%.
- GPFS group `pclancy3` on `/data`: 13.62 TB used of 10.00 TB, 136%.

New active calculations should run under `/scratch16/pclancy3/yi/revision1_migrationbench_runs`. Historical data and model checkpoints under `/data/pclancy3/yi/flare-data/...` should be treated as read-only inputs. The local scripts now record this in `configs/migrationbench_pipeline.yaml`.

## Slurm Smoke Tests

Two real Slurm jobs were run on Rockfish.

### Infrastructure Smoke

- Job: `30758041`
- Name: `mb_neb_smoke_s42`
- Node: `gpu02`
- State: `COMPLETED`
- Exit: `0:0`
- Purpose: end-to-end pipeline smoke with ASE/EMT, local export, QE input generation, and dataset table generation.
- Local copy: `data_processed/cluster/mb_neb_smoke_s42/`

### Real MACE/NEB Smoke

- Job: `30759837`
- Name: `mb_mace14_smoke_s42`
- Node: `gpu02`
- State: `COMPLETED`
- Exit: `0:0`
- Input images: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-4/sb2te3.xyz`
- Model: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model`
- Output: `data_processed/cluster/mb_mace14_smoke_s42/data_processed/pipeline_smoke/mace_omat_1-4_s42/`
- Manifest barrier proxy after two steps: 1.3050244901471046 eV.

The MACE value above is not a scientific barrier because the run was intentionally capped at two optimizer steps. Final `fmax` values remained about 2.36-2.83 eV/A. Its purpose is to verify that the real Rockfish GPU, MACE environment, checkpoint, existing 1-4 images, output manifest, and local sync path all work.

A second scratch16-backed run completed after the run-root update:

- Job: `30761086`
- Name: `mb_mace14_smoke_s42`
- Node: `gpu02`
- State: `COMPLETED`
- Exit: `0:0`
- Elapsed: 29 seconds
- Run root: `/scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_mace14_smoke_s42_30761086`
- Local copy: `data_processed/cluster/mb_mace14_smoke_s42_30761086/`
- Purpose: verify that active MACE NEB outputs are written to `/scratch16`, not `/data`.

## Active Production/Preflight Jobs

Long-running NEB jobs are monitored every 3 hours by the app automation `rockfish-neb-job-monitor`; routine pending/running states should stay quiet.

Latest submitted jobs:

| Job ID | Name | Type | Purpose |
|---|---|---|---|
| `30763472` | `mb_qe17_pf2_s42` | QE preflight | Clean-template 1-7 QE `neb.x` preflight with 2 MPI ranks after the 8-rank OOM |
| `30763473` | `mb_mlff_1_4_s42` | MACE pre-NEB | 1-4 500-step MACE pre-relax from historical DFT images |
| `30763474` | `mb_mlff_1_7c_s42` | MACE pre-NEB continuation | Continue 1-7 from the 100-step MACE path for 400 more steps |
| `30763477` | `mb_qe17_pfmem_s42` | QE preflight | Clean-template 1-7 QE preflight with 2 MPI ranks and explicit `--mem=160G`; startup log confirmed `slurm_mem_per_node=163840` |

The earlier 1-7 QE preflight job `30762290` failed with `OUT_OF_MEMORY` after successful input parsing and after starting SCF for multiple images. This means the generated QE input shape was valid, but the QE parallel/memory layout was not.

## Current Fixed-Geometry 1-4 Numbers

Source: `data_processed/cluster/neb_benchmark/unified_fixed_geometry_barriers_audit.csv`

Protocol: fixed-geometry single-point energy evaluation on the DFT 1-4 NEB path.

DFT reference:

- Path: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-4/neb.out`
- Barrier: 0.3360499862 eV.
- Status: numerically stationary but not formally `JOB DONE`.

Model barriers and errors:

| Model | Barrier (eV) | Signed Error vs DFT (eV) |
|---|---:|---:|
| MACE FT - 600K | 0.4967923920 | +0.1607424058 |
| MACE FT - Multi-T | 0.8230399404 | +0.4869899542 |
| MACE Foundation | 1.0242956897 | +0.6882457035 |
| MACE Scratch | 4.5376770418 | +4.2016270556 |

For manuscript text, round the DFT reference consistently as 0.336 eV or 0.34 eV. Avoid switching between 0.34 eV and about 0.3 eV in ways that obscure the comparison.

## Data to Quarantine

### Fig. S3 / Self-NEB Claim

The directory `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806-NEB/0-foundation-omat/` contains `neb_final_path.xyz`, but ASE reads no frame energies from it. The Slurm logs include:

- `Calculated Migration Energy Barrier: 0.0000 eV`
- warning that a 0.0000 eV barrier may indicate identical endpoints or a failed path
- post-processing errors such as missing `model_paths` or plotting-theme attributes

I therefore do not see an auditable raw source for the reported Foundation self-NEB value of about 0.41 eV.

### 1-2, 1-3, 1-5, and 1-7

The old benchmark CSVs used inconsistent or propagated references:

- `1-2`, `1-3`, and `1-5` CSVs imply the 1-4 reference of 0.3360499862 eV even though their own local DFT `neb.out` files have different, unconverged barriers.
- `1-7` has internally inconsistent implied references across model rows and result directories.
- These paths should not be used for final manuscript values until rerun or manually certified through the new manifest/database layer.

### Deep Penetration

The current 1-2/1-3-like deep-penetration results are not solid enough for claims about final barriers or model ranking. They can be described as exploratory or excluded pending convergence.

## Solid Secondary DFT Candidates

The 81-atom 2D trajectory NEB jobs under `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/2.2-layer-diffusion/2D_421_Diffusion_traj` appear formally completed:

| Path | DFT Barrier (eV) | Atoms |
|---|---:|---:|
| `neb_1` | 1.034728 | 81 |
| `neb_2` | 1.561262 | 81 |
| `neb_3` | 1.691053 | 81 |
| `neb_4` | 2.063858 | 81 |
| `neb_5` | 1.414868 | 81 |

These are candidates for a secondary benchmark set, but their physical mapping to the manuscript pathways should be explicitly documented before citation.

## Recommended QE NEB Settings

For DFT refinement from MLFF-preconditioned images:

- Use QE `neb.x` with explicit `BEGIN_PATH_INPUT`, `BEGIN_ENGINE_INPUT`, and `BEGIN_POSITIONS`.
- Set `num_of_images` to the exact number of images exported by the MLFF pre-NEB.
- Use `opt_scheme='broyden'` for robust NEB path optimization.
- Use a strict `path_thr` only for final production, and looser thresholds for preflight.
- Use `minimum_image=.true.` when periodic wrapping can cause jumps between adjacent images.
- Preserve the exact image coordinates from `mlff_neb_images.extxyz`; do not regenerate linear interpolation for production DFT once MLFF preconditioning exists.

The local generator `scripts/migrationbench/qe_neb_from_images.py` implements this explicit-image strategy.

## Pipeline Artifacts Added

- `configs/migrationbench_pipeline.yaml`: canonical path/model/split/output configuration.
- `configs/migrationbench_hf_schema.yaml`: Hugging Face-style schema and required metadata.
- `scripts/migrationbench/run_mlff_neb.py`: ASE NEB preconditioner for EMT smoke, MACE foundation, or explicit MACE checkpoint.
- `scripts/migrationbench/qe_neb_from_images.py`: QE `neb.x` input generator from existing image trajectories.
- `scripts/migrationbench/migrationbench_dataset.py`: JSONL/CSV/optional Parquet exporter with per-frame metadata.
- `scripts/migrationbench/audit_neb_benchmark_barriers.py`: recomputes fixed-geometry barriers from prediction CSVs and path-matched DFT references.
- `scripts/migrationbench/smoke_migrationbench_pipeline.py`: local/Rockfish smoke pipeline.
- `scripts/migrationbench/submit_slurm_smoke_neb.sh`: Slurm infrastructure smoke.
- `scripts/migrationbench/submit_slurm_mace_14_smoke.sh`: real MACE/checkpoint/1-4 Slurm smoke.

## Data Leakage Position

The new pipeline separates records by `record_type`, `path_id`, `source_run_id`, and split metadata, and the dataset exporter no longer fabricates zero forces when forces are unavailable. This gives us the infra needed to prevent leakage:

- DFT references and evaluation NEB paths must be marked `split: test` and `record_type: dft_neb_reference`.
- Training structures must be marked `split: train` and carry their training-source lineage.
- MLFF-proposed NEB images must be stored as proposals, not as DFT labels, until DFT single-point or DFT NEB calculations attach real labels.
- No benchmark path should be used both for model training and test evaluation unless explicitly declared as an ablation.

I have not found evidence in this audit that leakage is the explanation for the reported inconsistency. The stronger current diagnosis is reference/protocol mismatch plus non-auditable self-NEB post-processing.

## Manuscript Narrative Recommended Now

Use one consistent story:

> On the fixed DFT 1-4 in-gap migration path, the reference DFT barrier is 0.336 eV. Single-point evaluation of this same path gives barriers of 0.497 eV for the 600 K fine-tuned MACE model, 0.823 eV for the multi-temperature fine-tuned model, 1.024 eV for the Foundation/OMAT model, and 4.538 eV for the scratch model. Thus, under the fixed-path protocol, the 600 K fine-tuned model is closest to DFT, while the Foundation/OMAT model overestimates the barrier by 0.688 eV.

Remove or rewrite the Fig. S3 claim that Foundation/OMAT gives an exceptional about-0.41 eV self-NEB result unless a reproducible raw trajectory with energies, forces, endpoint definitions, model version, and Slurm logs is recovered.

## Next Production Jobs

1. Rerun 1-4 DFT NEB from MLFF-preconditioned images to convert the currently numerically stationary 0.336 eV reference into a formally converged production reference.
2. Rerun 1-7 as a clean in-gap counterpart with a fresh manifest; the old 1-7 CSVs are inconsistent.
3. Finish or replace the 81/82-atom in-gap calculations, especially the current `1.Gap-Center` job, which is not yet manuscript-grade.
4. Only then regenerate tables directly from the unified dataset exporter for Hugging Face upload to `alinacao2000/MigrationBench`.
