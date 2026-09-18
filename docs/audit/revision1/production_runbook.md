# MigrationBench Production Runbook

Date: 2026-09-09

## Storage Rule

Use `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3` only as a read-only historical source. Use `/scratch16/pclancy3/yi/revision1_migrationbench_runs` for all new active calculations.

Use CPU Slurm resources first. QE NEB should run on the `parallel` partition with account `pclancy3`; MACE preconditioning also has a CPU path and should use GPU only when CPU walltime is the limiting factor.

Local post-processing target:

```bash
Revision1/data_processed/cluster/<job>/
```

## Stage 0: Inputs

Canonical historical inputs:

```bash
DFT_ROOT=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3
MODEL_ROOT=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer
RUN_ROOT=/scratch16/pclancy3/yi/revision1_migrationbench_runs
```

Start with:

```bash
PATH_ID=1-4
IMAGES=${DFT_ROOT}/${PATH_ID}/sb2te3.xyz
PW_TEMPLATE=${DFT_ROOT}/${PATH_ID}/pw_1.in
MODEL=${MODEL_ROOT}/MACE-omat/MACE-matpes-pbe-omat-ft.model
```

## Stage 1: MLFF Pre-NEB

Purpose: get a smoother image path cheaply. This is not the final DFT barrier.

Script:

```bash
scripts/migrationbench/run_mlff_neb.py
```

Slurm template:

```bash
scripts/migrationbench/submit_slurm_mlff_neb.sh
scripts/migrationbench/submit_slurm_mlff_neb_cpu.sh
```

Convenience submitter:

```bash
scripts/migrationbench/submit_mlff_neb_job.sh PATH_ID IMAGES MODEL_PATH [STEPS] [SEED]
scripts/migrationbench/submit_mlff_neb_cpu_job.sh PATH_ID IMAGES MODEL_PATH [STEPS] [SEED] [TIME] [MEM] [CPUS]
```

For CPU-first production, increase optimizer steps from the smoke value:

```bash
bash scripts/migrationbench/submit_mlff_neb_cpu_job.sh \
  "${PATH_ID}" "${IMAGES}" "${MODEL}" 500 42 06:00:00 48G 8
```

Acceptance:

- `mlff_neb_manifest.json` exists.
- `mlff_neb_images.extxyz` exists.
- `initial_profile`, `profile`, and `relaxation_energy_delta` exist so the energy effect of preconditioning is inspectable image-by-image.
- `fmax_eV_A <= 0.05` for production or explicitly marked as preconditioner-only.
- no image has an unphysical energy spike greater than 5 eV relative to neighboring images.

## Stage 2: QE Input From MLFF Images

Purpose: convert the MLFF-preconditioned path into a QE `neb.x` input with explicit image coordinates.

Scripts:

```bash
scripts/migrationbench/qe_engine_template_from_pw.py
scripts/migrationbench/qe_neb_from_images.py
```

Command:

```bash
python scripts/migrationbench/qe_engine_template_from_pw.py \
  --input "${PW_TEMPLATE}" \
  --output "${RUN_ROOT}/mb_qe_${PATH_ID}_s42/qe_engine_template.in"

python scripts/migrationbench/qe_neb_from_images.py \
  --images "${RUN_ROOT}/mb_mace_${PATH_ID}_s42/mlff/mlff_neb_images.extxyz" \
  --engine-template "${RUN_ROOT}/mb_qe_${PATH_ID}_s42/qe_engine_template.in" \
  --output "${RUN_ROOT}/mb_qe_${PATH_ID}_s42/neb.in" \
  --manifest "${RUN_ROOT}/mb_qe_${PATH_ID}_s42/neb.manifest.json" \
  --path-id "${PATH_ID}" \
  --path-family "<continuous-descriptor-family>" \
  --measurement-purpose dft_refinement_of_mace_preconditioned_path \
  --source-neb-output "<historical-or-parent-neb.out>" \
  --source-mlff-manifest "${RUN_ROOT}/mb_mace_${PATH_ID}_s42/mlff/mlff_neb_manifest.json" \
  --nstep-path 1000 \
  --path-thr 0.03 \
  --opt-scheme broyden \
  --ci-scheme auto \
  --minimum-image
```

Acceptance:

- `neb.in` contains `BEGIN_POSITIONS`.
- `num_of_images` equals the MLFF image count.
- `INTERMEDIATE_IMAGE` count equals `num_of_images - 2`.
- The manifest records path id, path family, measurement purpose, source image SHA256, engine-template SHA256, source MLFF manifest SHA256, source historical QE output SHA256 when present, and the upstream MLFF energy-delta table.

## Stage 3: DFT NEB Refinement

Slurm template:

```bash
scripts/migrationbench/submit_slurm_qe_neb.sh
```

Convenience submitter:

```bash
scripts/migrationbench/submit_qe_neb_job.sh JOB_NAME LOCAL_OR_REMOTE_NEB_IN [NTASKS_PER_NODE] [TIME] [MEM]
```

Submit from Rockfish:

```bash
cd "$HOME/revision1_pipeline_20260909"
export MIGRATIONBENCH_RUN_ROOT=/scratch16/pclancy3/yi/revision1_migrationbench_runs
export MIGRATIONBENCH_NEB_INPUT=/scratch16/pclancy3/yi/revision1_migrationbench_runs/mb_qe_${PATH_ID}_s42/neb.in
sbatch scripts/migrationbench/submit_slurm_qe_neb.sh
```

Acceptance:

- Slurm state is `COMPLETED` with exit `0:0`.
- QE `neb.out` contains successful termination.
- Barrier is recomputed from raw `neb.out` or `neb.dat` by a local script.
- Final paper values require `path_thr <= 0.03 eV/A`, or the value must be marked as provisional.

If a QE preflight OOMs after input parsing, first retry with explicit memory and fewer MPI ranks because Slurm may otherwise assign a small default memory cgroup and QE can replicate substantial state per rank. The first 1-7 preflight with 8 ranks parsed successfully but failed with `OUT_OF_MEMORY`; the corrected submit path uses a cleaned template, 2 ranks by default, and `--mem=160G`.

## Stage 4: Local Sync

Copy only logs, manifests, inputs, summary outputs, and compact trajectory files back to the local revision folder:

```bash
mkdir -p Revision1/data_processed/cluster/<job>
ssh rockfish "cd /scratch16/pclancy3/yi/revision1_migrationbench_runs && tar czf - <job>" \
  | tar xzf - -C Revision1/data_processed/cluster/<job>
```

Avoid copying large wavefunction/outdir folders unless they are needed for restart.

Convenience sync:

```bash
scripts/migrationbench/sync_cluster_job.sh JOB_DIR_NAME
```

## Stage 5: Dataset Export

Script:

```bash
scripts/migrationbench/migrationbench_dataset.py
```

Record types:

- `dft_training_frame`: training labels from AIMD/DFT trajectories.
- `dft_neb_reference`: final DFT NEB reference path.
- `mlff_neb_proposal`: MLFF-relaxed path before DFT labeling.
- `mlff_fixed_path_prediction`: MLFF single-point energies on DFT images.

Every exported row must include:

- `path_id`
- `record_type`
- `source_run_id`
- `calculator_label`
- `method`
- `split`
- `convergence_status`
- raw artifact path
- SHA256 of source image/log where practical

Initial Hugging Face upload can use JSONL+CSV. Parquet should be added once `pyarrow>=13` or `fastparquet` is installed in the export environment.

## Immediate Production Queue

### P0: 1-4 In-Gap

Goal: formalize the currently defensible 0.336 eV reference.

- Input: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-4/sb2te3.xyz`
- Current DFT barrier: 0.3360499862 eV.
- Current status: numerically stationary but not formally `JOB DONE`.
- Run: MACE pre-NEB, then QE DFT NEB refinement.

### P1: 1-7 In-Gap

Goal: repair the second in-gap pathway.

- Input: `/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-7/sb2te3.xyz`
- Current DFT barrier: 0.5825860632 eV.
- Current status: unconverged and old benchmark CSVs are internally inconsistent.
- Run: full new manifest path; do not reuse old `neb_barrier_results.csv` errors.

### P2: 81/82-Atom Secondary System

Goal: decide whether this is paper-grade second-system evidence.

- Descriptor-classified 81-atom deep-penetration candidates:
  - `2D_421_Diffusion_traj/neb_1`: 1.034728 eV historical candidate, max image error 0.355916 eV/A.
  - `2D_421_Diffusion_traj/neb_2`: 1.561262 eV historical candidate, max image error 0.635720 eV/A.
  - `2D_421_Diffusion_traj/neb_3`: 1.691053 eV historical candidate, max image error 0.231831 eV/A.
  - `2D_421_Diffusion_traj/neb_4`: 2.063858 eV historical candidate, max image error 0.526984 eV/A.
  - `2D_421_Diffusion_traj/neb_5`: 1.414868 eV historical candidate, max image error 0.319104 eV/A.
- Weak current 82-atom path:
  - `1.Gap-Center`: not manuscript-grade yet.

Before citing these, rerun the CPU MACE preconditioning and CPU QE refinement path. The physical label should come from `scripts/migrationbench/classify_neb_path_geometry.py` plus the continuous descriptors, not from the directory name alone.

## Manuscript Gate

No number goes into the manuscript unless all are true:

- source artifact exists locally under `data_processed/cluster/`
- source artifact path on Rockfish is recorded
- local script re-derived the number from raw logs or raw trajectory
- convergence status is `converged` or explicitly accepted as `numerically_stationary_unconverged`
- protocol is named: fixed-geometry single-point, MLFF self-NEB, or DFT NEB refinement
- reference barrier is path-matched, not copied from another path
