#!/bin/bash
# Prepare and submit the first long restartable QE NEB segment for MigrationBench 1-7.

set -euo pipefail

echo "Deprecated historical recipe: use ../continue_qe_neb.py or ../launch_qe_neb_from_scf_warmups.py." >&2
exit 64

WALLTIME="${1:-24:00:00}"
MEM="${2:-160G}"
SAFETY_SECONDS="${3:-1800}"

LOCAL_RESTART_DIR="work/revision1_neb_restart_design"
REMOTE_PIPE="\$HOME/revision1_pipeline_20260909"
RUN_ROOT="/scratch16/pclancy3/yi/revision1_migrationbench_runs"
INPUT_ROOT="/scratch16/pclancy3/yi/revision1_migrationbench_inputs/mb_qe17_mace_r1_s42"

HIST_ROOT="/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3"
DFT_ROOT="${HIST_ROOT}/3.neb/1.vdw_corr_DFT_D3"
MLFF_RUN="${RUN_ROOT}/mb_mlff_1_7c_s42_30763474/mlff_1-7_cont_s42"

rsync -av \
  "${LOCAL_RESTART_DIR}/qe_neb_prepare_restart.py" \
  "${LOCAL_RESTART_DIR}/submit_slurm_qe_neb_restart.sh" \
  "${LOCAL_RESTART_DIR}/submit_qe_neb_restart_job.sh" \
  rockfish:"${REMOTE_PIPE}/scripts/migrationbench/"

ssh rockfish "bash --noprofile --norc -s" <<REMOTE
set -euo pipefail
cd ${REMOTE_PIPE}
mkdir -p "${INPUT_ROOT}"

python scripts/migrationbench/qe_engine_template_from_pw.py \
  --input "${DFT_ROOT}/1-7/pw_1.in" \
  --output "${INPUT_ROOT}/qe_engine_template.in"

python scripts/migrationbench/qe_neb_from_images.py \
  --images "${MLFF_RUN}/mlff_neb_images.extxyz" \
  --engine-template "${INPUT_ROOT}/qe_engine_template.in" \
  --output "${INPUT_ROOT}/neb.in" \
  --manifest "${INPUT_ROOT}/neb.manifest.json" \
  --path-id 1-7 \
  --path-family in_gap_by_prior_site_definition \
  --measurement-purpose dft_refinement_of_mace_preconditioned_in_gap_path \
  --source-neb-output "${DFT_ROOT}/1-7/neb.out" \
  --source-mlff-manifest "${MLFF_RUN}/mlff_neb_manifest.json" \
  --nstep-path 1000 \
  --path-thr 0.03 \
  --opt-scheme broyden \
  --ci-scheme auto \
  --minimum-image

bash scripts/migrationbench/submit_qe_neb_restart_job.sh \
  mb_qe17_mace_r1_s42 \
  "${INPUT_ROOT}/neb.in" \
  - \
  2 \
  "${WALLTIME}" \
  "${MEM}" \
  "${SAFETY_SECONDS}" \
  from_scratch
REMOTE
