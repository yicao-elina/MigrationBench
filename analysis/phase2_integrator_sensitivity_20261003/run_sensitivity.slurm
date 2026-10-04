#!/bin/bash -l
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@120
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
set -euo pipefail
ROOT=/scratch16/pclancy3/yi/phase2_integrator_sensitivity_20261003
SCRIPT=$ROOT/handoff_sensitivity.py
INPUT_ROOT=/scratch16/pclancy3/yi/phase2_handoff_md_20261002/snapshots/from_scratch_source
MODEL_ROOT=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer
PYTHONPATH=/data/pclancy3/yi/.python_pkgs:/data/pclancy3/yi/src
PY=/data/pclancy3/yi/.conda/envs/mace/bin/python
export PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 48 independent runs: 2 snapshots x 4 models x 6 settings.  NVE and
# Langevin(friction=0) are kept as distinct labels for the requested bath
# sensitivity audit; all runs retain the same 2 ps/10-step output protocol.
SNAPSHOTS=(461800 462000)
MODELS=(foundation-omat from-scratch naive-fine-tuning multi_T-fine-tuning)
MODEL_FILES=(
  "$MODEL_ROOT/MACE-omat/finetuned_MACE_compiled.model"
  "$MODEL_ROOT/MACE_models_l1_0802/mace_l1_0802_compiled.model"
  "$MODEL_ROOT/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model"
  "$MODEL_ROOT/MACE-multihead_Multi_T/finetuned_MACE_multihead0804.model"
)
SETTINGS=(nve_dt0p5 nve_dt1p0 nve_dt2p0 langevin_f0_dt1p0 langevin_f0p001_dt1p0 langevin_f0p01_dt1p0)
idx=${SLURM_ARRAY_TASK_ID:?}
setting_idx=$((idx % 6)); model_idx=$(((idx / 6) % 4)); snap_idx=$((idx / 24))
snap=${SNAPSHOTS[$snap_idx]}; model=${MODELS[$model_idx]}; setting=${SETTINGS[$setting_idx]}
case "$setting" in
  nve_dt0p5) integrator=nve; friction=0.0; dt=0.5 ;;
  nve_dt1p0) integrator=nve; friction=0.0; dt=1.0 ;;
  nve_dt2p0) integrator=nve; friction=0.0; dt=2.0 ;;
  langevin_f0_dt1p0) integrator=langevin; friction=0.0; dt=1.0 ;;
  langevin_f0p001_dt1p0) integrator=langevin; friction=0.001; dt=1.0 ;;
  langevin_f0p01_dt1p0) integrator=langevin; friction=0.01; dt=1.0 ;;
esac
case "$model" in
  foundation-omat) model_file=${MODEL_FILES[0]} ;;
  from-scratch) model_file=${MODEL_FILES[1]} ;;
  naive-fine-tuning) model_file=${MODEL_FILES[2]} ;;
  multi_T-fine-tuning) model_file=${MODEL_FILES[3]} ;;
esac
input=$INPUT_ROOT/snapshot_${snap}.extxyz
output=$ROOT/runs/snapshot_${snap}/model_${model}/setting_${setting}
exec "$PY" "$SCRIPT" --input "$input" --model "$model_file" --output "$output" \
  --start-step "$snap" --steps 2000 --interval 10 --temperature 600 \
  --friction "$friction" --timestep-fs "$dt" --integrator "$integrator" --device cuda
