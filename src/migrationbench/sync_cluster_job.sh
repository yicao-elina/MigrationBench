#!/bin/bash
# Sync a compact Rockfish job directory back into Revision1/data_processed/cluster.

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 JOB_DIR_NAME [REMOTE_RUN_ROOT] [LOCAL_CLUSTER_ROOT]" >&2
  exit 2
fi

JOB_DIR_NAME="$1"
REMOTE_RUN_ROOT="${2:-/scratch16/pclancy3/yi/revision1_migrationbench_runs}"
LOCAL_CLUSTER_ROOT="${3:-data_processed/cluster}"

mkdir -p "${LOCAL_CLUSTER_ROOT}/${JOB_DIR_NAME}"
ssh rockfish "cd '${REMOTE_RUN_ROOT}' && tar czf - '${JOB_DIR_NAME}'" \
  | tar xzf - -C "${LOCAL_CLUSTER_ROOT}/${JOB_DIR_NAME}"

echo "${LOCAL_CLUSTER_ROOT}/${JOB_DIR_NAME}"
