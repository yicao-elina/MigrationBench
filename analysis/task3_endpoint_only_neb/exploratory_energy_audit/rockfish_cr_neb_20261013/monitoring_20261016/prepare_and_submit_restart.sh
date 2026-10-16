#!/bin/bash -l
# Prepare isolated Rockfish restart directories and submit exactly one job per path.
# This script is intentionally idempotent: an existing job_map row is never resubmitted.

set -euo pipefail

ROOT="/scratch16/pclancy3/yi/MigrationBench_rockfish_cr_neb_20261013"
RESTART_ROOT="$ROOT/restarts_20261016"
WRAPPER="${RESTART_ROOT}/rockfish_cr_neb_restart.slurm"
JOB_MAP="$RESTART_ROOT/restart_job_map.tsv"
MANIFEST="$RESTART_ROOT/restart_manifest.tsv"

mkdir -p "$RESTART_ROOT"

if [[ ! -f "$WRAPPER" ]]; then
  echo "missing deployed wrapper: $WRAPPER" >&2
  exit 2
fi

printf 'path_class\tprevious_job\tsource_dir\trestart_dir\tprevious_last_iteration\tsource_neb_out_sha256\tsource_path_sha256\tstatus\tnew_job\n' > "${MANIFEST}.tmp"
if [[ -f "$MANIFEST" ]]; then
  tail -n +2 "$MANIFEST" >> "${MANIFEST}.tmp"
fi
mv "${MANIFEST}.tmp" "$MANIFEST"

printf 'path_class\tprevious_job\trestart_dir\tnew_job\tsubmit_time\n' > "${JOB_MAP}.tmp"
if [[ -f "$JOB_MAP" ]]; then
  tail -n +2 "$JOB_MAP" >> "${JOB_MAP}.tmp"
fi
mv "${JOB_MAP}.tmp" "$JOB_MAP"

prepare_one() {
  local path_class="$1"
  local previous_job="$2"
  local source_dir="$3"
  local restart_dir="$4"
  local nodes="$5"
  local previous_iteration

  if grep -Fq "${path_class}"$'\t'"${previous_job}"$'\t' "$JOB_MAP"; then
    echo "already mapped; skip resubmission: ${path_class} ${previous_job}"
    return 0
  fi

  [[ -d "$source_dir" ]] || { echo "missing source: $source_dir" >&2; exit 3; }
  [[ -f "$source_dir/neb.out" ]] || { echo "missing source neb.out: $source_dir" >&2; exit 3; }
  [[ -f "$source_dir/sb2te3.path" ]] || { echo "missing latest path: $source_dir" >&2; exit 3; }
  grep -q "JOB DONE" "$source_dir/neb.out" || { echo "source is not a completed NEB: $source_dir" >&2; exit 4; }

  if [[ -e "$restart_dir" ]]; then
    echo "restart directory already exists without a job-map row: $restart_dir" >&2
    echo "refusing to overwrite or silently resubmit" >&2
    exit 5
  fi
  mkdir -p "$restart_dir"

  # Copy only restart metadata/checkpoints. The original 92-GB out/ is left in place;
  # the continuation gets a fresh out/ and therefore cannot mutate the first run.
  for f in neb.in pw_1.in pw_2.in sb2te3.broyden sb2te3.crd sb2te3.int sb2te3.dat sb2te3.xyz sb2te3.axsf; do
    [[ -f "$source_dir/$f" ]] && cp -p "$source_dir/$f" "$restart_dir/$f"
  done
  cp -p "$source_dir"/sb2te3.path* "$restart_dir/"
  cp -p "$source_dir/neb.out" "$restart_dir/neb_previous.out"
  for f in "$source_dir"/slurm-*.out; do
    [[ -f "$f" ]] && cp -p "$f" "$restart_dir/previous_$(basename "$f")"
  done

  # Force a continuation input even for the one historical path that began from scratch.
  sed -E "s/^([[:space:]]*restart_mode[[:space:]]*=).*/\\1 'restart'/" \
    "$restart_dir/neb.in" > "$restart_dir/neb.in.tmp"
  mv "$restart_dir/neb.in.tmp" "$restart_dir/neb.in"

  previous_iteration="$(grep -E 'iteration[[:space:]]+[0-9]+' "$source_dir/neb.out" | tail -1 | sed -E 's/.*iteration[[:space:]]+([0-9]+).*/\1/' || true)"
  {
    echo "path_class=$path_class"
    echo "previous_job=$previous_job"
    echo "source_dir=$source_dir"
    echo "restart_dir=$restart_dir"
    echo "previous_last_iteration=$previous_iteration"
    echo "source_out_preserved_at=$source_dir/neb.out"
    echo "source_outdir_preserved_at=$source_dir/out"
    echo "new_outdir=$restart_dir/out"
    echo "restart_mode=$(grep -E '^[[:space:]]*restart_mode' "$restart_dir/neb.in" | head -1)"
    echo "prepared_at=$(date -Is)"
  } > "$restart_dir/restart_lineage.txt"

  local out_sha path_sha
  out_sha="$(sha256sum "$source_dir/neb.out" | awk '{print $1}')"
  path_sha="$(sha256sum "$source_dir/sb2te3.path" | awk '{print $1}')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tPREPARED\t\n' \
    "$path_class" "$previous_job" "$source_dir" "$restart_dir" "$previous_iteration" "$out_sha" "$path_sha" >> "$MANIFEST"

  (cd "$restart_dir" && sbatch --parsable --nodes="$nodes" --ntasks-per-node=48 --job-name="cr-${path_class}-r01" "$WRAPPER") > "$restart_dir/submit_result.txt"
  local new_job
  new_job="$(tr -d '[:space:]' < "$restart_dir/submit_result.txt")"
  [[ "$new_job" =~ ^[0-9]+$ ]] || { echo "unexpected sbatch result: $new_job" >&2; exit 6; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$path_class" "$previous_job" "$restart_dir" "$new_job" "$(date -Is)" >> "$JOB_MAP"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tSUBMITTED\t%s\n' \
    "$path_class" "$previous_job" "$source_dir" "$restart_dir" "$previous_iteration" "$out_sha" "$path_sha" "$new_job" >> "$MANIFEST"
  echo "$path_class submitted as $new_job"
}

prepare_one "within_QLs" "31916908" "$ROOT/2L-Cu__neb-QLs" "$RESTART_ROOT/01_within_QLs" "2"
prepare_one "gap_to_QL" "31916909" "$ROOT/2L-Cu__neb-gap2QL" "$RESTART_ROOT/02_gap_to_QL" "2"
prepare_one "within_gap" "31916910" "$ROOT/2L-Cu__neb-gap-diffuse" "$RESTART_ROOT/03_within_gap" "1"

echo "restart root: $RESTART_ROOT"
echo "job map: $JOB_MAP"
