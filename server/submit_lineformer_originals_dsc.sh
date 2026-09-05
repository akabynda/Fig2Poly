#!/usr/bin/env bash
# One CPU preparation, a two-GPU smoke test, then two sequential two-GPU trainings.
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$FIG2POLY_ROOT"
STATE="$RUNS_ROOT/lineformer_originals_dsc_exact_100k_v1"
mkdir -p "$STATE"
exec 9>"$STATE/submission.lock"
flock -n 9 || { echo "Another submission is in progress" >&2; exit 2; }

submit_stage() {
  local stage="$1"
  shift
  local receipt="$STATE/$stage.jobid"
  if [[ -s "$receipt" ]]; then
    cat "$receipt"
    return
  fi
  local job_id
  job_id=$(sbatch --parsable "$@")
  job_id="${job_id%%;*}"
  [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID: $job_id" >&2; exit 2; }
  printf '%s\n' "$job_id" > "$receipt.tmp"
  mv "$receipt.tmp" "$receipt"
  printf '%s\n' "$job_id"
}

# Optional already-submitted Adobe download, to reuse preparation in progress.
download_dependency=()
if [[ -n "${LINEFORMER_ADOBE_JOB:-}" ]]; then
  [[ "$LINEFORMER_ADOBE_JOB" =~ ^[0-9]+$ ]] || exit 2
  download_dependency=(--dependency="afterok:$LINEFORMER_ADOBE_JOB")
fi
prep=$(submit_stage prepare "${download_dependency[@]}" server/slurm/prepare_lineformer_originals_dsc.sbatch)
smoke=$(submit_stage smoke --dependency="afterok:$prep" server/slurm/smoke_lineformer_originals_dsc.sbatch)
mask2former=$(submit_stage mask2former --job-name=m2f_lf_dsc_exact \
  --dependency="afterok:$smoke" server/slurm/train_lineformer_originals_dsc.sbatch mask2former)
# Both models require successful smoke; the second releases only after the first
# has left its allocation, including a first-model failure that needs inspection.
maskdino=$(submit_stage maskdino --job-name=md_r50_lf_dsc_exact \
  --dependency="afterok:$smoke,afterany:$mask2former" \
  server/slurm/train_lineformer_originals_dsc.sbatch maskdino)
printf 'prepare=%s\nsmoke=%s\nmask2former=%s\nmaskdino=%s\nstate=%s\n' \
  "$prep" "$smoke" "$mask2former" "$maskdino" "$STATE"
