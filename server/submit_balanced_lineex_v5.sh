#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
public_job=$(sbatch --parsable "$ROOT/server/slurm/prepare_lineex_public_v5.sbatch")
synthetic_job=$(sbatch --parsable --dependency="afterok:$public_job" \
  "$ROOT/server/slurm/generate_balanced_synthetic_v5.sbatch")
assemble_job=$(sbatch --parsable --dependency="afterok:$synthetic_job" \
  "$ROOT/server/slurm/assemble_balanced_lineex_v5.sbatch")
printf 'public_job=%s\nsynthetic_job=%s\nassemble_job=%s\n' \
  "$public_job" "$synthetic_job" "$assemble_job"
