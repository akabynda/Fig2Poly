#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
cd "$PROJECT_ROOT"

DATA_JOB="${1:?usage: submit_curve_benchmark.sh DATA_PREP_JOB_ID SETUP_JOB_ID}"
SETUP_JOB="${2:?usage: submit_curve_benchmark.sh DATA_PREP_JOB_ID SETUP_JOB_ID}"
LINEFORMER_JOB=$(sbatch --parsable --dependency="afterok:${DATA_JOB}:${SETUP_JOB}" server/slurm/benchmark_lineformer.sbatch)
MASKDINO_JOB=$(sbatch --parsable --dependency="afterok:${DATA_JOB}" server/slurm/benchmark_maskdino.sbatch)
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${LINEFORMER_JOB}:${MASKDINO_JOB}" server/slurm/evaluate_curve_models.sbatch)
printf 'LineFormer: %s\nMaskDINO: %s\nEvaluation: %s\n' "$LINEFORMER_JOB" "$MASKDINO_JOB" "$EVAL_JOB"
