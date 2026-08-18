#!/bin/bash
#SBATCH --job-name=bench_maskdino
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=benchmark_maskdino_%j.out
#SBATCH --error=benchmark_maskdino_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/curve_model_benchmark_v1"
mkdir -p "$OUTPUT"
record_run_metadata "$OUTPUT"

"$PY" -m training.benchmark_maskdino \
  --maskdino-root "$MASKDINO_ROOT" \
  --weights "$RUNS_ROOT/maskdino_r50_lineex_balanced_v5/model_best.pth" \
  --variant r50 \
  --image-size 768 \
  --max-size 2048 \
  --output "$OUTPUT/maskdino.sqlite" \
  --min-threshold 0.15 \
  --dataset "ub_pmc22=manifest=$DATA_ROOT/lineformer_benchmarks/ub_pmc22_line_only/test.jsonl" \
  --dataset "balanced_v5_test=manifest=$DATA_ROOT/combined_lineex_balanced_v5/test.jsonl"
