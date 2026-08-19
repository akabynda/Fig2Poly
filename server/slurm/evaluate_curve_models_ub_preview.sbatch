#!/bin/bash
#SBATCH --job-name=eval_ub_preview
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=evaluate_curve_models_ub_preview_%j.out
#SBATCH --error=evaluate_curve_models_ub_preview_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
BENCHMARK="$RUNS_ROOT/curve_model_benchmark_v1"
OUTPUT="$BENCHMARK/csv_ub_preview"

mkdir -p "$OUTPUT"
record_run_metadata "$OUTPUT"

cd "$PROJECT_ROOT"
"$PY" training/evaluate_curve_benchmark.py \
  --lineformer-db "$BENCHMARK/lineformer.sqlite" \
  --maskdino-db "$BENCHMARK/maskdino.sqlite" \
  --output "$OUTPUT" \
  --threshold 0.30 \
  --rescue-threshold 0.15 \
  --dataset "ub_pmc22=manifest=$DATA_ROOT/lineformer_benchmarks/ub_pmc22_line_only/test.jsonl"

ls -lh "$OUTPUT/per_image_metrics.csv" \
  "$OUTPUT/summary_metrics.csv" \
  "$OUTPUT/paper_reference_metrics.csv" \
  "$OUTPUT/protocol.json"
