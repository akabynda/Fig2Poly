#!/bin/bash
#SBATCH --job-name=eval_curve_models
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=evaluate_curve_models_%j.out
#SBATCH --error=evaluate_curve_models_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/curve_model_benchmark_v1"

"$PY" -m training.evaluate_curve_benchmark \
  --lineformer-db "$OUTPUT/lineformer.sqlite" \
  --maskdino-db "$OUTPUT/maskdino.sqlite" \
  --output "$OUTPUT/csv" \
  --threshold 0.30 \
  --rescue-threshold 0.15 \
  --real-test-counts "$PROJECT_ROOT/configs/real_test_counts.json" \
  --dataset "lineex=manifest=$DATA_ROOT/public_lineex_100k_v5/val.jsonl" \
  --dataset "ub_pmc22=manifest=$DATA_ROOT/lineformer_benchmarks/ub_pmc22_line_only/test.jsonl" \
  --dataset "adobe_synth19=manifest=$DATA_ROOT/lineformer_benchmarks/adobe_synth19/test.jsonl" \
  --dataset "real_test=images=$PROJECT_ROOT/real_test"

ls -lh "$OUTPUT/csv/per_image_metrics.csv" \
  "$OUTPUT/csv/summary_metrics.csv" \
  "$OUTPUT/csv/paper_reference_metrics.csv"
