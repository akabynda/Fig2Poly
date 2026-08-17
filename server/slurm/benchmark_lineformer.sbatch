#!/bin/bash
#SBATCH --job-name=bench_lineformer
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=benchmark_lineformer_%j.out
#SBATCH --error=benchmark_lineformer_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/lineformer/bin/python"
OUTPUT="$RUNS_ROOT/curve_model_benchmark_v1"
mkdir -p "$OUTPUT"
record_run_metadata "$OUTPUT"

"$PY" -m training.benchmark_lineformer \
  --lineformer-root "$PROJECT_ROOT/third_party/LineFormer" \
  --weights "$CACHE_ROOT/weights/lineformer_iter_3000.pth" \
  --output "$OUTPUT/lineformer.sqlite" \
  --min-threshold 0.15 \
  --dataset "lineex=manifest=$DATA_ROOT/public_lineex_100k_v5/val.jsonl" \
  --dataset "ub_pmc22=manifest=$DATA_ROOT/lineformer_benchmarks/ub_pmc22_line_only/test.jsonl" \
  --dataset "adobe_synth19=manifest=$DATA_ROOT/lineformer_benchmarks/adobe_synth19/test.jsonl" \
  --dataset "real_test=images=$PROJECT_ROOT/real_test"
