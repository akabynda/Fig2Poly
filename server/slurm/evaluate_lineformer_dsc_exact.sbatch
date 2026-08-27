#!/bin/bash
#SBATCH --job-name=eval_lf_exact
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=evaluate_lineformer_dsc_exact_%j.out
#SBATCH --error=evaluate_lineformer_dsc_exact_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/lineformer/bin/python"
OLD_OUTPUT="$RUNS_ROOT/lineformer_dsc_test_comparison"
OUTPUT="$RUNS_ROOT/lineformer_dsc_exact_test"

"$PY" -m training.compare_lineformer_checkpoints \
  --classic-db "$OLD_OUTPUT/finetuned.sqlite" \
  --classic-label dilated_train \
  --finetuned-db "$OUTPUT/exact.sqlite" \
  --finetuned-label exact_train \
  --output "$OUTPUT/csv" \
  --threshold 0.30 \
  --dataset "dsc_test=manifest=$DATA_ROOT/dataset_dsc/test.jsonl"

ls -lh "$OUTPUT/csv/lineformer_article_metrics.csv" \
  "$OUTPUT/csv/summary_metrics.csv" \
  "$OUTPUT/csv/per_image_metrics.csv"
