#!/bin/bash
#SBATCH --job-name=eval_md_r50_dsc
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=evaluate_maskdino_r50_dsc_%j.out
#SBATCH --error=evaluate_maskdino_r50_dsc_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/lineformer/bin/python"
ORIGINAL="$RUNS_ROOT/lineformer_dsc_test_comparison/classic.sqlite"
OUTPUT="$RUNS_ROOT/maskdino_r50_dsc_exact_test"

"$PY" -m training.compare_lineformer_checkpoints \
  --classic-db "$ORIGINAL" \
  --classic-label original_lineformer \
  --finetuned-db "$OUTPUT/maskdino.sqlite" \
  --finetuned-label maskdino_r50_dsc_exact \
  --output "$OUTPUT/csv_vs_original" \
  --threshold 0.30 \
  --dataset "dsc_test=manifest=$DATA_ROOT/dataset_dsc/test.jsonl"

ls -lh \
  "$OUTPUT/csv_vs_original/lineformer_article_metrics.csv" \
  "$OUTPUT/csv_vs_original/summary_metrics.csv" \
  "$OUTPUT/csv_vs_original/per_image_metrics.csv"
