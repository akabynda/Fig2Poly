#!/bin/bash
#SBATCH --job-name=merge_curve_all6
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=merge_curve_models_all6_%j.out
#SBATCH --error=merge_curve_models_all6_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

BASE="$RUNS_ROOT/lineformer_dsc_dilated_train_val_test/comparison_all_5"
MASKDINO="$RUNS_ROOT/maskdino_r50_dsc_exact_test"
OUTPUT="$RUNS_ROOT/lineformer_dsc_dilated_train_val_test/comparison_all_6"
mkdir -p "$OUTPUT"

cp "$BASE/comparison_all_5.csv" "$OUTPUT/comparison_all_6.csv"
grep maskdino_r50_dsc_exact \
  "$MASKDINO/csv_vs_original/lineformer_article_metrics.csv" \
  >> "$OUTPUT/comparison_all_6.csv"

cp "$BASE/model_legend.csv" "$OUTPUT/model_legend.csv"
printf '%s\n' \
  "maskdino_r50_dsc_exact,stock_coco_finetuned,1,1,exact,$RUNS_ROOT/maskdino_r50_dsc_exact_200k_earlystop/model_best.pth" \
  >> "$OUTPUT/model_legend.csv"

ls -lh "$OUTPUT/comparison_all_6.csv" "$OUTPUT/model_legend.csv"
