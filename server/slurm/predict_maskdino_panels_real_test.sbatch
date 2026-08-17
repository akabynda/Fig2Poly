#!/bin/bash
#SBATCH --job-name=maskdino_panels
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=predict_maskdino_panels_real_test_%j.out
#SBATCH --error=predict_maskdino_panels_real_test_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
RUN="$RUNS_ROOT/maskdino_r50_lineex_balanced_v5"
OUTPUT="$RUNS_ROOT/maskdino_r50_lineex_balanced_v5_panels_centerline"

record_run_metadata "$OUTPUT"
"$PY" -m training.predict_maskdino_panels \
  --maskdino-root "$MASKDINO_ROOT" \
  --weights "$RUN/model_best.pth" \
  --input "$PROJECT_ROOT/real_test" \
  --output "$OUTPUT" \
  --variant r50 \
  --thresholds 0.15 0.25 0.30 0.40 0.50 0.70 \
  --image-size 768 \
  --max-size 2048
