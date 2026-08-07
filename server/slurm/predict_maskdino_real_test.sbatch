#!/bin/bash
#SBATCH --job-name=maskdino_real
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=predict_maskdino_real_test_%j.out
#SBATCH --error=predict_maskdino_real_test_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
RUN="$RUNS_ROOT/maskdino_r50_768_b8_e10_early"
OUTPUT="$RUNS_ROOT/maskdino_r50_real_test_best"

"$PY" -m training.predict_maskdino \
  --maskdino-root "$MASKDINO_ROOT" \
  --weights "$RUN/model_best.pth" \
  --input "$PROJECT_ROOT/real_test" \
  --output "$OUTPUT" \
  --variant r50 \
  --threshold 0.25 \
  --image-size 768
