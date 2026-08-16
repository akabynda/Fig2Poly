#!/bin/bash
#SBATCH --job-name=lineformer_real
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=predict_lineformer_real_test_%j.out
#SBATCH --error=predict_lineformer_real_test_%j.err

set -Eeuo pipefail
USER_NAME="$(id -un)"
PROJECT_ROOT="/mnt/tank/scratch/$USER_NAME/Fig2Poly"
STORAGE="/mnt/tank/scratch/$USER_NAME"

cd "$PROJECT_ROOT"
"$STORAGE/.venvs/lineformer/bin/python" -m training.predict_lineformer \
  --lineformer-root "$STORAGE/LineFormer" \
  --weights "$STORAGE/LineFormer_weights/iter_3000.pth" \
  --input "$PROJECT_ROOT/real_test" \
  --output "$STORAGE/Fig2Poly_runs/lineformer_official_real_test" \
  --threshold 0.3
