#!/bin/bash
#SBATCH --job-name=maskdino_smoke
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=smoke_maskdino_r50_%j.out
#SBATCH --error=smoke_maskdino_r50_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/maskdino_r50_smoke_empty_${SLURM_JOB_ID}"
"$PY" -m training.train_maskdino \
  --maskdino-root "$MASKDINO_ROOT" \
  --dataset "$DATA_ROOT/coco_smoke_empty" \
  --output "$OUTPUT" \
  --variant r50 \
  --weights "$CACHE_ROOT/weights/maskdino_r50.pth" \
  --epochs 1 \
  --global-batch 1 \
  --num-gpus 1 \
  --workers 0 \
  --max-iter 1 \
  --checkpoint-period 1 \
  --eval-period 0
