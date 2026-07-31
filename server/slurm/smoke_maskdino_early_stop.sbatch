#!/bin/bash
#SBATCH --job-name=maskdino_early
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=smoke_maskdino_early_stop_%j.out
#SBATCH --error=smoke_maskdino_early_stop_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/maskdino_early_stop_smoke_${SLURM_JOB_ID}"

"$PY" -m training.train_maskdino \
  --maskdino-root "$MASKDINO_ROOT" \
  --dataset "$DATA_ROOT/coco_smoke_empty" \
  --output "$OUTPUT" \
  --variant r50 \
  --weights "$CACHE_ROOT/weights/maskdino_r50.pth" \
  --epochs 4 \
  --global-batch 2 \
  --num-gpus 1 \
  --workers 0 \
  --image-size 768 \
  --amp \
  --checkpoint-period 1 \
  --early-stopping-patience 1 \
  --early-stopping-metric segm/AP \
  --early-stopping-min-delta 1000000000

test -s "$OUTPUT/model_best.pth"
"$PY" -c \
  "import json; p=json.load(open('$OUTPUT/early_stopping.json')); assert p['stopped'] is True, p"
