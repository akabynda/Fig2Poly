#!/bin/bash
#SBATCH --job-name=maskdino_curve_smoke
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=smoke_maskdino_curve_v5_%j.out
#SBATCH --error=smoke_maskdino_curve_v5_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/maskdino_curve_v5_smoke_${SLURM_JOB_ID}"

"$PY" -m training.train_maskdino \
  --maskdino-root "$MASKDINO_ROOT" \
  --dataset "$DATA_ROOT/coco_smoke_empty" \
  --output "$OUTPUT" \
  --variant r50 \
  --weights "$RUNS_ROOT/maskdino_r50_768_b8_e10_early/model_best.pth" \
  --epochs 1 \
  --global-batch 12 \
  --num-gpus 1 \
  --workers 0 \
  --image-size 768 \
  --base-lr 2e-5 \
  --amp \
  --max-iter 2 \
  --checkpoint-period 1 \
  --eval-period 1 \
  --curve-loss-weight 1.0 \
  --curve-metrics

test -s "$OUTPUT/model_final.pth"
test -s "$OUTPUT/inference/chart_line_metrics.json"
