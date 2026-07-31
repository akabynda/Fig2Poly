#!/bin/bash
#SBATCH --job-name=maskdino_bench
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmark_maskdino_r50_%j.out
#SBATCH --error=benchmark_maskdino_r50_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/maskdino_r50_benchmark_${SLURM_JOB_ID}"

"$PY" -m training.train_maskdino \
  --maskdino-root "$MASKDINO_ROOT" \
  --dataset "$DATA_ROOT/coco_synthetic_200k" \
  --output "$OUTPUT" \
  --variant r50 \
  --weights "$CACHE_ROOT/weights/maskdino_r50.pth" \
  --epochs 1 \
  --global-batch 8 \
  --num-gpus 1 \
  --workers "${SLURM_CPUS_PER_TASK:-12}" \
  --image-size 768 \
  --amp \
  --max-iter 100 \
  --checkpoint-period 100 \
  --eval-period 0
