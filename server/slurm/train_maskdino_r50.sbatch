#!/bin/bash
#SBATCH --job-name=maskdino_r50
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --requeue
#SBATCH --output=train_maskdino_r50_%j.out
#SBATCH --error=train_maskdino_r50_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/maskdino_r50"
record_run_metadata "$OUTPUT"
resume=()
[[ -f "$OUTPUT/last_checkpoint" ]] && resume=(--resume)

requeue_on_timeout() {
  echo "Received SLURM timeout signal; requeueing from the latest checkpoint"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_on_timeout TERM

"$PY" -m training.train_maskdino \
  --maskdino-root "$MASKDINO_ROOT" \
  --dataset "$DATA_ROOT/coco_synthetic_200k" \
  --output "$OUTPUT" \
  --variant r50 \
  --weights "$CACHE_ROOT/weights/maskdino_r50.pth" \
  --epochs 50 \
  --global-batch 2 \
  --num-gpus 1 \
  --workers "${SLURM_CPUS_PER_TASK:-12}" \
  "${resume[@]}" &
trainer_pid=$!
wait "$trainer_pid"
