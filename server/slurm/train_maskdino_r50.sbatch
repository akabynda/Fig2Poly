#!/bin/bash
#SBATCH --job-name=maskdino_r50
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=train_maskdino_r50_%j.out
#SBATCH --error=train_maskdino_r50_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/maskdino_r50_lineex_balanced_v5"
record_run_metadata "$OUTPUT"
resume=()
[[ -f "$OUTPUT/last_checkpoint" ]] && resume=(--resume)

requeue_on_timeout() {
  echo "Received SLURM timeout signal; requeueing from the latest checkpoint"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_on_timeout USR1

"$PY" -m training.train_maskdino \
  --maskdino-root "$MASKDINO_ROOT" \
  --dataset "$DATA_ROOT/coco_lineex_balanced_v5" \
  --output "$OUTPUT" \
  --variant r50 \
  --weights "$RUNS_ROOT/maskdino_r50_768_b8_e10_early/model_best.pth" \
  --epochs 10 \
  --global-batch 12 \
  --num-gpus 1 \
  --workers "${SLURM_CPUS_PER_TASK:-12}" \
  --image-size 768 \
  --base-lr 2e-5 \
  --amp \
  --curve-loss-weight 1.0 \
  --curve-metrics \
  --early-stopping-patience 2 \
  --early-stopping-metric curve/score_6b \
  --early-stopping-min-delta 0.05 \
  "${resume[@]}" &
trainer_pid=$!
wait "$trainer_pid"
