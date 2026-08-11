#!/bin/bash
#SBATCH --job-name=maskdino_r50
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
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

trainer_pid=""
requeue_on_timeout() {
  # The signal is delivered to the batch shell only (SBATCH --signal=B:...).
  # Stop its background trainer before requeueing, otherwise the old process
  # may keep the GPU allocation alive and collide with the resumed process.
  trap - USR1
  echo "Received SLURM timeout signal; stopping trainer before requeue"
  if [[ -n "$trainer_pid" ]] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill -TERM "$trainer_pid" 2>/dev/null || true
    wait "$trainer_pid" || true
  fi
  echo "Requeueing from the latest checkpoint"
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
