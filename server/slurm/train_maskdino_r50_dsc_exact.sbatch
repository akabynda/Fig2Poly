#!/bin/bash
#SBATCH --job-name=train_md_r50_dsc
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=train_maskdino_r50_dsc_%j.out
#SBATCH --error=train_maskdino_r50_dsc_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/maskdino_r50_dsc_exact_200k_earlystop"
record_run_metadata "$OUTPUT"
resume=()
[[ -f "$OUTPUT/last_checkpoint" ]] && resume=(--resume)

trainer_pid=""
requeue_on_timeout() {
  trap - USR1
  echo "Received timeout signal; stopping MaskDINO before requeue"
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
  --dataset "$DATA_ROOT/coco_lineformer_dsc_exact" \
  --output "$OUTPUT" \
  --variant r50 \
  --weights "$CACHE_ROOT/weights/maskdino_r50.pth" \
  --epochs 50 \
  --max-iter 200000 \
  --global-batch 16 \
  --num-gpus 2 \
  --workers 8 \
  --image-size 768 \
  --base-lr 2e-5 \
  --amp \
  --checkpoint-period 20000 \
  --eval-period 20000 \
  --curve-metrics \
  --curve-score-threshold 0.30 \
  --early-stopping-patience 3 \
  --early-stopping-metric curve/score_6b \
  --early-stopping-min-delta 0.05 \
  "${resume[@]}" &
trainer_pid=$!
wait "$trainer_pid"

test -s "$OUTPUT/model_best.pth"
