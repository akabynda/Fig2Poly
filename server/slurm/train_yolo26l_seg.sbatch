#!/bin/bash
#SBATCH --job-name=yolo26l_seg
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=train_yolo26l_seg_%j.out
#SBATCH --error=train_yolo26l_seg_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/yolo/bin/python"
OUTPUT="$RUNS_ROOT/yolo26l_seg_1024"
LAST="$OUTPUT/weights/last.pt"
record_run_metadata "$OUTPUT"

requeue_on_timeout() {
  echo "Received SLURM timeout signal; requeueing from the latest epoch checkpoint"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_on_timeout USR1

if [[ -f "$LAST" ]]; then
  "$PY" -m training.train_yolo \
    --resume "$LAST" \
    --device 0 \
    --workers "${SLURM_CPUS_PER_TASK:-12}" &
else
  "$PY" -m training.train_yolo \
    --model yolo26l-seg.pt \
    --data "$DATA_ROOT/synthetic_200k/curve_yolo.yaml" \
    --project "$RUNS_ROOT" \
    --name yolo26l_seg_1024 \
    --epochs 50 \
    --imgsz 1024 \
    --batch -1 \
    --workers "${SLURM_CPUS_PER_TASK:-12}" \
    --device 0 \
    --patience 7 &
fi
trainer_pid=$!
wait "$trainer_pid"
