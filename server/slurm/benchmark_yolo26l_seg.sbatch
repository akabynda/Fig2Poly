#!/bin/bash
#SBATCH --job-name=yolo26l_bench
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=benchmark_yolo26l_seg_%j.out
#SBATCH --error=benchmark_yolo26l_seg_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/yolo/bin/python"
"$PY" -m training.train_yolo \
  --model yolo26l-seg.pt \
  --data "$DATA_ROOT/synthetic_200k/curve_yolo_subset.yaml" \
  --project "$RUNS_ROOT" \
  --name "yolo26l_seg_benchmark_${SLURM_JOB_ID}" \
  --epochs 1 \
  --imgsz 1024 \
  --batch -1 \
  --workers "${SLURM_CPUS_PER_TASK:-12}" \
  --device 0 \
  --patience 0
