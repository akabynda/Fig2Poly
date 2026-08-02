#!/bin/bash
#SBATCH --job-name=prepare_yolo
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=prepare_yolo_%j.out
#SBATCH --error=prepare_yolo_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/yolo/bin/python"
DATASET="$DATA_ROOT/synthetic_200k"
"$PY" -m training.convert_yolo \
  --dataset "$DATASET" \
  --workers "${SLURM_CPUS_PER_TASK:-16}" \
  --subset-train 2000 \
  --subset-val 500
