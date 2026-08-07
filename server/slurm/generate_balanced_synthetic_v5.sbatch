#!/bin/bash
#SBATCH --job-name=synthetic_bal_v5
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=generate_balanced_synthetic_v5_%j.out
#SBATCH --error=generate_balanced_synthetic_v5_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/yolo/bin/python"

"$PY" -m training.generate_matching_synthetic \
  --public "$DATA_ROOT/public_lineex_100k_v5" \
  --config "$PROJECT_ROOT/configs/balanced_lineex_v5.json" \
  --output "$DATA_ROOT/synthetic_lineex_matched_v5" \
  --workers "${SLURM_CPUS_PER_TASK:-16}" \
  --resume
