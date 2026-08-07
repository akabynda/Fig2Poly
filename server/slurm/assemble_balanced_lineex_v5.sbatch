#!/bin/bash
#SBATCH --job-name=assemble_bal_v5
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=assemble_balanced_lineex_v5_%j.out
#SBATCH --error=assemble_balanced_lineex_v5_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/yolo/bin/python"
COMBINED="$DATA_ROOT/combined_lineex_balanced_v5"
COCO="$DATA_ROOT/coco_lineex_balanced_v5"

"$PY" -m training.merge_instance_datasets \
  --source "lineex=$DATA_ROOT/public_lineex_100k_v5" \
  --source "synthetic=$DATA_ROOT/synthetic_lineex_matched_v5" \
  --output "$COMBINED"
"$PY" -m training.convert_coco_instances --source "$COMBINED" --output "$COCO"
