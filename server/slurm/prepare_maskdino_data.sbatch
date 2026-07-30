#!/bin/bash
#SBATCH --job-name=fig2poly_coco
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=prepare_maskdino_%j.out
#SBATCH --error=prepare_maskdino_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/curveforge/bin/python"
SOURCE="$DATA_ROOT/synthetic_200k"
COCO="$DATA_ROOT/coco_synthetic_200k"

"$PY" -m training.convert_coco_instances --source "$SOURCE" --output "$COCO"
"$PY" -m training.create_maskdino_smoke_dataset \
  --source "$COCO" --output "$DATA_ROOT/coco_smoke_empty"
