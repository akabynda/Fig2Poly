#!/bin/bash
#SBATCH --job-name=setup_yolo
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=setup_yolo_%j.out
#SBATCH --error=setup_yolo_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

VENV="$FIG2POLY_STORAGE/.venvs/yolo"
BOOTSTRAP_PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
"$BOOTSTRAP_PY" -m pip install --upgrade virtualenv
"$BOOTSTRAP_PY" -m virtualenv --clear "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
"$VENV/bin/pip" install -e "$PROJECT_ROOT"
"$VENV/bin/pip" install -r "$PROJECT_ROOT/server/requirements-yolo.txt"
"$VENV/bin/python" -c \
  'import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)'
