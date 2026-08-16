#!/bin/bash
#SBATCH --job-name=lineformer_setup
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=setup_lineformer_%j.out
#SBATCH --error=setup_lineformer_%j.err

set -Eeuo pipefail
USER_NAME="$(id -un)"
PROJECT_ROOT="/mnt/tank/scratch/$USER_NAME/Fig2Poly"
STORAGE="/mnt/tank/scratch/$USER_NAME"
ROOT="$STORAGE/LineFormer"
VENV="$STORAGE/.venvs/lineformer"
WEIGHTS="$STORAGE/LineFormer_weights"

if [[ ! -d "$ROOT/.git" ]]; then
  git clone https://github.com/TheJaeLal/LineFormer.git "$ROOT"
fi
python3.10 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel 'setuptools<70'
"$VENV/bin/pip" install \
  'torch==1.13.1+cu117' 'torchvision==0.14.1+cu117' \
  --index-url https://download.pytorch.org/whl/cu117
"$VENV/bin/pip" install \
  'numpy<2' openmim chardet scikit-image matplotlib opencv-python-headless \
  pillow 'scipy==1.9.3' bresenham tqdm gdown
"$VENV/bin/mim" install 'mmcv-full==1.7.1'
"$VENV/bin/pip" install -e "$ROOT/mmdetection"

mkdir -p "$WEIGHTS"
if [[ ! -s "$WEIGHTS/iter_3000.pth" ]]; then
  "$VENV/bin/gdown" --folder \
    'https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T' \
    -O "$WEIGHTS"
fi
test -s "$WEIGHTS/iter_3000.pth"
git -C "$ROOT" rev-parse HEAD
"$VENV/bin/python" -c \
  'import torch, mmcv, mmdet; print(torch.__version__, mmcv.__version__, mmdet.__version__, torch.cuda.get_device_name(0))'
ls -lh "$WEIGHTS/iter_3000.pth"
