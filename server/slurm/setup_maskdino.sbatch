#!/bin/bash
#SBATCH --job-name=maskdino_setup
#SBATCH --partition=aichem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=setup_maskdino_%j.out
#SBATCH --error=setup_maskdino_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export FORCE_CUDA=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

VENV="$FIG2POLY_STORAGE/.venvs/maskdino"
DETECTRON2_ROOT="$FIG2POLY_ROOT/third_party/detectron2"
mkdir -p "$(dirname "$VENV")" "$FIG2POLY_ROOT/third_party" "$CACHE_ROOT/weights"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --upgrade \
  pip wheel ninja 'setuptools==69.5.1'
"$VENV/bin/pip" install \
  'torch==2.1.2' 'torchvision==0.16.2' \
  --index-url https://download.pytorch.org/whl/cu121
"$VENV/bin/pip" install \
  'numpy==1.26.4' 'opencv-python-headless==4.10.0.84' \
  cython 'scipy<1.14' shapely 'timm==0.9.16' h5py submitit scikit-image \
  pycocotools
"$VENV/bin/pip" install -e "$FIG2POLY_ROOT"

if [[ ! -d "$DETECTRON2_ROOT/.git" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/detectron2.git \
    "$DETECTRON2_ROOT"
fi
"$VENV/bin/pip" install --no-build-isolation -e "$DETECTRON2_ROOT"
git -C "$DETECTRON2_ROOT" rev-parse HEAD > "$CACHE_ROOT/detectron2_git_sha.txt"

if [[ ! -d "$MASKDINO_ROOT/.git" ]]; then
  git clone https://github.com/IDEA-Research/MaskDINO.git "$MASKDINO_ROOT"
fi
git -C "$MASKDINO_ROOT" rev-parse HEAD > "$CACHE_ROOT/maskdino_git_sha.txt"

(
  cd "$MASKDINO_ROOT/maskdino/modeling/pixel_decoder/ops"
  "$VENV/bin/python" setup.py build install
)

if [[ ! -s "$CACHE_ROOT/weights/maskdino_r50.pth" ]]; then
  curl -fL --retry 5 -o "$CACHE_ROOT/weights/maskdino_r50.pth" \
    https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth
fi

"$VENV/bin/python" -c \
  "import torch, detectron2, MultiScaleDeformableAttention; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
