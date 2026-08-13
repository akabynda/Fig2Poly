#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

VENV="$FIG2POLY_ROOT/.venvs/maskdino"
"${PYTHON_BIN:-python3.10}" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel setuptools ninja
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
"$VENV/bin/pip" install torch torchvision --index-url "$TORCH_INDEX_URL"
"$VENV/bin/pip" install -e "$FIG2POLY_ROOT"
"$VENV/bin/pip" install opencv-python-headless pycocotools scipy shapely timm
"$VENV/bin/pip" install 'git+https://github.com/facebookresearch/detectron2.git'

if [[ ! -d "$MASKDINO_ROOT/.git" ]]; then
  git clone https://github.com/IDEA-Research/MaskDINO.git "$MASKDINO_ROOT"
fi
git -C "$MASKDINO_ROOT" rev-parse HEAD > "$CACHE_ROOT/maskdino_git_sha.txt"
"$VENV/bin/python" -m training.patch_maskdino_numerics \
  --maskdino-root "$MASKDINO_ROOT"
"$VENV/bin/pip" install -r "$MASKDINO_ROOT/requirements.txt"
(
  cd "$MASKDINO_ROOT/maskdino/modeling/pixel_decoder/ops"
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}" "$VENV/bin/python" setup.py build install
)

mkdir -p "$CACHE_ROOT/weights"
curl -fL --retry 5 -o "$CACHE_ROOT/weights/maskdino_r50.pth" \
  https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth
curl -fL --retry 5 -o "$CACHE_ROOT/weights/maskdino_swinl.pth" \
  https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth
"$VENV/bin/python" -c "import torch, detectron2; print(torch.__version__, torch.cuda.is_available())"
