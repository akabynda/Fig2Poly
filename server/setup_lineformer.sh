#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

VENV="$FIG2POLY_STORAGE/.venvs/lineformer"
ROOT="$FIG2POLY_ROOT/third_party/LineFormer"
COMMIT="7952e27b4653dea025394618fbd655f41d82ab6b"

"${PYTHON_BIN:-python3.10}" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade "pip<25" setuptools==59.5.0 wheel
"$PY" -m pip install \
  torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
"$PY" -m pip install addict pyyaml yapf==0.40.1 packaging "pillow<10"
"$PY" -m pip install --no-deps mmcv-full==1.7.1 \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
"$PY" -m pip uninstall -y opencv-python || true
"$PY" -m pip install \
  "numpy<1.24" scipy==1.9.3 scikit-image==0.19.3 \
  opencv-python-headless==4.8.1.78 matplotlib==3.7.5 pillow \
  bresenham tqdm terminaltables pycocotools
"$PY" -m pip install --force-reinstall --no-deps opencv-python-headless==4.8.1.78

if [[ ! -d "$ROOT/.git" ]]; then
  git clone https://github.com/TheJaeLal/LineFormer.git "$ROOT"
fi
git -C "$ROOT" fetch --all --tags
git -C "$ROOT" checkout --detach "$COMMIT"
# The benchmark adds the vendored mmdetection tree to PYTHONPATH. Installing it
# editable would invoke torch's legacy extension build machinery even though
# inference only needs the prebuilt operators shipped by mmcv-full.
PYTHONPATH="$ROOT/mmdetection:$ROOT" "$PY" - <<'PY'
import mmcv, mmdet, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("mmcv", mmcv.__version__, "mmdet", mmdet.__version__)
PY
