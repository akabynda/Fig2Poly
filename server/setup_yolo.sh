#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

VENV="$FIG2POLY_ROOT/.venvs/yolo"
"${PYTHON_BIN:-python3.10}" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
"$VENV/bin/pip" install torch torchvision --index-url "$TORCH_INDEX_URL"
"$VENV/bin/pip" install -e "$FIG2POLY_ROOT"
"$VENV/bin/pip" install -r "$FIG2POLY_ROOT/server/requirements-yolo.txt"
"$VENV/bin/python" -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
