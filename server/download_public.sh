#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

PY="$FIG2POLY_ROOT/.venvs/yolo/bin/python"
"$PY" -m training.download_public_benchmarks \
  --root "$DATA_ROOT/public" \
  --extract \
  --delete-archives
