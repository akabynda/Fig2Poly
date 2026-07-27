#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

PY="$FIG2POLY_ROOT/.venvs/yolo/bin/python"
"$PY" -m curveforge \
  --config "$FIG2POLY_ROOT/configs/full_v4.json" \
  --output "$DATA_ROOT/synthetic" \
  --count "${SYNTH_COUNT:-400000}" \
  --workers "${WORKERS:-16}" \
  --resume
