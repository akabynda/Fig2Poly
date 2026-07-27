#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

PY="$FIG2POLY_ROOT/.venvs/maskdino/bin/python"
OUTPUT="$RUNS_ROOT/maskdino_swin_l"
record_run_metadata "$OUTPUT"
resume=()
[[ -f "$OUTPUT/last_checkpoint" ]] && resume=(--resume)
"$PY" -m training.train_maskdino \
  --maskdino-root "$MASKDINO_ROOT" --dataset "$DATA_ROOT/coco" \
  --output "$OUTPUT" --variant swin_l \
  --weights "$CACHE_ROOT/weights/maskdino_swinl.pth" \
  --epochs "${EPOCHS:-100}" --global-batch "${MASKDINO_SWINL_BATCH:-1}" \
  --num-gpus "${NUM_GPUS:-1}" --workers "${WORKERS:-16}" "${resume[@]}"
