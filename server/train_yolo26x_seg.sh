#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

PY="$FIG2POLY_ROOT/.venvs/yolo/bin/python"
OUTPUT="$RUNS_ROOT/yolo26x_seg"
record_run_metadata "$OUTPUT"
LAST="$OUTPUT/weights/last.pt"
if [[ -f "$LAST" ]]; then
  "$PY" -m training.train_yolo --resume "$LAST" --device 0 --workers "${WORKERS:-16}"
else
  "$PY" -m training.train_yolo \
    --model yolo26x-seg.pt \
    --data "$DATA_ROOT/combined/curve_yolo.yaml" \
    --project "$RUNS_ROOT" --name yolo26x_seg \
    --epochs "${EPOCHS:-100}" --imgsz 1024 --batch "${YOLO_BATCH:--1}" \
    --workers "${WORKERS:-16}" --device 0 --patience 40
fi
