#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

YOLO_PY="$FIG2POLY_ROOT/.venvs/yolo/bin/python"
MASK_PY="$FIG2POLY_ROOT/.venvs/maskdino/bin/python"
YOLO_WEIGHTS="$RUNS_ROOT/yolo26x_seg/weights/best.pt"
if [[ -f "$YOLO_WEIGHTS" ]]; then
  "$YOLO_PY" -m training.evaluate_yolo \
    --weights "$YOLO_WEIGHTS" --data "$DATA_ROOT/combined/curve_yolo.yaml" \
    --split test --imgsz 1024 --output "$RUNS_ROOT/yolo26x_seg/test_metrics.json"
fi

for variant in r50 swin_l; do
  run_name="maskdino_${variant}"
  [[ "$variant" == "swin_l" ]] && run_name="maskdino_swin_l"
  output="$RUNS_ROOT/$run_name"
  if [[ -f "$output/last_checkpoint" ]]; then
    checkpoint="$(<"$output/last_checkpoint")"
    weights="$output/$checkpoint"
    "$MASK_PY" -m training.train_maskdino \
      --maskdino-root "$MASKDINO_ROOT" --dataset "$DATA_ROOT/coco" \
      --output "$output" --variant "$variant" --weights "$weights" \
      --num-gpus "${NUM_GPUS:-1}" --global-batch 1 --eval-only --eval-split test \
      --report "$output/test_metrics.json"
  fi
done

"$YOLO_PY" -m training.compare_metrics \
  "$RUNS_ROOT/yolo26x_seg/test_metrics.json" \
  "$RUNS_ROOT/maskdino_r50/test_metrics.json" \
  "$RUNS_ROOT/maskdino_swin_l/test_metrics.json" \
  --output "$RUNS_ROOT/comparison_test.json"
