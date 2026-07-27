#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

PY="$FIG2POLY_ROOT/.venvs/yolo/bin/python"
RAW="$DATA_ROOT/public/raw"
OUT="$DATA_ROOT/public_instances"

for split in train val test; do
  "$PY" -m training.convert_public_instances \
    --format lineex --raw-root "$RAW/lineex/$split" --output "$OUT" \
    --dataset-name lineex --official-split "$split"
done

"$PY" -m training.convert_public_instances \
  --format chartinfo --raw-root "$RAW/ub_pmc22" --output "$OUT" \
  --dataset-name ub_pmc22 --official-split train --validation-fraction 0.1 \
  --annotation-path-contains train
"$PY" -m training.convert_public_instances \
  --format chartinfo --raw-root "$RAW/ub_pmc22" --output "$OUT" \
  --dataset-name ub_pmc22_test --official-split test \
  --annotation-path-contains test

ADOBE="$RAW/adobe_synth19"
"$PY" -m training.convert_public_instances \
  --format chartinfo --raw-root "$ADOBE" --output "$OUT" \
  --dataset-name adobe_synth19 --official-split train --validation-fraction 0.1 \
  --annotation-root "$ADOBE/json_gt"
"$PY" -m training.convert_public_instances \
  --format chartinfo --raw-root "$ADOBE/test_release/task6" --output "$OUT" \
  --dataset-name adobe_synth19_test --official-split test \
  --annotation-root "$ADOBE/test_release/task6/gt_json"
