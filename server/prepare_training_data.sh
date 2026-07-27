#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

PY="$FIG2POLY_ROOT/.venvs/yolo/bin/python"
CANONICAL="$DATA_ROOT/combined"
sources=(--source "synthetic=$DATA_ROOT/synthetic")
if [[ -f "$DATA_ROOT/public_instances/train.jsonl" ]]; then
  sources+=(--source "public=$DATA_ROOT/public_instances")
fi
"$PY" -m training.merge_instance_datasets "${sources[@]}" --output "$CANONICAL"
"$PY" -m training.convert_yolo --dataset "$CANONICAL" --workers "${WORKERS:-16}" \
  --subset-train 0 --subset-val 0
"$PY" -m training.convert_coco_instances \
  --source "$CANONICAL" --output "$DATA_ROOT/coco"
