#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/lineformer/bin/python"
SOURCE="${DSC_DATASET:-$DATA_ROOT/synthetic_dsc}"
OUTPUT="${LINEFORMER_DSC_COCO:-$DATA_ROOT/coco_lineformer_dsc}"

if [[ ! -f "$SOURCE/train.jsonl" ]]; then
  echo "Missing $SOURCE/train.jsonl; set DSC_DATASET to the generated CurveForge dataset" >&2
  exit 2
fi

"$PY" -m training.convert_coco_instances \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --category-name line \
  --train-mask-dilation "${LINEFORMER_TRAIN_MASK_WIDTH:-3}" \
  --val-mask-dilation "${LINEFORMER_VAL_MASK_WIDTH:-1}"

wc -l "$SOURCE/train.jsonl" "$SOURCE/val.jsonl" "$SOURCE/test.jsonl"
echo "Prepared LineFormer COCO dataset at $OUTPUT"
