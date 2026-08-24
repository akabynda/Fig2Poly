#!/usr/bin/env bash
set -Eeuo pipefail

# Preserve resources selected by Slurm before common.sh loads server/.env.
requested_num_gpus="${NUM_GPUS:-}"
allocated_cuda_devices="${CUDA_VISIBLE_DEVICES:-}"
source "$(dirname "$0")/lib/common.sh"
[[ -n "$requested_num_gpus" ]] && NUM_GPUS="$requested_num_gpus"
[[ -n "$allocated_cuda_devices" ]] && export CUDA_VISIBLE_DEVICES="$allocated_cuda_devices"

PY="$FIG2POLY_STORAGE/.venvs/lineformer/bin/python"
LINEFORMER_ROOT="${LINEFORMER_ROOT:-$FIG2POLY_ROOT/third_party/LineFormer}"
DATASET="${LINEFORMER_DSC_COCO:-$DATA_ROOT/coco_lineformer_dsc}"
OUTPUT="${LINEFORMER_DSC_RUN:-$RUNS_ROOT/lineformer_dsc_finetune}"
WEIGHTS="${LINEFORMER_WEIGHTS:-$CACHE_ROOT/weights/lineformer_iter_3000.pth}"

record_run_metadata "$OUTPUT"
resume=()
[[ -f "$OUTPUT/latest.pth" ]] && resume=(--resume)

"$PY" -m training.train_lineformer \
  --lineformer-root "$LINEFORMER_ROOT" \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --weights "$WEIGHTS" \
  --max-iters "${LINEFORMER_MAX_ITERS:-10000}" \
  --base-lr "${LINEFORMER_BASE_LR:-2e-5}" \
  --samples-per-gpu "${LINEFORMER_SAMPLES_PER_GPU:-2}" \
  --workers-per-gpu "${LINEFORMER_WORKERS_PER_GPU:-4}" \
  --num-gpus "${NUM_GPUS:-1}" \
  --eval-interval "${LINEFORMER_EVAL_INTERVAL:-500}" \
  --checkpoint-interval "${LINEFORMER_CHECKPOINT_INTERVAL:-500}" \
  "${resume[@]}"
