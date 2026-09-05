#!/usr/bin/env bash
# Download-only CPU stage; training preparation can depend on this job.
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$FIG2POLY_ROOT"

PY="${PUBLIC_PREP_PYTHON:-$FIG2POLY_STORAGE/.venvs/yolo/bin/python}"
ADOBE_DOWNLOAD_ROOT="${LINEFORMER_ADOBE_DOWNLOAD:-$DATA_ROOT/lineformer_originals_dsc_download}"
EXISTING_ROOT="${LINEFORMER_ADOBE_METADATA_SOURCE:-$DATA_ROOT/lineformer_benchmarks_download}"

STAGE_STATUS=0
"$PY" -m training.reuse_adobe_metadata \
  --source-root "$EXISTING_ROOT" \
  --destination-root "$ADOBE_DOWNLOAD_ROOT" || STAGE_STATUS=$?
ASSET_ARGS=()
if [[ "$STAGE_STATUS" == 0 ]]; then
  ASSET_ARGS=(--asset-splits all)
elif [[ "$STAGE_STATUS" != 2 ]]; then
  exit "$STAGE_STATUS"
fi

"$PY" -m training.download_public_benchmarks \
  --root "$ADOBE_DOWNLOAD_ROOT" \
  --datasets adobe_synth19 \
  --download-workers "${LINEFORMER_ADOBE_DOWNLOAD_WORKERS:-8}" \
  "${ASSET_ARGS[@]}" \
  --extract \
  --delete-archives
