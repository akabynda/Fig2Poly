#!/usr/bin/env bash
# CPU-only preparation; invoke from a CPU Slurm job, never on the login node.
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"
cd "$FIG2POLY_ROOT"

PY="${PUBLIC_PREP_PYTHON:-$FIG2POLY_STORAGE/.venvs/yolo/bin/python}"
RAW="${LINEFORMER_PUBLIC_RAW:-$DATA_ROOT/lineformer_benchmarks_download/raw}"
PUBLIC_ROOT="${LINEFORMER_PUBLIC_OUTPUT:-$DATA_ROOT/public_lineformer_originals_dsc_v1}"
ADOBE_DOWNLOAD_ROOT="${LINEFORMER_ADOBE_DOWNLOAD:-$DATA_ROOT/lineformer_originals_dsc_download}"
LINEEX_RAW="${LINEFORMER_LINEEX_RAW:-$DATA_ROOT/public_lineex_v5/raw/lineex}"
DSC_COCO="${LINEFORMER_DSC_COCO:-$DATA_ROOT/coco_lineformer_dsc_exact}"
DSC_SOURCE="${DSC_DATASET:-$DATA_ROOT/dataset_dsc}"

# Fresh download state avoids reusing old receipts for a test-only image subset.
# Download ~11.6 GB of Adobe image archives plus metadata; keep only eligible
# line/scatter-line images. Each new archive is deleted after checked extraction.
# Existing raw benchmarks and the existing DSC COCO files remain read-only.
if [[ "${LINEFORMER_SKIP_ADOBE_DOWNLOAD:-0}" != 1 ]]; then
  "$PY" -m training.download_public_benchmarks \
    --root "$ADOBE_DOWNLOAD_ROOT" \
    --datasets adobe_synth19 \
    --extract \
    --delete-archives
fi

"$PY" -m training.prepare_lineformer_public \
  --pmc-train-root "$RAW/ub_pmc22/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0" \
  --pmc-test-root "$RAW/ub_pmc22/ICPR2022_CHARTINFO_UB_UNITEC_PMC_TEST_v2.1" \
  --adobe-root "$ADOBE_DOWNLOAD_ROOT/raw/adobe_synth19" \
  --lineex-root "$LINEEX_RAW" \
  --dsc-coco "$DSC_COCO" \
  --dsc-source "$DSC_SOURCE" \
  --output "$PUBLIC_ROOT" \
  --validation-fraction "${LINEFORMER_PUBLIC_VAL_FRACTION:-0.1}" \
  --line-width "${LINEFORMER_PUBLIC_LINE_WIDTH:-1}"

echo "Public + DSC recipe: $PUBLIC_ROOT/recipe.json"
