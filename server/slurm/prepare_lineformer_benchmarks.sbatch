#!/bin/bash
#SBATCH --job-name=line_bench_data
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=prepare_lineformer_benchmarks_%j.out
#SBATCH --error=prepare_lineformer_benchmarks_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"

PY="$FIG2POLY_STORAGE/.venvs/yolo/bin/python"
RAW="$DATA_ROOT/lineformer_benchmarks_download"
OUT="$DATA_ROOT/lineformer_benchmarks"

"$PY" -m training.download_public_benchmarks \
  --root "$RAW" \
  --datasets ub_pmc22,adobe_synth19 \
  --asset-splits test \
  --extract \
  --delete-archives

# The paper evaluates UB-PMC22 on the official test release (158 line charts).
"$PY" -m training.convert_public_instances \
  --format chartinfo \
  --raw-root "$RAW/raw/ub_pmc22" \
  --output "$OUT/ub_pmc22" \
  --dataset-name ub_pmc22 \
  --official-split test \
  --annotation-path-contains test \
  --line-width 3

# Adobe test annotations and the image shards are extracted under the same
# root. Restrict pairing to the official Task-6 ground-truth directory.
ADOBE="$RAW/raw/adobe_synth19"
"$PY" -m training.convert_public_instances \
  --format chartinfo \
  --raw-root "$ADOBE" \
  --output "$OUT/adobe_synth19" \
  --dataset-name adobe_synth19 \
  --official-split test \
  --annotation-root "$ADOBE/test_release/task6/gt_json" \
  --line-width 3

wc -l "$OUT/ub_pmc22/test.jsonl" "$OUT/adobe_synth19/test.jsonl"
