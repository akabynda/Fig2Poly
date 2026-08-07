#!/bin/bash
#SBATCH --job-name=lineex_public_v5
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=prepare_lineex_public_v5_%j.out
#SBATCH --error=prepare_lineex_public_v5_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/yolo/bin/python"
RAW_ROOT="$DATA_ROOT/public_lineex_v5"
OUT="$DATA_ROOT/public_lineex_100k_v5"

"$PY" -m training.download_public_benchmarks \
  --root "$RAW_ROOT" --datasets lineex --extract --delete-archives

for split in train val test; do
  limit=()
  [[ "$split" == train ]] && limit=(--limit 100000)
  "$PY" -m training.convert_public_instances \
    --format lineex \
    --raw-root "$RAW_ROOT/raw/lineex/$split" \
    --output "$OUT" \
    --dataset-name lineex \
    --official-split "$split" \
    --line-width 3 \
    "${limit[@]}"
done
