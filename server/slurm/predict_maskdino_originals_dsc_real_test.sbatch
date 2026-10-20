#!/bin/bash
#SBATCH --job-name=md_lf_dsc_real
#SBATCH --partition=aichem
#SBATCH --qos=devdokimov
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=predict_maskdino_originals_dsc_real_test_%j.out
#SBATCH --error=predict_maskdino_originals_dsc_real_test_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
allocated_cuda_devices="${CUDA_VISIBLE_DEVICES:?Slurm CUDA allocation missing}"
source "$PROJECT_ROOT/server/lib/common.sh"
export CUDA_VISIBLE_DEVICES="$allocated_cuda_devices"
export OMP_NUM_THREADS=4
cd "$PROJECT_ROOT"
PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
RUN="$RUNS_ROOT/maskdino_r50_lineformer_originals_dsc_exact_100k_v1"
OUTPUT="$RUNS_ROOT/maskdino_r50_lineformer_originals_dsc_exact_100k_real_test_best"
record_run_metadata "$OUTPUT"
cp "$RUN/best_checkpoint.json" "$OUTPUT/training_best_checkpoint.json"
"$PY" -m training.predict_maskdino \
  --maskdino-root "$MASKDINO_ROOT" \
  --config "$RUN/config.yaml" \
  --weights "$RUN/model_best.pth" \
  --input "$PROJECT_ROOT/real_test" \
  --output "$OUTPUT" \
  --threshold 0.25
