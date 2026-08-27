#!/bin/bash
#SBATCH --job-name=test_lf_exact
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=benchmark_lineformer_dsc_exact_%j.out
#SBATCH --error=benchmark_lineformer_dsc_exact_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
ALLOCATED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
source "$PROJECT_ROOT/server/lib/common.sh"
PY="$FIG2POLY_STORAGE/.venvs/lineformer/bin/python"
WEIGHTS="$RUNS_ROOT/lineformer_dsc_exact_5ep_earlystop/best_segm_mAP_iter_20000.pth"
TEST_OUTPUT="$RUNS_ROOT/lineformer_dsc_exact_test"
REAL_OUTPUT="$RUNS_ROOT/lineformer_dsc_exact_real_test_thresholds"

IFS=',' read -r gpu_test gpu_real _ <<< "$ALLOCATED_CUDA_DEVICES"
if [[ -z "$gpu_test" || -z "$gpu_real" ]]; then
  echo "Expected two Slurm GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_CUDA_DEVICES" >&2
  exit 2
fi
mkdir -p "$TEST_OUTPUT" "$REAL_OUTPUT"
record_run_metadata "$TEST_OUTPUT"
record_run_metadata "$REAL_OUTPUT"
printf '%s\n' "$WEIGHTS" > "$TEST_OUTPUT/checkpoint.txt"
printf '%s\n' "$WEIGHTS" > "$REAL_OUTPUT/checkpoint.txt"

CUDA_VISIBLE_DEVICES="$gpu_test" "$PY" -m training.benchmark_lineformer \
  --lineformer-root "$PROJECT_ROOT/third_party/LineFormer" \
  --weights "$WEIGHTS" \
  --output "$TEST_OUTPUT/exact.sqlite" \
  --min-threshold 0.15 \
  --stop-on-error \
  --dataset "dsc_test=manifest=$DATA_ROOT/dataset_dsc/test.jsonl" &
test_pid=$!

CUDA_VISIBLE_DEVICES="$gpu_real" "$PY" -m training.predict_lineformer_panels \
  --root "$PROJECT_ROOT/third_party/LineFormer" \
  --weights "$WEIGHTS" \
  --input "$PROJECT_ROOT/real_test" \
  --output "$REAL_OUTPUT" \
  --thresholds 0.15 0.30 0.50 0.70 &
real_pid=$!

set +e
wait "$test_pid"
test_status=$?
wait "$real_pid"
real_status=$?
set -e
if [[ "$test_status" -ne 0 || "$real_status" -ne 0 ]]; then
  echo "Inference failed: dsc_test=$test_status real_test=$real_status" >&2
  exit 1
fi
echo "DSC test and real_test inference completed."
