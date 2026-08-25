#!/bin/bash
#SBATCH --job-name=bench_lf_dsc_pair
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=benchmark_lineformer_dsc_pair_%j.out
#SBATCH --error=benchmark_lineformer_dsc_pair_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
# Let Slurm assign a different device to each parallel step. common.sh may set
# a workstation-style CUDA_VISIBLE_DEVICES default from server/.env.
unset CUDA_VISIBLE_DEVICES
PY="$FIG2POLY_STORAGE/.venvs/lineformer/bin/python"
OUTPUT="$RUNS_ROOT/lineformer_dsc_test_comparison"
DATASET="dsc_test=manifest=$DATA_ROOT/dataset_dsc/test.jsonl"
CLASSIC_WEIGHTS="$CACHE_ROOT/weights/lineformer_iter_3000.pth"
FINETUNED_WEIGHTS="$RUNS_ROOT/lineformer_dsc_5ep_earlystop/best_segm_mAP_iter_60000.pth"

mkdir -p "$OUTPUT"
record_run_metadata "$OUTPUT"
printf '%s\n' "$CLASSIC_WEIGHTS" > "$OUTPUT/classic_checkpoint.txt"
printf '%s\n' "$FINETUNED_WEIGHTS" > "$OUTPUT/finetuned_checkpoint.txt"

run_model() {
  local weights="$1"
  local database="$2"
  "$PY" -m training.benchmark_lineformer \
    --lineformer-root "$PROJECT_ROOT/third_party/LineFormer" \
    --weights "$weights" \
    --output "$database" \
    --min-threshold 0.15 \
    --stop-on-error \
    --dataset "$DATASET"
}
export -f run_model
export PY PROJECT_ROOT DATASET

srun --exclusive --ntasks=1 --cpus-per-task=8 --mem=44G --gres=gpu:1 \
  bash -c 'run_model "$1" "$2"' _ "$CLASSIC_WEIGHTS" "$OUTPUT/classic.sqlite" &
classic_pid=$!
srun --exclusive --ntasks=1 --cpus-per-task=8 --mem=44G --gres=gpu:1 \
  bash -c 'run_model "$1" "$2"' _ "$FINETUNED_WEIGHTS" "$OUTPUT/finetuned.sqlite" &
finetuned_pid=$!

set +e
wait "$classic_pid"
classic_status=$?
wait "$finetuned_pid"
finetuned_status=$?
set -e
if [[ "$classic_status" -ne 0 || "$finetuned_status" -ne 0 ]]; then
  echo "Inference failed: classic=$classic_status finetuned=$finetuned_status" >&2
  exit 1
fi

echo "Both LineFormer checkpoints completed inference."
