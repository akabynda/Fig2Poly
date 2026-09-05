#!/bin/bash
#SBATCH --job-name=lf_originals_dsc_smoke
#SBATCH --partition=aichem
#SBATCH --qos=devdokimov
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=smoke_lineformer_originals_dsc_%j.out
#SBATCH --error=smoke_lineformer_originals_dsc_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
allocated_cuda_devices="${CUDA_VISIBLE_DEVICES:?Slurm CUDA allocation missing}"
source "$PROJECT_ROOT/server/lib/common.sh"
export CUDA_VISIBLE_DEVICES="$allocated_cuda_devices"
IFS=',' read -ra allocated_gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#allocated_gpus[@]} -eq 2 ]] || { echo "Expected 2 Slurm GPUs" >&2; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4
export TORCH_HOME="$CACHE_ROOT/torch"
mkdir -p "$TORCH_HOME/hub/checkpoints"
if [[ ! -e "$TORCH_HOME/hub/checkpoints/swin_tiny_patch4_window7_224.pth" ]]; then
  ln -s "$CACHE_ROOT/weights/swin_tiny_patch4_window7_224.pth" \
    "$TORCH_HOME/hub/checkpoints/swin_tiny_patch4_window7_224.pth"
fi
cd "$PROJECT_ROOT"
OUTPUT="$RUNS_ROOT/lineformer_originals_dsc_exact_smoke_v1/$SLURM_JOB_ID"
DATASET="$OUTPUT/dataset"
record_run_metadata "$OUTPUT"
"$FIG2POLY_STORAGE/.venvs/yolo/bin/python" -m training.prepare_lineformer_smoke \
  --dataset "$DATA_ROOT/coco_lineformer_originals_dsc_exact_v1" --output "$DATASET"

"$FIG2POLY_STORAGE/.venvs/lineformer/bin/python" -m training.train_mask2former_lineformer \
  --lineformer-root "$PROJECT_ROOT/third_party/LineFormer" \
  --dataset "$DATASET" --output "$OUTPUT/mask2former" --num-gpus 2 \
  --workers-per-gpu 2 --smoke-test

"$FIG2POLY_STORAGE/.venvs/maskdino/bin/python" -m training.train_maskdino \
  --maskdino-root "$MASKDINO_ROOT" --dataset "$DATASET" --output "$OUTPUT/maskdino" \
  --variant r50 --weights "$CACHE_ROOT/weights/R-50_imagenet.pkl" \
  --training-profile lineformer --num-gpus 2 --global-batch 8 --workers 2 \
  --max-iter 2 --eval-period 1 --checkpoint-period 1 --log-interval 1

test -s "$OUTPUT/mask2former/latest.pth"
test -s "$OUTPUT/maskdino/model_final.pth"
printf 'Both models passed 2-GPU training, validation and checkpoint smoke tests.\n' \
  > "$OUTPUT/SUCCESS"
