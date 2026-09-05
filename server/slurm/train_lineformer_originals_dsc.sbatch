#!/bin/bash
#SBATCH --job-name=lf_originals_dsc_train
#SBATCH --partition=aichem
#SBATCH --qos=devdokimov
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=train_lineformer_originals_dsc_%j.out
#SBATCH --error=train_lineformer_originals_dsc_%j.err

set -Eeuo pipefail
variant="${1:?Expected mask2former or maskdino}"
case "$variant" in mask2former|maskdino) ;; *) exit 2 ;; esac
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
allocated_cuda_devices="${CUDA_VISIBLE_DEVICES:?Slurm CUDA allocation missing}"
source "$PROJECT_ROOT/server/lib/common.sh"
export CUDA_VISIBLE_DEVICES="$allocated_cuda_devices"
IFS=',' read -ra allocated_gpus <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#allocated_gpus[@]} -eq 2 ]] || { echo "Expected 2 Slurm GPUs" >&2; exit 2; }
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4
# torch.load_state_dict_from_url can reuse the ImageNet checkpoint downloaded by prep.
export TORCH_HOME="$CACHE_ROOT/torch"
mkdir -p "$TORCH_HOME/hub/checkpoints"
if [[ ! -e "$TORCH_HOME/hub/checkpoints/swin_tiny_patch4_window7_224.pth" ]]; then
  ln -s "$CACHE_ROOT/weights/swin_tiny_patch4_window7_224.pth" \
    "$TORCH_HOME/hub/checkpoints/swin_tiny_patch4_window7_224.pth"
fi
DATASET="$DATA_ROOT/coco_lineformer_originals_dsc_exact_v1"
test -s "$DATASET/mixture_summary.json"
cd "$PROJECT_ROOT"

if [[ "$variant" == mask2former ]]; then
  OUTPUT="$RUNS_ROOT/mask2former_swin_t_lineformer_originals_dsc_exact_100k_v1"
  PY="$FIG2POLY_STORAGE/.venvs/lineformer/bin/python"
  command=("$PY" -m training.train_mask2former_lineformer
    --lineformer-root "$PROJECT_ROOT/third_party/LineFormer"
    --dataset "$DATASET" --output "$OUTPUT" --num-gpus 2 --workers-per-gpu 4)
  [[ -f "$OUTPUT/latest.pth" ]] && command+=(--resume)
else
  OUTPUT="$RUNS_ROOT/maskdino_r50_lineformer_originals_dsc_exact_100k_v1"
  PY="$FIG2POLY_STORAGE/.venvs/maskdino/bin/python"
  command=("$PY" -m training.train_maskdino
    --maskdino-root "$MASKDINO_ROOT" --dataset "$DATASET" --output "$OUTPUT"
    --variant r50 --weights "$CACHE_ROOT/weights/R-50_imagenet.pkl"
    --training-profile lineformer --num-gpus 2 --global-batch 8 --workers 8)
  [[ -f "$OUTPUT/last_checkpoint" ]] && command+=(--resume)
fi
record_run_metadata "$OUTPUT"
cp "$DATASET/mixture_summary.json" "$OUTPUT/mixture_summary.json"
trainer_pid=""
requeue_on_timeout() {
  trap - USR1
  echo "Stopping process group before Slurm requeue"
  if [[ -n "$trainer_pid" ]] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill -TERM -- "-$trainer_pid" 2>/dev/null || true
    wait "$trainer_pid" || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_on_timeout USR1
setsid "${command[@]}" &
trainer_pid=$!
wait "$trainer_pid"
