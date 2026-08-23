#!/bin/bash
#SBATCH --job-name=lineformer_dsc
#SBATCH --partition=aichem
#SBATCH --nodelist=aihub
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=train_lineformer_dsc_%j.out
#SBATCH --error=train_lineformer_dsc_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

trainer_pid=""
requeue_on_timeout() {
  trap - USR1
  echo "Received timeout signal; stopping LineFormer before requeue"
  if [[ -n "$trainer_pid" ]] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill -TERM "$trainer_pid" 2>/dev/null || true
    wait "$trainer_pid" || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_on_timeout USR1

bash "$PROJECT_ROOT/server/train_lineformer_dsc.sh" &
trainer_pid=$!
wait "$trainer_pid"
