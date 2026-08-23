#!/bin/bash
#SBATCH --job-name=prep_lf_dsc
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=prepare_lineformer_dsc_%j.out
#SBATCH --error=prepare_lineformer_dsc_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
bash "$PROJECT_ROOT/server/prepare_lineformer_dsc.sh"
