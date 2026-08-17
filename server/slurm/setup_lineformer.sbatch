#!/bin/bash
#SBATCH --job-name=setup_lineformer
#SBATCH --partition=aichem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=setup_lineformer_%j.out
#SBATCH --error=setup_lineformer_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
bash "$PROJECT_ROOT/server/setup_lineformer.sh"
