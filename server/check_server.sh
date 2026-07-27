#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

echo "Project: $FIG2POLY_ROOT"
echo "Storage: $FIG2POLY_STORAGE"
nvidia-smi
df -h "$FIG2POLY_STORAGE"
free -h
"${PYTHON_BIN:-python3}" --version
command -v nvcc >/dev/null && nvcc --version || echo "nvcc is not installed"
