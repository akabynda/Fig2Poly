#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/lib/common.sh"

nvidia-smi
for run in yolo26x_seg maskdino_r50 maskdino_swin_l; do
  echo "===== $run ====="
  find "$RUNS_ROOT/$run" -maxdepth 2 -type f \
    \( -name 'last.pt' -o -name 'best.pt' -o -name 'last_checkpoint' -o -name 'model_*.pth' \) \
    -printf '%TY-%Tm-%Td %TH:%TM %10s %p\n' 2>/dev/null | tail -10 || true
done
