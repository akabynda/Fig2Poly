#!/bin/bash
#SBATCH --job-name=lf_originals_dsc_prep
#SBATCH --partition=aichem
#SBATCH --qos=devdokimov
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=prepare_lineformer_originals_dsc_%j.out
#SBATCH --error=prepare_lineformer_originals_dsc_%j.err

set -Eeuo pipefail
PROJECT_ROOT="/mnt/tank/scratch/$(id -un)/Fig2Poly"
source "$PROJECT_ROOT/server/lib/common.sh"
cd "$PROJECT_ROOT"
bash "$PROJECT_ROOT/server/prepare_lineformer_originals_dsc.sh"
PY="$FIG2POLY_STORAGE/.venvs/yolo/bin/python"
"$PY" -m training.prepare_lineformer_mixture \
  --recipe "$DATA_ROOT/public_lineformer_originals_dsc_v1/recipe.json" \
  --output "$DATA_ROOT/coco_lineformer_originals_dsc_exact_v1"

# Download only ImageNet backbone initialization, before allocating training GPUs.
"$PY" - "$CACHE_ROOT/weights" <<'PY'
from pathlib import Path
import sys
import urllib.request

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
for name, url in [
    ("R-50_imagenet.pkl", "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl"),
    ("swin_tiny_patch4_window7_224.pth", "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"),
]:
    target = root / name
    if not target.is_file():
        temporary = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size < 1000000:
            raise RuntimeError(f"Invalid pretrained weights download: {temporary}")
        temporary.replace(target)
    print(f"ImageNet checkpoint: {target} ({target.stat().st_size} bytes)", flush=True)
PY
