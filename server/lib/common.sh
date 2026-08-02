#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${FIG2POLY_ENV:-$SCRIPT_DIR/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Copy server/.env.example to server/.env and edit paths." >&2
  exit 2
fi
set -a
source "$ENV_FILE"
set +a

: "${FIG2POLY_ROOT:?}"
: "${FIG2POLY_STORAGE:?}"
: "${DATA_ROOT:?}"
: "${RUNS_ROOT:?}"
: "${CACHE_ROOT:?}"
: "${MASKDINO_ROOT:?}"

mkdir -p "$DATA_ROOT" "$RUNS_ROOT" "$CACHE_ROOT" "$FIG2POLY_STORAGE/logs"
export PYTHONPATH="$FIG2POLY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${CUDA_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

record_run_metadata() {
  local output="$1"
  mkdir -p "$output"
  if git -C "$FIG2POLY_ROOT" rev-parse HEAD >/dev/null 2>&1; then
    git -C "$FIG2POLY_ROOT" rev-parse HEAD > "$output/fig2poly_git_sha.txt"
  elif [[ -f "$FIG2POLY_ROOT/DEPLOYED_COMMIT" ]]; then
    cp "$FIG2POLY_ROOT/DEPLOYED_COMMIT" "$output/fig2poly_git_sha.txt"
  else
    printf 'unknown\n' > "$output/fig2poly_git_sha.txt"
  fi
  if ! nvidia-smi > "$output/nvidia_smi.txt" 2>&1; then
    printf 'nvidia-smi unavailable; training may still use CUDA\n' \
      > "$output/nvidia_smi.txt"
  fi
}
