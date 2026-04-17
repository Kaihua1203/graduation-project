#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/run_with_venv.sh"

die() {
  printf 'error: %s\n' "$1" >&2
  exit 1
}

count_visible_gpus() {
  local visible_devices="${CUDA_VISIBLE_DEVICES// /}"
  local -a device_ids=()
  IFS=, read -r -a device_ids <<< "$visible_devices"
  local count=0
  local device_id
  for device_id in "${device_ids[@]}"; do
    [ -n "$device_id" ] && count=$((count + 1))
  done
  printf '%s\n' "$count"
}

CONFIG_PATH="${1:?usage: bash scripts/run_train.sh <config.yaml>}"
shift

train_entrypoint="$(
  python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).expanduser().resolve()
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

model_config = config.get("model") or {}
if model_config.get("model_type") == "uncond_ldm":
    print("run_train_uncond_ldm.py")
else:
    print("run_train.py")
PY
)"

launcher_args=()
multi_gpu=0
num_processes=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --multi_gpu)
      multi_gpu=1
      launcher_args+=("$1")
      ;;
    --num_processes)
      launcher_args+=("$1")
      shift
      [ "$#" -gt 0 ] || die "--num_processes requires a value."
      num_processes="$1"
      launcher_args+=("$num_processes")
      ;;
    --num_processes=*)
      num_processes="${1#*=}"
      launcher_args+=("$1")
      ;;
    *)
      launcher_args+=("$1")
      ;;
  esac
  shift
done

if [ -n "$num_processes" ]; then
  case "$num_processes" in
    ''|*[!0-9]*)
      die "--num_processes must be a positive integer."
      ;;
  esac
  [ "$num_processes" -gt 0 ] || die "--num_processes must be a positive integer."
fi

if [ "$multi_gpu" -eq 1 ] && [ -z "$num_processes" ]; then
  die "multi-GPU launches must pass --num_processes."
fi

if [ -n "$num_processes" ] && [ "$num_processes" -gt 1 ]; then
  [ "$multi_gpu" -eq 1 ] || die "multi-GPU launches must pass --multi_gpu explicitly."
fi

if [ "$multi_gpu" -eq 1 ]; then
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || die "multi-GPU launches require CUDA_VISIBLE_DEVICES to be set."

  visible_gpu_count="$(count_visible_gpus)"
  [ "$visible_gpu_count" -eq "$num_processes" ] || die \
    "CUDA_VISIBLE_DEVICES lists ${visible_gpu_count} GPU(s) but --num_processes is ${num_processes}."
fi

exec accelerate launch "${launcher_args[@]}" "$SCRIPT_DIR/../src/train/$train_entrypoint" --config "$CONFIG_PATH"
