#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/run_with_venv.sh"

die() {
  printf 'error: %s\n' "$1" >&2
  exit 1
}

CONFIG_PATH="${1:?usage: bash scripts/run_infer.sh <config.yaml> [--resume]}"
shift
extra_args=("$@")

gpu_ids_csv="$(
  python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).expanduser().resolve()
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

infer_config = config.get("infer") or {}
gpu_ids = infer_config.get("gpu_ids")
if gpu_ids is None:
    print("")
    raise SystemExit(0)
if not isinstance(gpu_ids, list) or not gpu_ids:
    raise SystemExit("infer.gpu_ids must be a non-empty list of integers.")

normalized_gpu_ids = []
for gpu_id in gpu_ids:
    if not isinstance(gpu_id, int):
        raise SystemExit("infer.gpu_ids must contain integers only.")
    normalized_gpu_ids.append(str(gpu_id))

print(",".join(normalized_gpu_ids))
PY
)"

if [ -n "$gpu_ids_csv" ]; then
  export CUDA_VISIBLE_DEVICES="$gpu_ids_csv"
  IFS=, read -r -a gpu_ids_array <<< "$gpu_ids_csv"
  num_processes="${#gpu_ids_array[@]}"
  [ "$num_processes" -gt 0 ] || die "infer.gpu_ids must contain at least one GPU id."

  if [ "$num_processes" -gt 1 ]; then
    exec accelerate launch \
      --multi_gpu \
      --num_processes "$num_processes" \
      --main_process_port 0 \
      "$SCRIPT_DIR/../src/infer/generator.py" \
      --config "$CONFIG_PATH" \
      "${extra_args[@]}"
  fi
fi

exec python "$SCRIPT_DIR/../src/infer/generator.py" --config "$CONFIG_PATH" "${extra_args[@]}"
