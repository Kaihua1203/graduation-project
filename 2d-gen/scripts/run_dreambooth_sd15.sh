#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/run_with_venv.sh"

CONFIG_PATH="${1:?usage: bash scripts/run_dreambooth_sd15.sh <config.yaml>}"
shift

exec accelerate launch "$@" "$SCRIPT_DIR/../src/train/run_dreambooth_sd15.py" --config "$CONFIG_PATH"
