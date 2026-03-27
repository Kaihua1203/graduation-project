#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/run_with_venv.sh"

CONFIG_PATH="${1:?usage: bash scripts/run_eval.sh <config.yaml>}"
python "$SCRIPT_DIR/../src/eval/run_evaluate.py" --config "$CONFIG_PATH"
