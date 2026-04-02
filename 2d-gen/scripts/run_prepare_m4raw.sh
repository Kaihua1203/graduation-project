#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/run_with_venv.sh"

python "$SCRIPT_DIR/../src/data/prepare_m4raw_dataset.py" "$@"
