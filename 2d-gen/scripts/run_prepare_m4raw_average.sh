#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/run_with_venv.sh"

PYTHONPATH="$SCRIPT_DIR/../src${PYTHONPATH:+:$PYTHONPATH}" \
python "$SCRIPT_DIR/../src/data/prepare_m4raw_average_dataset.py" "$@"
