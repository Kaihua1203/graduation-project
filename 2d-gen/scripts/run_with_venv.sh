#!/usr/bin/env bash
set -eu

DEFAULT_VENV_DIR="/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/diffusers"
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$VENV_DIR/bin/activate" ]; then
  . "$VENV_DIR/bin/activate"
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ -n "${DIFFUSERS_SRC_PATH:-}" ]; then
  export PYTHONPATH="$DIFFUSERS_SRC_PATH:$PYTHONPATH"
fi
