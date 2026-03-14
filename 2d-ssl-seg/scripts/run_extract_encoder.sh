#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_extract_encoder.sh /path/to/ssl_checkpoint.ckpt"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/solo-learn}"
INPUT_CKPT="$1"
OUTPUT_ENCODER="${PROJECT_DIR}/outputs/encoders/lits_vicreg_encoder.pth"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON="${VENV_DIR}/bin/python"
else
  PYTHON="python"
fi

"${PYTHON}" "${PROJECT_DIR}/src/extract_backbone.py" \
  --input "${INPUT_CKPT}" \
  --output "${OUTPUT_ENCODER}"
