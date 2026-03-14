#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/solo-learn}"
GPU_IDS="${GPU_IDS:-0}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON="${VENV_DIR}/bin/python"
else
  PYTHON="python"
fi

"${PYTHON}" "${PROJECT_DIR}/src/train_segmentation.py" \
  --config "${PROJECT_DIR}/configs/seg/train_random.yaml" \
  --random-init \
  --gpus "${GPU_IDS}"
