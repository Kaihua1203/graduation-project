#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/solo-learn}"
GPU_IDS="${GPU_IDS:-2,3}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/seg/train_ssl_100epochs.yaml}"
SESSION_NAME="${SESSION_NAME:-seg-ssl-100epochs}"

IFS=',' read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
if [[ ${#GPU_ID_ARR[@]} -eq 0 ]]; then
  echo "GPU_IDS is empty."
  exit 1
fi
LOCAL_GPU_IDS="$(seq 0 $((${#GPU_ID_ARR[@]} - 1)) | paste -sd, -)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Please install tmux first."
  exit 1
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON="${VENV_DIR}/bin/python"
else
  PYTHON="python"
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 1
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd \"${PROJECT_DIR}\" && CUDA_VISIBLE_DEVICES=\"${GPU_IDS}\" \"${PYTHON}\" src/train_segmentation.py --config \"${CONFIG_PATH}\" --gpus \"${LOCAL_GPU_IDS}\""

echo "Started tmux session: ${SESSION_NAME}"
echo "Config: ${CONFIG_PATH}"
echo "Physical GPUs: ${GPU_IDS}"
echo "Local GPUs (--gpus): ${LOCAL_GPU_IDS}"
echo "Attach: tmux attach -t ${SESSION_NAME}"
