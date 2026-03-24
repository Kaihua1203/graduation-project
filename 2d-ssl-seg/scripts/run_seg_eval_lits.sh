#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/solo-learn}"
GPU_IDS="${GPU_IDS:-0}"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_seg_eval_lits.sh <ssl|random>"
  exit 1
fi
EVAL_TARGET="$1" # ssl | random

TEST_IMAGES_DIR="${TEST_IMAGES_DIR:-/home/jupyter-wenkaihua/data3_link/kaihua.wen/dataset/MedSegFactory/test/LiTS/imagesTs}"
TEST_MASKS_DIR="${TEST_MASKS_DIR:-/home/jupyter-wenkaihua/data3_link/kaihua.wen/dataset/MedSegFactory/test/LiTS/labelsTs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
SEG_SSL_EXPERIMENT_NAME="${SEG_SSL_EXPERIMENT_NAME:-seg-ssl}"
SEG_RANDOM_EXPERIMENT_NAME="${SEG_RANDOM_EXPERIMENT_NAME:-seg-random}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON="${VENV_DIR}/bin/python"
else
  PYTHON="python"
fi

run_eval() {
  local checkpoint="$1"
  local run_name="$2"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint not found: ${checkpoint}"
    exit 1
  fi

  "${PYTHON}" "${PROJECT_DIR}/src/evaluate_segmentation.py" \
    --checkpoint "${checkpoint}" \
    --test-images-dir "${TEST_IMAGES_DIR}" \
    --test-masks-dir "${TEST_MASKS_DIR}" \
    --gpus "${GPU_IDS}" \
    --output-dir "${OUTPUT_ROOT}" \
    --run-name "${run_name}"
}

if [[ "${EVAL_TARGET}" == "ssl" ]]; then
  run_eval \
    "${PROJECT_DIR}/outputs/seg_ssl/${SEG_SSL_EXPERIMENT_NAME}/best_model.pt" \
    "seg-ssl"
fi

if [[ "${EVAL_TARGET}" == "random" ]]; then
  run_eval \
    "${PROJECT_DIR}/outputs/seg_random/${SEG_RANDOM_EXPERIMENT_NAME}/best_model.pt" \
    "seg-random"
fi

if [[ "${EVAL_TARGET}" != "ssl" && "${EVAL_TARGET}" != "random" ]]; then
  echo "Invalid target: ${EVAL_TARGET}. Expected one of: ssl, random."
  exit 1
fi

echo "Evaluation finished. Logs:"
echo "  ${OUTPUT_ROOT}/logs/seg-ssl/evaluate_history.jsonl"
echo "  ${OUTPUT_ROOT}/logs/seg-random/evaluate_history.jsonl"
