#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOLO_LEARN_DIR="${SOLO_LEARN_DIR:-/data3/kaihua.wen/code/solo-learn}"
VENV_DIR="${VENV_DIR:-/data3/kaihua.wen/venv/solo-learn}"
GPU_IDS="${GPU_IDS:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
SSL_CONFIG_NAME="${1:-${SSL_CONFIG_NAME:-vicreg_lits}}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${PROJECT_DIR}/outputs/.cache/matplotlib}"
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${PROJECT_DIR}/outputs/.cache/numba}"

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON="${VENV_DIR}/bin/python"
else
  PYTHON="python"
fi

if [[ "${GPU_IDS}" == *","* ]]; then
  DEVICES="[${GPU_IDS}]"
  STRATEGY="ddp"
  SYNC_BN="true"
else
  DEVICES="[${GPU_IDS}]"
  STRATEGY="auto"
  SYNC_BN="false"
fi

cd "${PROJECT_DIR}"
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

EXTRA_ARGS=()
if [[ -n "${EXPERIMENT_NAME}" ]]; then
  EXTRA_ARGS+=("swanlab.experiment_name=${EXPERIMENT_NAME}")
fi

MPLCONFIGDIR="${MPLCONFIGDIR}" \
NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}" \
SOLO_LEARN_DIR="${SOLO_LEARN_DIR}" \
PYTHONPATH="${SOLO_LEARN_DIR}:${PYTHONPATH:-}" \
"${PYTHON}" src/run_ssl_pretrain.py \
  --config-path "${PROJECT_DIR}/configs/ssl" \
  --config-name "${SSL_CONFIG_NAME}" \
  devices="${DEVICES}" \
  strategy="${STRATEGY}" \
  sync_batchnorm="${SYNC_BN}" \
  "${EXTRA_ARGS[@]}"
