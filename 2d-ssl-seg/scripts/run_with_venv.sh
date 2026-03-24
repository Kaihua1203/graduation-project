#!/usr/bin/env bash
# Activate solo-learn venv and set env for 2d-ssl-seg (SSL + seg).
# Usage: source scripts/run_with_venv.sh   OR   . scripts/run_with_venv.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/solo-learn}"
SOLO_LEARN_DIR="${SOLO_LEARN_DIR:-/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/solo-learn}"

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
  set +u
  source "${VENV_DIR}/bin/activate"
  set -u
  export SOLO_LEARN_DIR
  export PYTHONPATH="${SOLO_LEARN_DIR}:${PYTHONPATH:-}"
  echo "Activated venv: ${VENV_DIR}"
  echo "SOLO_LEARN_DIR=${SOLO_LEARN_DIR}"
else
  echo "Venv not found: ${VENV_DIR}. Set VENV_DIR or create the venv first."
  return 1 2>/dev/null || exit 1
fi
