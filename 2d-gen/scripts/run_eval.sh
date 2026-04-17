#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/run_with_venv.sh"

CONFIG_PATH="${1:?usage: bash scripts/run_eval.sh <config.yaml>}"

eval_entrypoint="$(
  python - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1]).expanduser().resolve()
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

eval_config = config.get("eval") or {}
if "generated_manifest" in eval_config:
    print("run_evaluate.py")
else:
    print("run_evaluate_uncond.py")
PY
)"

python "$SCRIPT_DIR/../src/eval/$eval_entrypoint" --config "$CONFIG_PATH"
