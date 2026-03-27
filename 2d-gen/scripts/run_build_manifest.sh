#!/usr/bin/env bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/run_with_venv.sh"

IMAGES_DIR="${1:?usage: bash scripts/run_build_manifest.sh <images_dir> <prompts_dir> <output_jsonl>}"
PROMPTS_DIR="${2:?usage: bash scripts/run_build_manifest.sh <images_dir> <prompts_dir> <output_jsonl>}"
OUTPUT_PATH="${3:?usage: bash scripts/run_build_manifest.sh <images_dir> <prompts_dir> <output_jsonl>}"

python "$SCRIPT_DIR/../src/data/manifest_builder.py" \
  --images-dir "$IMAGES_DIR" \
  --prompts-dir "$PROMPTS_DIR" \
  --output-path "$OUTPUT_PATH"
