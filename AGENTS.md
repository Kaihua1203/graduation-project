# Repository Guidelines

## Project Structure & Module Organization
This repo contains a single project in `2d-ssl-seg/`.

- `2d-ssl-seg/src/`: Python entrypoints (`run_ssl_pretrain.py`, `extract_backbone.py`, `train_segmentation.py`).
- `2d-ssl-seg/configs/`: YAML configs, including SSL and segmentation presets and `CONFIG_GUIDE.md`.
- `2d-ssl-seg/scripts/`: One-command launchers used in the README.
- `2d-ssl-seg/outputs/`: Training artifacts (ignored by git).

Data paths are external and not stored in the repo; see `2d-ssl-seg/README.md` for expected LiTS2017 directories.

## Build, Test, and Development Commands
Run commands from `2d-ssl-seg/` unless noted.

- `source scripts/run_with_venv.sh`: Activate the recommended venv and path setup.
- `python -m pip install -r requirements.txt`: Install dependencies in your venv.
- `bash scripts/run_ssl_pretrain.sh`: SSL pretraining (VICReg + ResNet).
- `bash scripts/run_extract_encoder.sh /path/to/ssl_checkpoint.ckpt`: Export encoder weights.
- `bash scripts/run_seg_train_ssl.sh`: Segmentation fine-tuning with SSL encoder.
- `bash scripts/run_seg_train_random.sh`: Random-init baseline.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard PEP 8 naming.
- Use `snake_case` for functions/variables, `PascalCase` for classes.
- Config files are YAML; keep keys lower_snake_case and group related settings together.
- No formatter/linter is configured; keep changes small and consistent with nearby code.

## Testing Guidelines
- There is no automated test suite in this repo yet.
- Validate changes with training + evaluation artifacts:
  - Train and confirm checkpoints exist under `output_dir/experiment_name`:
    - `outputs/seg_ssl/seg-ssl/best_model.pt`
    - `outputs/seg_random/seg-random/best_model.pt`
  - Run evaluation pipeline (GPU):
    - `bash scripts/run_seg_eval_lits.sh ssl`
    - `bash scripts/run_seg_eval_lits.sh random`
    - Optional overrides for custom experiment names:
      - `SEG_SSL_EXPERIMENT_NAME=<name>`
      - `SEG_RANDOM_EXPERIMENT_NAME=<name>`
  - Check evaluation logs (JSONL only):
    - `outputs/logs/seg-ssl/evaluate_history.jsonl`
    - `outputs/logs/seg-random/evaluate_history.jsonl`
  - Confirm each JSONL record contains `dice`, `iou`, and `hd95` (rounded to 3 decimals).

## Commit & Pull Request Guidelines
- Use conventional commits with short, imperative subjects:
  - Examples: `feat: add vicreg config for lits`, `fix: handle empty lits directory`.
- When Codex makes functional changes (e.g., adding/removing evaluation pipeline pieces, training algorithms, or other behavior changes), or performs substantial multi-file/code-volume edits, create at least one commit for that work.
- PRs should include:
  - A concise summary of changes and rationale.
  - Links to related issues (if any).
  - Key command(s) run and resulting metrics or logs.
  - Config files and output paths used for reproducibility.

## Security & Configuration Tips
- Do not commit datasets, checkpoints, or large artifacts; keep them under `outputs/`.
- If you add new data paths or secrets, document them in `2d-ssl-seg/README.md` and keep them out of git.
