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
- Validate changes by running the relevant training script and checking output artifacts:
  - `outputs/seg_ssl/**/metrics_history.json`
  - `outputs/seg_ssl/**/best_model.pt`

## Commit & Pull Request Guidelines
- This repo has no existing git history to infer a convention. Use short, imperative subjects:
  - Example: `add vicreg config for lits`.
- PRs should include:
  - A concise summary of changes and rationale.
  - Links to related issues (if any).
  - Key command(s) run and resulting metrics or logs.
  - Config files and output paths used for reproducibility.

## Security & Configuration Tips
- Do not commit datasets, checkpoints, or large artifacts; keep them under `outputs/`.
- If you add new data paths or secrets, document them in `2d-ssl-seg/README.md` and keep them out of git.
