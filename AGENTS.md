# Repository Guidelines

## Project Structure & Module Organization
See `ARCHITECTURE.md` for the current repository layout and module map.

## Build, Test, and Development Commands
Run commands from `2d-ssl-seg/` unless noted.

- `source scripts/run_with_venv.sh`: setup environment.
- `bash scripts/run_ssl_pretrain.sh`: run SSL pretraining.
- `bash scripts/run_extract_encoder.sh /path/to/ssl_checkpoint.ckpt`: export encoder weights.
- `bash scripts/run_seg_train_ssl.sh` / `bash scripts/run_seg_train_random.sh`: train segmentation (SSL vs random baseline).
- `bash scripts/run_seg_eval_lits.sh ssl` / `bash scripts/run_seg_eval_lits.sh random`: evaluate on LiTS.

## Testing Guidelines
- There is no automated test suite in this repo yet.
- Validate with end-to-end train + eval:
  - Check checkpoints: `outputs/seg_ssl/seg-ssl/best_model.pt` and `outputs/seg_random/seg-random/best_model.pt`.
  - Run: `bash scripts/run_seg_eval_lits.sh ssl` and `bash scripts/run_seg_eval_lits.sh random`.
  - Verify logs: `outputs/logs/seg-ssl/evaluate_history.jsonl` and `outputs/logs/seg-random/evaluate_history.jsonl`.
  - Ensure each JSONL record has `dice`, `iou`, and `hd95` (3-decimal rounded).

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard PEP 8 naming.
- Use `snake_case` for functions/variables, `PascalCase` for classes.
- Config files are YAML; keep keys lower_snake_case and group related settings together.
- No formatter/linter is configured; keep changes small and consistent with nearby code.

## Commit & Pull Request Guidelines
- Use conventional commits with short, imperative subjects:
  - Examples: `feat: add vicreg config for lits`, `fix: handle empty lits directory`.
- When Codex makes functional changes (e.g., adding/removing evaluation pipeline pieces, training algorithms, or other behavior changes), or performs substantial multi-file/code-volume edits, create at least one commit for that work and push it in the github.
- PRs should include:
  - A concise summary of changes and rationale.
  - Links to related issues (if any).
  - Key command(s) run and resulting metrics or logs.
  - Config files and output paths used for reproducibility.

## Security & Configuration Tips
- Do not commit datasets, checkpoints, or large artifacts; keep them under `outputs/`.
- If you add new data paths or secrets, document them in `2d-ssl-seg/README.md` and keep them out of git.

## solo-learn Progressive Disclosure
Scope in this repo:
- use `solo-learn` only for SSL pretraining and encoder/backbone weight export for downstream segmentation.

Read in order:
1. `docs/solo-learn/solo-learn-core.md`
2. `docs/solo-learn/methods_and_backbones.md`