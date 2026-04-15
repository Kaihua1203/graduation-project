# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

Read `ARCHITECTURE.md` for the current repository layout and module map.

## Project Environment Rules

- Before starting implementation work, first align with the user on which project environment or `venv` should be used. Do not assume the environment if the user has not specified it.
- When working in `2d-gen`, use the virtual environment at `/home/jupyter-wenkaihua/data3/kaihua.wen/code/graduation-project_link/kaihua.wen/venv/diffusers` by default.
- When working in `2d-ssl-seg`, activate via `source scripts/run_with_venv.sh`.

## Commands

Run all commands from `2d-ssl-seg/` unless noted.

**Environment setup:**
```bash
source scripts/run_with_venv.sh   # activate venv and set SOLO_LEARN_DIR
python -m pip install -r requirements.txt
```

**Training pipeline (run in order):**
```bash
bash scripts/run_ssl_pretrain.sh                                    # Stage 1: VICReg SSL pretraining
bash scripts/run_extract_encoder.sh /path/to/ssl_checkpoint.ckpt   # Stage 2: extract encoder weights
bash scripts/run_seg_train_ssl.sh                                   # Stage 3: segmentation with SSL encoder
bash scripts/run_seg_train_random.sh                                # Stage 4: random-init baseline
```

**Multi-GPU:**
```bash
GPU_IDS=0,1 bash scripts/run_ssl_pretrain.sh
```

**Custom experiment name:**
```bash
EXPERIMENT_NAME="my-run" bash scripts/run_ssl_pretrain.sh
```

**Validation:** No automated test suite. Validate by running training and checking:
- `outputs/seg_ssl/**/metrics_history.json`
- `outputs/seg_ssl/**/best_model.pt`

## Test Execution Rules

- For work in `2d-gen`, always run test commands with `PYTHONPATH=2d-gen/src` so imports resolve against the repository source tree.
- Prefer running tests from the repository root with the environment set inline, for example: `PYTHONPATH=2d-gen/src pytest ...`
- Do not assume a specific virtual environment. If a command depends on one, first align with the user per Project Environment Rules above.

## Architecture

This repo contains one project (`2d-ssl-seg/`) — a 4-stage medical imaging pipeline for liver segmentation using Self-Supervised Learning (SSL) on the LiTS2017 dataset.

### Pipeline stages

1. **SSL Pretraining** (`src/run_ssl_pretrain.py`): VICReg on unlabeled data using the [solo-learn](https://github.com/vturrisi/solo-learn) framework and PyTorch Lightning. Backbone: ResNet18. Outputs checkpoints to `outputs/ssl/trained_models/vicreg/<run_id>/`.

2. **Encoder Extraction** (`src/extract_backbone.py`): Loads a solo-learn `.ckpt` and strips out `backbone.*`/`encoder.*` weights (handles `module.` prefixes from DataParallel). Saves to `outputs/encoders/lits_vicreg_encoder.pth`.

3. **Segmentation Fine-tuning** (`src/train_segmentation.py`): MONAI `FlexibleUNet` with ResNet18 backbone loaded non-strictly from the extracted encoder. Two-stage training: freeze encoder for N epochs, then joint fine-tune with separate LRs for encoder vs. decoder. 3-class output (background/liver/tumor). Metrics: Dice, IoU, HD95.

4. **Random-Init Baseline**: Same segmentation script, different config (`configs/seg/train_random.yaml`) — no pretrained weights, `freeze_encoder_epochs=0`.

### Configuration

All hyperparameters live in `configs/`:
- `configs/ssl/vicreg_lits.yaml` — VICReg method, LARS optimizer, batch size, epochs, SwanLab logging
- `configs/ssl/augmentations/lits_asymmetric.yaml` — two asymmetric augmentation branches (medical-appropriate: limited color jitter, no solarization)
- `configs/seg/train_ssl.yaml` — segmentation with SSL encoder path
- `configs/seg/train_random.yaml` — segmentation baseline
- `configs/CONFIG_GUIDE.md` — full parameter reference

### External dependencies and data

- **solo-learn**: installed in dev mode from `/home/jupyter-wenkaihua/data3/kaihua.wen/code/graduation-project_link/kaihua.wen/code/solo-learn`; venv at `/home/jupyter-wenkaihua/data3/kaihua.wen/code/graduation-project_link/kaihua.wen/code/graduation-project/kaihua.wen/venv/solo-learn`
- **Outputs** (`outputs/`) are git-ignored; do not commit checkpoints or datasets

### Logging

SwanLab (not W&B) is used for experiment tracking. Project name: `"2d-ssl-seg"`.

## Coding Conventions

- Python: 4-space indentation, PEP 8, `snake_case` functions/variables, `PascalCase` classes
- YAML configs: `lower_snake_case` keys, group related settings together
- No formatter/linter configured — keep changes small and consistent with nearby code

## Commit & Pull Request Guidelines

- Use conventional commits with short imperative subjects (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, etc.).
- Always include `fixes #<number>` or `closes #<number>` in the commit message when there is a related issue or PR.
- Never include unrelated local changes in a commit. Stage and commit only the files that belong to the current task.
- Claude may commit directly for small, low-risk, self-contained changes (docs, comments, formatting, typo fixes, narrowly scoped housekeeping).
- Prefer a PR for behavior changes, multi-file refactors, new features, non-trivial bug fixes, changes to training/evaluation logic, changes that affect public interfaces or configs. If unclear, prefer opening a PR.
- PRs should include:
  - A concise summary of changes and rationale.
  - Links to related issues (if any).
  - Key command(s) run and resulting metrics or logs.
  - Config files and output paths used for reproducibility.
  - If Claude creates the PR, note "Created by Claude" in the description.

## Security & Configuration

- Do not commit datasets, checkpoints, or large artifacts; keep them under `outputs/`.
- If you add new data paths or secrets, document them in `2d-ssl-seg/README.md` and keep them out of git.

## Progressive Disclosure: diffusers

Scope: use these docs when analyzing or implementing training code for diffusion/generation models based on diffusers pipelines.
- diffusers repo: `/home/jupyter-wenkaihua/data3/kaihua.wen/code/graduation-project_link/kaihua.wen/code/diffusers`

Use as a narrowing path — start at the highest-level doc that answers the question, drill down only when needed, stop at the first layer with enough context:

1. `docs/diffusers/training_mapping.md` — cross-model mental model and family split
2. `docs/diffusers/training_architecture.md` — trainer and adapter contract
3. `docs/diffusers/stable_diffusion.md`, `docs/diffusers/stable_diffusion_3.md`, `docs/diffusers/sdxl.md`, `docs/diffusers/flux.md`, `docs/diffusers/qwenimage.md` — family-specific behavior
4. `docs/diffusers/reference/` — upstream scripts and implementation details

## Tool Usage Rules

- NEVER use `sed`/`cat` to read a file or a range of a file. Always use the Read tool (use offset + limit for ranged reads).
- You MUST read every file you modify in full before editing.

## Git Safety Rules

### Staging and committing
- **Only commit files you changed in the current session.**
- Always use `git add <specific-file-paths>` — never `git add -A` or `git add .`.
- Run `git status` before committing and verify you are only staging your own files.

### Forbidden operations
These commands can destroy work and must not be used:
- `git reset --hard` — destroys uncommitted changes
- `git checkout .` — destroys uncommitted changes
- `git clean -fd` — deletes untracked files
- `git stash` — stashes all changes including other sessions' work
- `git add -A` / `git add .` — stages unrelated uncommitted work
- `git commit --no-verify` — bypasses required checks

### Rebase conflicts
- Resolve conflicts only in files you modified.
- If a conflict is in a file you did not modify, abort and ask the user.
- Never force push.
