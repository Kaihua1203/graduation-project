# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

- **solo-learn**: installed in dev mode from `/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/solo-learn`; venv at `/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/graduation-project/kaihua.wen/venv/solo-learn`
- **Outputs** (`outputs/`) are git-ignored; do not commit checkpoints or datasets

### Logging

SwanLab (not W&B) is used for experiment tracking. Project name: `"2d-ssl-seg"`.

## Coding conventions

- Python: 4-space indentation, PEP 8, `snake_case` functions/variables, `PascalCase` classes
- YAML configs: `lower_snake_case` keys, group related settings together
- No formatter/linter configured — keep changes small and consistent with nearby code
- Commits: conventional commits with short imperative subjects (`feat:`, `fix:`, etc.). If Claude creates and pushes a commit, prefix the subject with `Claude` — e.g., `fix: Claude xxx`
- PRs: include summary, commands run, resulting metrics, and config paths used. If Claude creates the PR, note "Created by Claude" in the description.
