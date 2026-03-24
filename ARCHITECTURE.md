# graduation-project Architecture

This document describes the current directory layout and module responsibilities of `graduation-project` (as of 2026-03-24).

## 1) Repository Root

```text
graduation-project/
├── 2d-ssl-seg/
├── doc/
├── AGENTS.md
├── ARCHITECTURE.md
├── CLAUDE.md
├── README.md
├── .gitignore
└── .claude/
```

- `2d-ssl-seg/`: Main project directory (SSL pretraining + segmentation training + evaluation).
- `doc/`: Supplemental documentation (currently includes `solo-learn.md`).
- `AGENTS.md`: Codex collaboration and development instructions.
- `ARCHITECTURE.md`: Current architecture documentation (this file).
- `CLAUDE.md`: Claude Code instructions.
- `README.md`: Repository-level overview.
- `.claude/`: Claude related folder (not core business code).

## 2) `2d-ssl-seg` Project Structure

```text
2d-ssl-seg/
├── src/
│   ├── run_ssl_pretrain.py
│   ├── extract_backbone.py
│   ├── train_segmentation.py
│   └── evaluate_segmentation.py
├── configs/
│   ├── CONFIG_GUIDE.md
│   ├── ssl/
│   │   ├── vicreg_lits.yaml
│   │   └── augmentations/
│   └── seg/
│       ├── train_ssl.yaml
│       ├── train_ssl_100epochs.yaml
│       ├── train_random.yaml
│       └── train_random_100epochs.yaml
├── scripts/
│   ├── run_with_venv.sh
│   ├── run_ssl_pretrain.sh
│   ├── run_extract_encoder.sh
│   ├── run_seg_train_ssl.sh
│   ├── run_seg_train_ssl_tmux.sh
│   ├── run_seg_train_random.sh
│   ├── run_seg_train_random_tmux.sh
│   └── run_seg_eval_lits.sh
├── outputs/
│   ├── encoders/
│   ├── logs/
│   ├── seg_ssl/
│   ├── seg_random/
│   ├── ssl/
│   └── swanlab/
├── requirements.txt
└── README.md
```

## 3) Module Responsibilities

- `src/`
  - `run_ssl_pretrain.py`: Entry point for self-supervised pretraining.
  - `extract_backbone.py`: Exports encoder weights from an SSL checkpoint.
  - `train_segmentation.py`: Entry point for 2D segmentation training (SSL-initialized or random-initialized).
  - `evaluate_segmentation.py`: Runs segmentation evaluation on LiTS2017 and writes metric logs.

- `configs/`
  - `ssl/`: SSL pretraining configs (for example, VICReg).
  - `seg/`: Segmentation training configs (random-init and SSL-init variants, including multi-epoch presets).
  - `CONFIG_GUIDE.md`: Config field and usage reference.

- `scripts/`
  - One-command launch scripts that wrap common training and evaluation workflows, including normal and `tmux` modes.

- `outputs/`
  - Training and evaluation artifacts (models, logs, swanlab records, etc.).
  - Typical files include `best_model.pt` and `evaluate_history.jsonl`.

## 4) Scope Boundaries

- Dataset directories (LiTS2017) are external to the repository and follow the path conventions documented in `2d-ssl-seg/README.md`.
- `.git/` and `__pycache__/` are version-control/runtime internals and are not considered part of the business architecture.
