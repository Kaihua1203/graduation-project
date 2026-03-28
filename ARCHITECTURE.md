# graduation-project Architecture

This document describes the current directory layout and module responsibilities of `graduation-project` (as of 2026-03-28).

## 1) Repository Root

```text
graduation-project/
├── 2d-gen/
├── 2d-ssl-seg/
├── docs/
├── AGENTS.md
├── ARCHITECTURE.md
├── CLAUDE.md
├── README.md
├── .gitignore
└── .claude/
```

- `2d-gen/`: 2D medical image generation project for LoRA fine-tuning, local inference, and generation evaluation.
- `2d-ssl-seg/`: SSL pretraining + segmentation training + evaluation project.
- `docs/`: Project documentation for diffusers training references and related notes.
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

## 3) `2d-gen` Project Structure

```text
2d-gen/
├── src/
│   ├── common/
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── diffusers_import.py
│   │   ├── runtime.py
│   │   └── types.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── manifest_builder.py
│   ├── train/
│   │   ├── adapters/
│   │   │   ├── base.py
│   │   │   ├── stable_diffusion.py
│   │   │   ├── sdxl.py
│   │   │   ├── flux.py
│   │   │   └── qwenimage.py
│   │   ├── base_trainer.py
│   │   └── run_train.py
│   ├── infer/
│   │   └── generator.py
│   ├── eval/
│   │   ├── metrics.py
│   │   └── run_evaluate.py
│   └── tests/
├── configs/
│   ├── README.md
│   ├── train_sd_lora.yaml
│   ├── train_sd_lora_example.yaml
│   ├── infer_sd_example.yaml
│   └── eval_example.yaml
├── scripts/
│   ├── run_with_venv.sh
│   ├── run_train.sh
│   ├── run_infer.sh
│   ├── run_eval.sh
│   └── run_build_manifest.sh
├── outputs/
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## 4) `2d-gen` Module Responsibilities

- `src/common/`
  - Shared config loading, filesystem/runtime helpers, constants, and lightweight typed containers.
  - `diffusers_import.py` centralizes optional local diffusers source resolution.

- `src/data/`
  - `dataset.py`: JSONL manifest-backed image/prompt dataset loading for training.
  - `manifest_builder.py`: Builds manifests from paired image and prompt directories.

- `src/train/`
  - `run_train.py`: CLI entry point for generation training.
  - `base_trainer.py`: Shared training loop, dataloader construction, checkpoint writing, and loss summary output.
  - `adapters/`: Model-family-specific integration layer.
    - `base.py`: Adapter interface and validation hooks.
    - `stable_diffusion.py`: Implemented Stable Diffusion LoRA training path.
    - `sdxl.py`, `flux.py`, `qwenimage.py`: Validation and interface stubs for future model-family support.

- `src/infer/`
  - `generator.py`: Local-path-only inference entrypoint for base model + LoRA adapter generation.

- `src/eval/`
  - `metrics.py`: Generation metric implementations such as `FID`, `IS`, `CLIP-I`, and `CLIP-T`.
  - `run_evaluate.py`: Evaluation runner over generated outputs and manifests.

- `src/tests/`
  - Focused smoke and validation tests for config loading, datasets, manifest building, metrics, lazy imports, and adapter shape checks.

- `configs/`
  - YAML examples for training, inference, and evaluation runs.
  - `train_sd_lora.yaml` is the current concrete Stable Diffusion LoRA training config.

- `scripts/`
  - Thin shell wrappers that activate the selected venv and launch train/infer/eval or manifest-building workflows.

- `outputs/`
  - Runtime artifacts such as LoRA checkpoints, generated images, and evaluation summaries.

## 5) `2d-ssl-seg` Module Responsibilities

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

## 6) Scope Boundaries

- Dataset directories (LiTS2017) are external to the repository and follow the path conventions documented in `2d-ssl-seg/README.md`.
- Generation model weights, LoRA adapters, generated images, and metric artifacts are not intended to be committed; they belong under `2d-gen/outputs/`.
- `docs/diffusers/reference/` contains upstream-style training reference scripts for analysis and implementation guidance, not the main project runtime entrypoints.
- `.git/` and `__pycache__/` are version-control/runtime internals and are not considered part of the business architecture.
