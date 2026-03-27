---
name: seg-trainer
description: Launch 2d segmentation training jobs in this repository with tmux and scheduled GPUs. Use when the user wants to start one or more segmentation, either with an SSL pretrained encoder or with random initialization, and may provide a custom segmentation config or ask Codex to create a new YAML from the repository templates.
---

# Seg Trainer

Use this skill to convert segmentation training requests into concrete tmux sessions with generated config files and one-time startup verification.

## Inputs

Collect only the runtime values you need:
- `mode`: `ssl` or `random`
- `experiment_name`: required; use it as both tmux session name and `swanlab.experiment_name`
- `encoder_name`: required only for `ssl`; prefer `mocov3`, `simclr`, `vicreg`
- `custom_config`: optional path to an existing seg YAML; if missing, create a new YAML from the repository default template
- `gpu_ids`: optional explicit physical GPU ids like `0,1`; if omitted, auto-schedule two GPUs per job

Default repository templates:
- `ssl`: `2d-ssl-seg/configs/seg/train_ssl_50epochs.yaml`
- `random`: `2d-ssl-seg/configs/seg/train_random_50epochs.yaml`

## Workflow

1. Resolve the target mode and gather one or more jobs.
2. Choose the base config:
- if the user gives a custom seg config, clone that config into a generated YAML
- otherwise clone the mode-specific default template into a generated YAML
3. For `ssl`, resolve the encoder checkpoint path from the encoder name:
- `mocov3-ep80` -> `outputs/encoders/lits_mocov3_ep80_encoder.pth`
- `simclr-ep80` -> `outputs/encoders/lits_simclr_ep80_encoder.pth`
- `vicreg-ep80` -> `outputs/encoders/lits_vicreg_ep80_encoder.pth`
4. Generate a new YAML under `2d-ssl-seg/configs/seg/` and update:
- `pretrained_encoder.enabled`
- `pretrained_encoder.path`
- `output_dir`
- `train.freeze_encoder_epochs`
- `swanlab.experiment_name`
5. Assign two GPUs per job. If the user does not specify GPUs, pick the least-used GPUs from `nvidia-smi` and keep assignments disjoint across jobs in the same launch batch.
6. Launch tmux sessions with the repository-native scripts:
- `2d-ssl-seg/scripts/run_seg_train_ssl_tmux.sh`
- `2d-ssl-seg/scripts/run_seg_train_random_tmux.sh`
7. Verify startup once:
- confirm `tmux` session exists
- capture the last pane lines once for quick failure detection
- check GPU memory/utilization once after launch
8. Do not keep monitoring unless the user explicitly asks.

## Script

Use the bundled launcher:

```bash
python .codex/skills/seg-trainer/scripts/launch_seg_training.py \
  --job mode=ssl,encoder=mocov3-ep80,experiment=seg-ssl-50epochs-mocov3-ep80 \
  --job mode=random,experiment=seg-random-50epochs-baseline \
  --allowed-gpus 0,1,2,3,4,5
```

Optional keys inside each `--job`:
- `mode`
- `experiment`
- `encoder`
- `config`
- `gpu_ids`

Rules:
- `config` may be relative to `2d-ssl-seg/` or absolute
- never edit the user-provided config in place; always write a generated YAML
- if sandbox blocks `tmux` socket or `nvidia-smi`, rerun with escalation
- if a requested tmux session already exists, stop and report the collision instead of reusing it silently

## Output

Report only the launch facts:
- experiment name
- mode
- encoder name if present
- generated config path
- assigned GPUs
- tmux session name
- startup status

Do not claim the training completed. Report only that it was launched and whether startup checks passed.
