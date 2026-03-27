---
name: ssl-trainer
description: Parse user-provided self-supervised learning pretraining specs and launch GPU training jobs with startup verification. Use when the user wants to start or batch-start SSL pretraining runs for methods such as `mocov3`, `simclr`, or `vicreg`, choose SSL config files under `configs/ssl/`, override pretraining hyperparameters, or schedule tmux sessions for encoder pretraining. Do not use for downstream segmentation fine-tuning with pretrained encoders or random initialization; use `seg-trainer` for those segmentation tasks.
---

# Ssl Trainer

Use this skill to turn model/config requests into concrete GPU training launches and quick startup verification.

## Inputs

Accept one or more job specs in this DSL:

`model:config[#key=value,key=value,...]`

Examples:

- `mocov3:mocov3_lits.yaml`
- `simclr:simclr_lits.yaml`
- `mocov3:mocov3_lits.yaml,simclr:simclr_lits.yaml`
- `mocov3:mocov3_lits.yaml#exp=ssl-mocov3-100e`
- `simclr:simclr_lits.yaml#gpu=1`
- `mocov3:mocov3_lits.yaml#max_epochs=200,batch_size=64`
- `simclr:simclr_lits.yaml#exp=ssl-simclr-test,gpu=2,max_epochs=50`

Also accept reduced forms:

- config-only: `mocov3_lits.yaml` (infer model from YAML `method`)
- name-only config: `mocov3_lits` (resolve to `mocov3_lits.yaml` if unique)

Supported inline keys:

- `exp`: experiment/run name override
- `gpu`: target GPU id
- any trainer/config override such as `max_epochs`, `batch_size`, `lr`

Collect project root from current workspace unless user gives one explicitly.

## Parsing and Validation

1. Split batch specs by top-level commas.
2. Parse each job into `model`, `config`, and optional overrides.
3. Resolve config path with `rg` when needed.
4. Read config and validate method/model consistency.
5. If user omitted model, infer from config `method`.
6. If conflict exists between provided model and config method, stop and report clear mismatch.
7. Build launcher arguments from validated job info.

## Launch Workflow

1. Locate configs and launcher scripts with `rg`.
2. Confirm launcher arguments from script content before executing.
3. Check available GPUs using `nvidia-smi`.
4. Assign one GPU per training job.
5. Start jobs in separate `tmux` sessions, each with:
- explicit `GPU_IDS`
- explicit config name
- dedicated log file path
- optional experiment name override when user requests
- optional config overrides passed as launcher overrides
6. Record session names, command pattern, and log paths.

Prefer repository-native scripts (for example `scripts/run_ssl_pretrain.sh`) over ad-hoc Python entrypoints.

## Startup Verification

Verify only successful startup, not full training completion:

1. Check `tmux ls` for expected sessions.
2. Tail each log and confirm it passed import/config stage and entered trainer startup (for example `Training:` or epoch progress lines).
3. Check `nvidia-smi` for expected GPU memory/utilization increase on assigned devices.

If startup fails, diagnose immediate cause from logs, fix launch environment, and relaunch once.

Do not continuously monitor training after startup is confirmed unless user explicitly asks.

## Output Requirements

Report:

- Which config ran on which GPU.
- Session names.
- Log file paths.
- Current startup status (started/failed and reason).

Do not claim completion of training unless explicitly verified at the end of all epochs.
