---
name: seg-eval
description: Run 2d segmentation evaluations on GPU, including multi-experiment parallel execution via one-subagent-per-GPU, and return aggregated metrics/conclusions from JSONL logs.
---

# Seg Eval

## Inputs

Collect only the minimum runtime variables:
- `project_dir`: default `/data3/kaihua.wen/code/graduation-project/2d-ssl-seg`
- `eval_script`: e.g. `scripts/run_seg_eval_lits.sh` (later can be `scripts/run_seg_eval_acdc.sh`)
- `runs`: one or more `<target>:<experiment_name>` specs
- `target` must be `ssl` or `random`
- `experiment_name` examples: `seg-ssl-100epochs`, `seg-random-100epochs`

Assume checkpoints follow repository convention:
- `ssl`: `outputs/seg_ssl/<experiment_name>/best_model.pt`
- `random`: `outputs/seg_random/<experiment_name>/best_model.pt`

## Workflow

1. Parse user variables into run specs (`ssl:...`, `random:...`).
2. Detect available GPUs and schedule runs:
- if only one run exists, execute directly on one GPU
- if multiple runs exist, split them into multiple subagents and assign one GPU per subagent
- each subagent must set its own `CUDA_VISIBLE_DEVICES=<gpu_id>` and execute exactly one run at a time
3. Execute evaluation with skill script per run:

```bash
python /home/jupyter-wenkaihua/data3_link/kaihua.wen/code/graduation-project/.codex/skills/seg-eval/scripts/run_gpu_eval.py \
  --project-dir /data3/kaihua.wen/code/graduation-project/2d-ssl-seg \
  --eval-script scripts/run_seg_eval_lits.sh \
  --run random:seg-random-100epochs \
  --run ssl:seg-ssl-100epochs
```

4. If sandbox blocks multiprocessing with `SemLock Permission denied`, rerun with escalation.
5. Each subagent returns run status and latest metrics sourced from the tail of JSONL.
6. Main agent aggregates all subagent outputs and returns a concise conclusion:
- always include `dice`, `iou`, `hd95`, `timestamp`, `checkpoint`, and JSONL path per run
- when both `ssl` and `random` exist, state which one is better (higher `dice`/`iou`, lower `hd95`)

## Output Contract

Respond in this structure:
1. `Executed runs`: list each target and experiment name.
2. `Metrics`: one line per run with `dice / iou / hd95` and log path.
3. `Conclusion`: direct comparison result if both runs are present; otherwise single-run summary.

Keep response concise and avoid extra narrative.
