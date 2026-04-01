# 2D Gen

End-to-end 2D medical image generation project for:

1. LoRA fine-tuning of text-to-image foundation models.
2. Standalone SD1.5 DreamBooth LoRA fine-tuning with local instance/class image folders.
3. Local-path-only inference with exported LoRA adapters.
4. Generation-side quality evaluation with `FID`, `IS`, `CLIP-I`, and `CLIP-T`.

## Current Status

Implemented in this first milestone:

- project scaffold and config/runtime helpers
- manifest-based dataset loading
- shared trainer shell
- `Stable Diffusion` LoRA training path
- `Stable Diffusion` inference path
- generation-side evaluation metrics with local weights only
- adapter interfaces and validation hooks for `SDXL`, `Flux`, and `Qwen Image`

Planned next:

- full `SDXL` training/inference adapter
- full `Flux` training/inference adapter
- full `Qwen Image` training/inference adapter

## Layout

- `configs/train/`: train configs
- `configs/infer/`: inference configs
- `configs/eval/`: evaluation configs
- `scripts/`: shell wrappers
- `src/common/`: config, paths, shared runtime objects
- `src/data/`: manifest dataset loader
- `src/train/`: base trainer and model adapters
- `src/train/run_dreambooth_sd15.py`: standalone SD1.5 DreamBooth LoRA trainer
- `src/infer/`: generation entrypoint
- `src/eval/`: metric implementations and evaluation runner
- `outputs/train/`: checkpoints and logs for training runs
- `outputs/infer/`: generated images and manifests
- `outputs/eval/`: evaluation metrics and caches

## Dataset Contract

Training data is driven by a JSONL manifest. Each line must contain:

```json
{"image_path": "/abs/path/image.png", "prompt": "liver ct slice"}
```

## Weight Loading Policy

All weights are loaded from local paths only.

Default evaluation paths:

- `DEFAULT_INCEPTION_WEIGHTS_PATH=/home/jupyter-wenkaihua/data3_link/kaihua.wen/download_model/inception_v3/inception_v3_google-0cc3c7bd.pth`
- `DEFAULT_CLIP_MODEL_PATH=/home/jupyter-wenkaihua/data3_link/kaihua.wen/download_model/clip-vit-base-patch32`

Base model paths must also be local diffusers-compatible directories.

## Environment

The default shell wrappers target this venv:

```bash
/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/diffusers
```

You can override it per run with `VENV_DIR=/path/to/venv`.

The current `/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/diffusers` snapshot was validated with:

- `torch>=2.11.0`
- `torchvision>=0.26.0`
- `diffusers>=0.37.1`
- `transformers>=4.57.3`
- `peft>=0.18.1`
- `accelerate>=1.13.0`
- `swanlab>=0.7.13`
- `numpy>=2.2.6`
- `scipy>=1.15.3`
- `Pillow>=12.1.1`

Install `2d-gen/requirements.txt` in your target venv before running training or inference.

If you want to import from the local diffusers source tree instead of an installed package, set:

```bash
export DIFFUSERS_SRC_PATH=/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/diffusers/src
```

## Commands

Run from `2d-gen/`:

```bash
bash scripts/run_train.sh configs/train/train_sd_lora_example.yaml
bash scripts/run_dreambooth_sd15.sh configs/train/train_sd15_dreambooth_example.yaml
bash scripts/run_infer.sh configs/infer/infer_sd_example.yaml
bash scripts/run_infer.sh configs/infer/infer_sd_example.yaml --resume
bash scripts/run_eval.sh configs/eval/eval_example.yaml
```

`scripts/run_train.sh` now wraps `accelerate launch`. Pass extra launcher arguments after the config path:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_train.sh configs/train/train_sd_lora_example.yaml --multi_gpu --num_processes 2
bash scripts/run_dreambooth_sd15.sh configs/train/train_sd15_dreambooth_example.yaml --num_processes 2
```

The train config follows a diffusers-style schema with top-level `model`, `data`, `train`, `validation`, `logging`, and `distributed` sections. Legacy keys such as `model.pretrained_path`, `train.batch_size`, and `train.num_epochs` are rejected.

The standalone SD1.5 DreamBooth path uses its own config file and does not route through `BaseDiffusionTrainer`. It expects local directories for:

- `data.instance_data_dir`
- `data.class_data_dir` when `data.with_prior_preservation: true`

The DreamBooth trainer preserves the official diffusers instance/class-image flow, saves LoRA checkpoints under `<output_dir>/checkpoint-*`, and writes final weights to `<output_dir>/final_lora`.

Validation is controlled by:

- `validation.validation_prompt`
- `validation.num_validation_images`
- `validation.validation_steps`

When `logging.report_to: swanlab`, training logs include scalar metrics plus validation images. Use `logging.project_name` and `logging.experiment_name` to name the SwanLab project and run.

Current runtime limits:

- `distributed.ddp_backend` must remain `nccl`
- `train.optimizer.use_8bit_adam` is parsed but intentionally rejected for now

Build a training manifest from single-level `images/` and `prompts/` directories:

```bash
bash scripts/run_build_manifest.sh \
  /absolute/path/to/images \
  /absolute/path/to/prompts \
  /absolute/path/to/train_manifest.jsonl
```

The builder matches files by the same stem, for example `images/0001.png` with `prompts/0001.txt`.
It only scans the first directory level and writes absolute `image_path` entries into the JSONL.

Evaluation notes:

- `bash scripts/run_eval.sh <config.yaml>` writes a timestamped metrics file based on `eval.output_path`, for example `metrics_20260331_123456.json`, so repeated runs do not overwrite each other.
- `eval.real_inception_cache_dir` stores cached real-image Inception features/probabilities for reuse across runs. The first run builds the cache; later runs with the same real set and Inception weights reuse it.
- `eval.num_workers` now controls eval-side image loading workers for both Inception and CLIP metrics.

Override the venv when needed:

```bash
VENV_DIR=/some/other/venv bash scripts/run_train.sh configs/train/train_sd_lora_example.yaml
```

Single-GPU execution still works without `--multi_gpu`; multi-GPU runs must set `CUDA_VISIBLE_DEVICES` and match `--num_processes` to the visible GPU count.
Inference can also run on multiple GPUs by setting `infer.gpu_ids` in the infer YAML. If multiple ids are configured, `scripts/run_infer.sh` automatically launches `accelerate` distributed inference. Use `--resume` to skip already generated `sample_*.png` outputs and continue an interrupted run.

## Metric Assumptions

- `FID`: computed between feature distributions of real and generated images.
- `IS`: computed on generated images only with local inception weights.
- `CLIP-T`: average cosine similarity between generated images and their prompts from the generated manifest.
- `CLIP-I`: average cosine similarity between real and generated image embeddings, paired by matching filenames. Directory sizes and filenames must match.
