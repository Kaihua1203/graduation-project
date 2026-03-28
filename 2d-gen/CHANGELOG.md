# Changelog

This file summarizes what Codex changed in `2d-gen` before the planned refactor on 2026-03-28.

## Current Snapshot

- Project path: `2d-gen/`
- Current focus implemented in repo: `Stable Diffusion` LoRA baseline, shared train/infer/eval scaffold, local-weight-only evaluation, and manifest generation from `images/` + `prompts/`
- Current helper scripts:
  - `scripts/run_train.sh`
  - `scripts/run_infer.sh`
  - `scripts/run_eval.sh`
  - `scripts/run_build_manifest.sh`

## Commit History

### `041ada5` `feat: add 2d-gen manifest builder`

Added a manifest-generation flow for training data.

- Added [manifest_builder.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/data/manifest_builder.py)
  - builds JSONL from single-level `images/` and `prompts/`
  - pairs files by the same stem, for example `0001.png` with `0001.txt`
  - writes absolute `image_path`
  - strips prompt whitespace
  - fails on missing prompt, missing image, empty prompt, and duplicate image stem
- Added [run_build_manifest.sh](/data3/kaihua.wen/code/graduation-project/2d-gen/scripts/run_build_manifest.sh)
- Added [test_manifest_builder.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/tests/test_manifest_builder.py)
- Updated [README.md](/data3/kaihua.wen/code/graduation-project/2d-gen/README.md) with manifest-builder usage

### `f5df872` `chore: align 2d-gen env defaults with diffusers venv`

Polished the environment wrapper and README wording.

- Updated [run_with_venv.sh](/data3/kaihua.wen/code/graduation-project/2d-gen/scripts/run_with_venv.sh)
  - default venv points to `/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/diffusers`
  - emits a warning if `VENV_DIR` does not exist
- Tightened README wording to describe the current validated venv snapshot instead of overstating compatibility

### `194f24c` `fix: align 2d-gen env defaults`

Aligned `2d-gen` environment metadata with the dedicated diffusers venv.

- Updated [requirements.txt](/data3/kaihua.wen/code/graduation-project/2d-gen/requirements.txt)
- Updated [README.md](/data3/kaihua.wen/code/graduation-project/2d-gen/README.md)
- Updated [run_with_venv.sh](/data3/kaihua.wen/code/graduation-project/2d-gen/scripts/run_with_venv.sh)

This commit established the dedicated runtime path and documented how to override it with `VENV_DIR`.

### `1c7831a` `feat: add 2d-gen scaffold and sd baseline`

Initial `2d-gen` project creation.

Added the baseline project structure and first working vertical slice:

- project scaffold, configs, runtime helpers, and shell entrypoints
- manifest-based dataset loader in [dataset.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/data/dataset.py)
- shared trainer shell in [base_trainer.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/train/base_trainer.py)
- `Stable Diffusion` LoRA adapter in [stable_diffusion.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/train/adapters/stable_diffusion.py)
- inference entrypoint in [generator.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/infer/generator.py)
- evaluation metrics in [metrics.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/eval/metrics.py)
- adapter skeletons for:
  - [sdxl.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/train/adapters/sdxl.py)
  - [flux.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/train/adapters/flux.py)
  - [qwenimage.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/train/adapters/qwenimage.py)
- initial tests under [src/tests](/data3/kaihua.wen/code/graduation-project/2d-gen/src/tests/test_config_and_dataset.py)

## Runtime Context Confirmed During This Iteration

This context was validated during terminal checks, even though not every detail is encoded in git history.

- Dedicated venv used for `2d-gen`: `/home/jupyter-wenkaihua/data3_link/kaihua.wen/venv/diffusers`
- The CLIP evaluation path had a runtime incompatibility under `transformers==5.4.0`
- Rolling back the local environment to:
  - `transformers==4.57.3`
  - `huggingface-hub==0.36.2`
  resolved the `CLIP-I` / `CLIP-T` runtime issue in local smoke tests
- A validated local environment snapshot from the most recent checks was:
  - `torch==2.11.0+cu128`
  - `torchvision==0.26.0+cu128`
  - `diffusers==0.37.1`
  - `transformers==4.57.3`
  - `huggingface_hub==0.36.2`
  - `accelerate==1.13.0`
  - `peft==0.18.1`
  - `numpy==2.2.6`
  - `scipy==1.15.3`

## Validation History

Validation steps run during the latest Codex work included:

- import checks for:
  - `train.base_trainer`
  - `train.adapters.stable_diffusion`
  - `infer.generator`
  - `eval.metrics`
  - `data.dataset`
- full unit test suite
- local metric smoke checks for:
  - `compute_clip_t`
  - `compute_clip_i`
  - `evaluate_generation_quality`
- manifest-builder smoke test through [run_build_manifest.sh](/data3/kaihua.wen/code/graduation-project/2d-gen/scripts/run_build_manifest.sh)

The most recent recorded test status before this changelog was:

- `18/18` unit tests passing

## Likely Refactor Targets Tomorrow

If the goal is cleanup before adding new functionality, these are the most obvious pressure points:

- [metrics.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/eval/metrics.py)
  - currently contains multiple metrics and model-loading paths in one file
- [stable_diffusion.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/train/adapters/stable_diffusion.py)
  - first real adapter, likely the template for later model families
- [base_trainer.py](/data3/kaihua.wen/code/graduation-project/2d-gen/src/train/base_trainer.py)
  - shared control flow for future adapters
- environment documentation in [README.md](/data3/kaihua.wen/code/graduation-project/2d-gen/README.md)
  - should stay in sync with the actual validated venv when dependencies move

## Immediate Next Functional Area

The most natural next implementation target after the current baseline is:

- full `SDXL` training and inference support on top of the existing adapter framework
