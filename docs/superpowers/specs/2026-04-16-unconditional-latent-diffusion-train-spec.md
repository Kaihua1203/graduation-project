# Unconditional Latent Diffusion Training Spec

## Topic

This spec defines the training worktree for standalone unconditional latent diffusion in `2d-gen`.

This worktree owns:

- `2d-gen/src/uncond_ldm/config.py`
- `2d-gen/src/uncond_ldm/dataset.py`
- `2d-gen/src/uncond_ldm/trainer.py`
- `2d-gen/src/train/run_train_uncond_ldm.py`
- train YAMLs

## Goal

Implement a standalone unconditional latent diffusion training path that:

- reuses a fixed pretrained Stable Diffusion `AutoencoderKL`
- trains a pure `UNet2DModel` from scratch in latent space
- uses image-only datasets with existing repo preprocessing semantics
- saves resumable checkpoints and a stable exported bundle for inference

## Fixed Context

- data type: 2D medical image slices
- input resolution: `512x512`
- grayscale source images are still converted to `RGB`
- image tensors are normalized to `[-1, 1]`
- no prompt text, tokenizer, or text encoder exists anywhere in this path
- the VAE is fixed and kept in eval mode
- the unconditional path must stay outside the existing prompt-conditioned `family` abstraction

## Module Layout

The training worktree is responsible for the following package shape:

```text
2d-gen/src/uncond_ldm/
  __init__.py
  config.py
  dataset.py
  trainer.py
```

It also owns the dedicated entrypoint:

```text
2d-gen/src/train/run_train_uncond_ldm.py
```

## Responsibilities

`uncond_ldm/config.py`

- load and validate unconditional training config structure
- keep schema separate from prompt-conditioned train configs

`uncond_ldm/dataset.py`

- read image-only manifests or image directories
- preserve current medical image preprocessing semantics
- return image tensors plus image path metadata only

`uncond_ldm/trainer.py`

- construct `AutoencoderKL`, `UNet2DModel`, optimizer, scheduler, and accelerator state
- run the latent diffusion training loop
- save training checkpoints
- export the final inference bundle
- optionally generate lightweight validation samples during training

`run_train_uncond_ldm.py`

- load the training YAML
- validate config
- construct trainer state
- run training and export

## Training Data Contract

The training dataset should follow the same effective preprocessing already used by current medical experiments:

1. read exported PNG slices
2. convert each image to `RGB`
3. resize to `512`
4. apply the configured crop policy
5. optionally apply horizontal flip
6. convert to tensor
7. normalize to `[-1, 1]`

The trainer should support the same manifest style as current generation training so dataset preparation stays aligned with existing exports.

## Training Model Contract

The latent diffusion loop should be:

1. encode input images to latents with the fixed VAE
2. scale latents with the VAE scaling factor
3. sample Gaussian noise
4. sample discrete timesteps
5. add noise with the training scheduler
6. run `UNet2DModel(noisy_latents, timesteps)`
7. predict epsilon
8. optimize MSE between predicted and sampled noise

Only the unconditional `UNet2DModel` is trainable.

- `AutoencoderKL` stays fixed
- no LoRA path is required in the first version
- no text-conditioning modules exist

## First-Version Hyperparameter Shape

The first implementation should keep the baseline narrow:

- `prediction_type: epsilon`
- `sample_size: 64` latent spatial size for `512x512` inputs with 8x VAE downsampling
- `in_channels: 4`
- `out_channels: 4`
- standard DDPM training noise schedule

The exact UNet width and block layout can be exposed in YAML, but this path should assume one unconditional latent diffusion baseline rather than a broad model zoo.

## Validation During Training

Validation should stay lightweight and image-only:

- generate a fixed small number of unconditional samples at configured steps
- save them under a validation output directory
- optionally log them to the configured tracker

Validation must not require prompts or prompt files.

## Checkpoint And Export Contract

### Training Checkpoints

Training checkpoints should capture:

- unconditional UNet weights
- optimizer state
- LR scheduler state
- accelerator or distributed state
- global step metadata

### Final Export Bundle

The final export written by training must contain:

- `unet` weights
- inference scheduler config
- metadata about the reused VAE source
- training summary JSON

Inference is allowed to depend on this exported bundle, but it must not depend on trainer internals or intermediate checkpoint layout.

## YAML Deliverables

Add dataset-specific training YAMLs for:

- `lits`
- `m4raw_t1`
- `m4raw_t2`
- `m4raw_flair`

Each train YAML should include:

- model section
- data section
- train section
- validation section
- logging section
- distributed section

The key differences from existing train YAMLs are:

- no prompt-conditioning config
- no LoRA config
- no model family selection
- an explicit unconditional model type instead

Dataset-specific YAMLs should vary only in:

- dataset manifest paths
- output directories
- experiment names
- optional dataset-specific batch sizes

## Interfaces To Other Worktrees

- training must finalize the export bundle layout before inference implementation starts
- training should not depend on evaluation code
- training should avoid edits to the prompt-conditioned adapter stack

## Testing Requirements

At minimum, this worktree should add or cover:

- unconditional config validation
- image-only dataset loading and output tensor shape
- VAE encode and UNet forward smoke path
- checkpoint save or load smoke coverage if practical

If runtime constraints are tight, use tiny fixtures and mocked lightweight modules.

## Success Criteria

This worktree is complete when the repo can:

1. train an unconditional latent diffusion UNet on medical slice PNGs at `512x512`
2. resume from training checkpoints
3. export a stable inference bundle consumed without trainer internals
4. run ready-to-use train YAMLs for `lits`, `m4raw_t1`, `m4raw_t2`, and `m4raw_flair`
