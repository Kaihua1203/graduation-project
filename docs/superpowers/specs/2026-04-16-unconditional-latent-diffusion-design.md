# Unconditional Latent Diffusion Design

## Topic

This parent spec defines the shared contract for adding a standalone unconditional latent diffusion workflow to `2d-gen` for 2D medical image slices.

The implementation is intentionally split into three child specs so they can be executed in separate sessions and worktrees with minimal overlap:

- [Training spec](./2026-04-16-unconditional-latent-diffusion-train-spec.md)
- [Inference spec](./2026-04-16-unconditional-latent-diffusion-infer-spec.md)
- [Evaluation spec](./2026-04-16-unconditional-latent-diffusion-eval-spec.md)

## Shared Goal

Add a standalone unconditional latent diffusion workflow that:

- uses a fixed pretrained Stable Diffusion `AutoencoderKL`
- trains a pure `UNet2DModel` from scratch in latent space
- performs unconditional sampling without prompts
- evaluates generated images with image-only metrics already used in the repo
- provides ready-to-run YAML configs for `lits`, `m4raw_t1`, `m4raw_t2`, and `m4raw_flair`

## Shared Project Context

The current `2d-gen` codebase is organized around prompt-conditioned generation families such as Stable Diffusion, SD3, SDXL, Flux, and QwenImage. The shared training path assumes text conditioning through model adapters and prompt-bearing manifests.

The current medical image data path already matches the basic preprocessing constraints needed for this project:

- raw medical slices are exported as grayscale PNG files
- the training dataset loader converts each image to `RGB`
- image tensors are normalized to `[-1, 1]`
- the default training resolution already supports `512x512`

The unconditional path should therefore reuse the existing medical image asset format and preprocessing semantics, but it should not be forced into the prompt-conditioned family abstraction.

## Fixed Decisions

- data type: 2D medical image slices
- input resolution: `512x512`
- image channel handling: keep the current repo behavior and convert grayscale images to `RGB`
- latent encoder: reuse a pretrained Stable Diffusion `AutoencoderKL`
- diffusion model: train a pure unconditional `UNet2DModel`
- training initialization: train the diffusion UNet from scratch
- conditioning: none
- evaluation pairing mode: generated outputs must align one-to-one with real-image filenames so that `CLIP-I` and `BiomedCLIP-I` remain valid

## Non-goals

- do not integrate unconditional generation into the existing `family` adapter framework
- do not support prompt-conditioned generation in this workflow
- do not train or fine-tune the VAE
- do not redesign the existing medical dataset export pipeline
- do not preserve `CLIP-T` or `BiomedCLIP-T` for unconditional experiments

## Why This Stays Outside The Existing Family Framework

The current family abstraction is built around prompt-conditioned diffusion pipelines. Its training interfaces assume prompt text exists, its inference path reads prompt files, and its evaluation path expects generated manifest records with prompts.

Forcing unconditional latent diffusion into that structure would create artificial branches such as fake text-conditioning objects, no-op text encoders, unused prompt columns, and prompt-dependent inference settings that do not apply.

The cleaner design is a standalone unconditional path that reuses lower-level repository utilities where they still fit:

- YAML loading and validation patterns
- runtime directory handling
- existing grayscale-to-RGB dataset behavior
- existing image-only evaluation metric implementations

## Worktree Split

### Worktree 1: Training

Child spec:

- [2026-04-16-unconditional-latent-diffusion-train-spec.md](./2026-04-16-unconditional-latent-diffusion-train-spec.md)

Owns:

- `2d-gen/src/uncond_ldm/config.py`
- `2d-gen/src/uncond_ldm/dataset.py`
- `2d-gen/src/uncond_ldm/trainer.py`
- `2d-gen/src/train/run_train_uncond_ldm.py`
- train YAMLs

### Worktree 2: Inference

Child spec:

- [2026-04-16-unconditional-latent-diffusion-infer-spec.md](./2026-04-16-unconditional-latent-diffusion-infer-spec.md)

Owns:

- `2d-gen/src/uncond_ldm/pipeline.py`
- `2d-gen/src/uncond_ldm/checkpointing.py`
- `2d-gen/src/infer/run_infer_uncond_ldm.py`
- infer YAMLs

### Worktree 3: Evaluation

Child spec:

- [2026-04-16-unconditional-latent-diffusion-eval-spec.md](./2026-04-16-unconditional-latent-diffusion-eval-spec.md)

Owns:

- `2d-gen/src/eval/run_evaluate_uncond.py`
- any small shared evaluation helpers needed for unconditional mode
- eval YAMLs
- tests for eval contract and filename alignment

## Shared Interface Rules

To keep the three worktrees independent:

- training must define the final exported model layout early
- inference must consume only that exported layout and not training internals
- evaluation must consume only generated image directories and not training internals
- avoid shared mutable edits to the existing text-conditioned adapter stack

The main coordination point is the exported model bundle contract. That contract should be treated as fixed before parallel implementation starts.

## Export Bundle Contract

The final exported inference bundle must contain:

- `unet` weights
- inference scheduler config
- metadata about the reused VAE source
- training summary JSON

The VAE itself may be referenced by path or config rather than duplicated into every run output if the codebase already assumes local pretrained assets.

## Recommended Implementation Order

Even with parallel worktrees, the dependency order should remain:

1. fix the config and export contract
2. implement training
3. implement inference against the fixed export contract
4. implement unconditional evaluation
5. add YAMLs and tests

Parallel work is still viable as long as step 1 is resolved first and treated as stable.

## Success Criteria

The overall design is successful when the repository can:

1. train an unconditional latent diffusion UNet on medical slice PNGs at `512x512`
2. generate unconditional samples with filenames aligned to a reference directory
3. evaluate those samples with `FID`, `IS`, `CLIP-I`, `Med-FID`, and `BiomedCLIP-I`
4. run dataset-specific experiments using ready-made YAMLs for `lits`, `m4raw_t1`, `m4raw_t2`, and `m4raw_flair`
