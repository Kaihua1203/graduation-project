# Unconditional Latent Diffusion Design

## Topic

Design for adding a standalone unconditional latent diffusion training, inference, and evaluation workflow to `2d-gen` for 2D medical image slices.

## Project Context

The current `2d-gen` codebase is organized around text-conditioned image generation families such as Stable Diffusion, SD3, SDXL, Flux, and QwenImage. The shared training path assumes prompt-conditioned generation through model adapters and prompt-bearing manifests.

The current medical image data path already provides the preprocessing constraints required for this project:

- raw medical slices are exported as grayscale PNG files
- the training dataset loader converts each image to `RGB`
- image tensors are normalized to `[-1, 1]`
- the default training resolution already supports `512x512`

This means the new unconditional workflow should reuse the existing medical image asset format and image preprocessing conventions, but it should not be forced into the existing text-to-image family abstraction.

## Goal

Add a standalone unconditional latent diffusion workflow that:

- uses a fixed pretrained `AutoencoderKL` from Stable Diffusion
- trains a pure `UNet2DModel` from scratch in latent space
- performs unconditional sampling without prompts
- evaluates generated images with image-only metrics already used in the repo
- provides ready-to-run YAML configs for `lits`, `m4raw_t1`, `m4raw_t2`, and `m4raw_flair`

## Fixed Decisions

- Data type: 2D medical image slices
- Input resolution: `512x512`
- Image channel handling: keep the current repo behavior and convert grayscale images to `RGB` before model input
- Latent encoder: reuse a pretrained Stable Diffusion `AutoencoderKL`
- Diffusion model: train a pure unconditional `UNet2DModel`
- Training initialization: train the diffusion UNet from scratch
- Conditioning: none
- Evaluation pairing mode: generated outputs must align one-to-one with real-image filenames so that `CLIP-I` and `BiomedCLIP-I` remain valid

## Non-goals

- Do not integrate unconditional generation into the existing `family` adapter framework
- Do not support prompt-conditioned generation in this workflow
- Do not train or fine-tune the VAE
- Do not redesign the existing medical dataset export pipeline
- Do not preserve `CLIP-T` or `BiomedCLIP-T` for unconditional experiments

## Why This Should Stay Outside The Existing Family Framework

The current family abstraction is built around prompt-conditioned diffusion pipelines. Its training interfaces assume prompt text exists, its inference path reads prompt files, and its evaluation path expects generated manifest records with prompts.

Forcing unconditional latent diffusion into that structure would create several artificial branches:

- fake text-conditioning objects
- no-op text encoders
- unused prompt columns in training logic
- prompt-dependent inference configuration that does not apply

The cleaner design is a standalone unconditional path that reuses lower-level repository utilities where they still fit:

- YAML loading and validation patterns
- runtime directory handling
- existing grayscale-to-RGB dataset behavior
- existing evaluation metric implementations for image-only metrics

## Recommended Architecture

### New Module Layout

Add a dedicated unconditional package:

```text
2d-gen/src/uncond_ldm/
  __init__.py
  config.py
  dataset.py
  trainer.py
  pipeline.py
  checkpointing.py
```

Add dedicated entrypoints:

```text
2d-gen/src/train/run_train_uncond_ldm.py
2d-gen/src/infer/run_infer_uncond_ldm.py
2d-gen/src/eval/run_evaluate_uncond.py
```

Add YAMLs:

```text
2d-gen/configs/train/train_uncond_ldm_lits.yaml
2d-gen/configs/train/train_uncond_ldm_m4raw_t1.yaml
2d-gen/configs/train/train_uncond_ldm_m4raw_t2.yaml
2d-gen/configs/train/train_uncond_ldm_m4raw_flair.yaml

2d-gen/configs/infer/infer_uncond_ldm_lits.yaml
2d-gen/configs/infer/infer_uncond_ldm_m4raw_t1.yaml
2d-gen/configs/infer/infer_uncond_ldm_m4raw_t2.yaml
2d-gen/configs/infer/infer_uncond_ldm_m4raw_flair.yaml

2d-gen/configs/eval/eval_uncond_ldm_lits.yaml
2d-gen/configs/eval/eval_uncond_ldm_m4raw_t1.yaml
2d-gen/configs/eval/eval_uncond_ldm_m4raw_t2.yaml
2d-gen/configs/eval/eval_uncond_ldm_m4raw_flair.yaml
```

### Component Responsibilities

`uncond_ldm/config.py`

- load and validate unconditional train/infer/eval config structure
- keep schema separate from prompt-conditioned configs

`uncond_ldm/dataset.py`

- read image-only manifests or image directories
- preserve current image preprocessing semantics used by `2d-gen`
- return only image tensors plus image path metadata

`uncond_ldm/trainer.py`

- construct `AutoencoderKL`, `UNet2DModel`, optimizer, scheduler, and accelerator state
- run latent diffusion training loop
- save checkpoints and a final exported model bundle
- optionally generate validation samples during training

`uncond_ldm/pipeline.py`

- expose a small unconditional generation helper
- sample Gaussian latent noise
- run scheduler denoising with `UNet2DModel`
- decode latents with VAE
- postprocess to images

`uncond_ldm/checkpointing.py`

- save and load unconditional model state cleanly
- separate training checkpoints from final inference export format

### Core Runtime Objects

The training and inference stack only needs:

- `AutoencoderKL`
- `UNet2DModel`
- one training noise scheduler such as `DDPMScheduler`
- one inference scheduler such as `DDIMScheduler` or `DDPMScheduler`

No tokenizer, text encoder, pooled conditioning, or prompt embeddings should exist anywhere in this path.

## Data Handling Design

### Training Data

The new workflow should follow the same effective image preprocessing that current medical experiments already use:

1. read exported PNG slices
2. convert images to `RGB`
3. resize to `512`
4. apply the configured crop policy
5. optionally apply horizontal flip
6. convert to tensor
7. normalize to `[-1, 1]`

The unconditional trainer should support the same manifest style as current generation training so dataset preparation stays consistent with existing exports.

### Why RGB Is Kept

Although the original medical slices are grayscale, the current repository converts images to `RGB` before training. This design keeps that behavior because:

- it matches the current data path for `lits` and `m4raw`
- it is compatible with the reused Stable Diffusion `AutoencoderKL`
- it avoids introducing a second image-channel convention for the repo

This is a pragmatic compatibility choice, not a claim that the reused VAE is medically optimal.

## Training Design

### Model Contract

Training should implement the standard latent diffusion loop:

1. encode input images to latents with the fixed VAE
2. scale latents with the VAE scaling factor
3. sample Gaussian noise
4. sample discrete timesteps
5. add noise to clean latents with the training scheduler
6. run `UNet2DModel(noisy_latents, timesteps)`
7. predict epsilon
8. optimize MSE loss between predicted and sampled noise

### Trainable Parameters

Only the unconditional `UNet2DModel` is trainable.

- `AutoencoderKL` is fixed and kept in eval mode
- no LoRA path is required in the first version
- no text-conditioning modules exist

### Suggested First-Version Hyperparameter Shape

The first implementation should keep the model and loss simple:

- `prediction_type: epsilon`
- `sample_size: 64` latent spatial size for `512x512` inputs with 8x VAE downsampling
- `in_channels: 4`
- `out_channels: 4`
- standard DDPM training noise schedule

The exact UNet width and block layout can be exposed in YAML, but the code path should assume one unconditional latent diffusion baseline rather than a broad model zoo.

### Validation During Training

Validation should be image-only and lightweight:

- generate a fixed small number of unconditional samples at configured steps
- save them under a validation output directory
- optionally log them to the configured tracker

Validation should not require prompts or prompt files.

## Inference Design

### Input Contract

Unconditional inference should not accept prompts. It should accept:

- trained model path
- VAE path
- reference image directory
- output directory
- number of inference steps
- batch size
- seed
- image size

### Filename Alignment Requirement

To preserve `CLIP-I` and `BiomedCLIP-I`, inference must generate one image per reference image and save it with the same filename.

Required behavior:

1. list and sort files in `reference_image_dir`
2. for each file, generate one unconditional sample
3. save the sample under `output_dir/<same_filename>`
4. write metadata including:
   - reference image path
   - generated image path
   - sample index
   - seed used

This preserves the evaluation contract already enforced by the current metric code.

### Why Reference Images Are Still Needed In Inference

The reference directory is not used as a conditioning signal. It exists only to:

- define the required number of generated samples
- define deterministic filename alignment for pairwise image metrics

This keeps the generation model unconditional while still making evaluation reproducible.

## Evaluation Design

### Supported Metrics

The unconditional evaluation entrypoint should compute:

- `FID`
- `Inception Score`
- `CLIP-I`
- `Med-FID`
- `BiomedCLIP-I`

### Explicitly Removed Metrics

The unconditional evaluation path should not compute:

- `CLIP-T`
- `BiomedCLIP-T`

These metrics depend on prompt-image pairs and are not meaningful for unconditional generation.

### Eval Input Contract

The unconditional eval config should require:

- `real_image_dir`
- `generated_image_dir`
- `batch_size`
- `num_workers`
- `inception_weights_path`
- `clip_model_path`
- `biomedclip_model_path`
- cache directories for real-image feature caching when desired

It should additionally validate:

- both image directories exist
- image counts match
- sorted filenames match exactly

If filenames do not align, evaluation must fail early with a clear error.

## Checkpoint And Export Design

### Training Checkpoints

Training checkpoints should capture:

- unconditional UNet weights
- optimizer state
- LR scheduler state
- accelerator/distributed state
- global step metadata

### Final Export Bundle

The final exported inference bundle should contain:

- `unet` weights
- inference scheduler config
- metadata about the reused VAE source
- training summary JSON

The VAE itself can be referenced by path/config rather than duplicated into every run output if the codebase already assumes local pretrained assets.

## YAML Design

### Training YAML Shape

Each train YAML should include:

- model section
- data section
- train section
- validation section
- logging section
- distributed section

The key difference from existing train YAMLs is that:

- no prompt-conditioning config exists
- no LoRA config exists
- the model family name is replaced by an explicit unconditional model type

Dataset-specific YAMLs should vary only in:

- dataset manifest paths
- output directories
- experiment names
- optional dataset-specific batch sizes if needed

### Inference YAML Shape

Each infer YAML should include:

- model export path
- VAE path
- scheduler settings
- `reference_image_dir`
- `output_dir`
- `batch_size`
- `num_inference_steps`
- `seed`

### Eval YAML Shape

Each eval YAML should include:

- `real_image_dir`
- `generated_image_dir`
- `output_path`
- `batch_size`
- `num_workers`
- model paths for Inception, CLIP, and BiomedCLIP
- optional cache directories

## Dataset-Specific Config Deliverables

The implementation must provide YAMLs for:

- `lits`
- `m4raw_t1`
- `m4raw_t2`
- `m4raw_flair`

These configs should follow the existing repository naming style and should be ready for direct use after local path verification.

## Parallel Implementation Strategy

Yes, this work can be split across three worktrees cleanly. In fact, that is a good fit for this project because train, infer, and eval have narrow interfaces and limited overlap if boundaries are defined carefully.

Recommended split:

Each implementation worktree must also have its own dedicated review and test coverage. Do not treat review and testing as one shared global phase for the whole project. For every worktree below, assign:

- one implementation owner
- one reviewer agent
- one tester agent

### Worktree 1: Training

Ownership:

- `2d-gen/src/uncond_ldm/config.py`
- `2d-gen/src/uncond_ldm/dataset.py`
- `2d-gen/src/uncond_ldm/trainer.py`
- `2d-gen/src/train/run_train_uncond_ldm.py`
- train YAMLs

Responsibilities:

- config schema
- image-only dataset path
- unconditional latent diffusion training loop
- checkpointing integration on the training side
- training-specific code review for bugs, regressions, and interface drift
- training-specific validation and test execution

### Worktree 2: Inference

Ownership:

- `2d-gen/src/uncond_ldm/pipeline.py`
- `2d-gen/src/uncond_ldm/checkpointing.py`
- `2d-gen/src/infer/run_infer_uncond_ldm.py`
- infer YAMLs

Responsibilities:

- unconditional generation helper
- final export/load contract
- reference-filename-aligned sample generation
- inference-specific code review for sampling correctness and export compatibility
- inference-specific validation and test execution

### Worktree 3: Evaluation

Ownership:

- `2d-gen/src/eval/run_evaluate_uncond.py`
- any small shared evaluation helpers needed for unconditional mode
- eval YAMLs
- tests for eval contract and filename alignment

Responsibilities:

- image-only metric orchestration
- removal of prompt-dependent metrics
- directory and filename consistency validation
- evaluation-specific code review for metric correctness and failure modes
- evaluation-specific validation and test execution

### Shared Interface Rules

To keep the worktrees independent:

- training must define the final exported model layout early
- inference must consume only that exported layout and not training internals
- evaluation must consume only generated image directories and not training internals
- avoid shared mutable edits to the existing text-conditioned adapter stack

The main coordination point is the exported model bundle contract. That contract should be fixed before parallel implementation starts.

### Required Agent Composition

If the project is executed with three parallel worktrees, the minimum recommended staffing is:

- `train` worktree: 1 implementation agent, 1 reviewer agent, 1 tester agent
- `infer` worktree: 1 implementation agent, 1 reviewer agent, 1 tester agent
- `eval` worktree: 1 implementation agent, 1 reviewer agent, 1 tester agent

The main coordinating session remains responsible for:

- fixing the shared export contract before parallel work begins
- integrating cross-worktree decisions
- checking that reviewer and tester outputs are consistent with the final merged state

## Recommended Implementation Order

Even with parallel worktrees, the logical dependency order should be:

1. define config and export contract
2. implement training
3. implement inference against the fixed export contract
4. implement unconditional evaluation entrypoint
5. add YAMLs and tests

Parallel work is still viable if step 1 is resolved first and treated as stable.

## Testing Plan

At minimum, add tests for:

- unconditional config validation
- image-only dataset loading and output tensor shape
- VAE encode and UNet forward smoke path
- unconditional inference filename alignment
- unconditional eval rejecting mismatched filenames
- unconditional eval computing only the intended metrics

If runtime constraints are tight, smoke tests should use tiny local fixtures and mocked lightweight modules where possible.

## Risks

### Risk 1: Reused SD VAE May Not Be Domain-Optimal

This is the largest modeling risk. The design accepts it because the project explicitly prioritizes implementation speed and reuse of the current image path.

### Risk 2: Pairwise Image Metrics Depend On Filename Alignment

This is an operational risk rather than a modeling risk. It is solved by making reference-driven filename generation part of the unconditional inference contract.

### Risk 3: Duplicate Logic Relative To Existing Family Code

Some train/infer/eval structure will partially overlap with current family code. This is acceptable because the unconditional path is conceptually different and would become harder to maintain if forced into prompt-conditioned abstractions.

## Success Criteria

This design is successful when the repository can:

1. train an unconditional latent diffusion UNet on medical slice PNGs at `512x512`
2. generate unconditional samples with filenames aligned to a reference directory
3. evaluate those samples with `FID`, `IS`, `CLIP-I`, `Med-FID`, and `BiomedCLIP-I`
4. run dataset-specific experiments using ready-made YAMLs for `lits`, `m4raw_t1`, `m4raw_t2`, and `m4raw_flair`

## Self-review

This design is intentionally narrow. It does not mix prompt-conditioned and unconditional abstractions, it keeps the medical image preprocessing consistent with the current repository behavior, and it defines the file-alignment rule needed to preserve pairwise metrics. Scope is limited to one unconditional latent diffusion baseline with a fixed pretrained VAE and a from-scratch unconditional UNet.
