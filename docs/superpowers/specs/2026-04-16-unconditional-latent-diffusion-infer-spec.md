# Unconditional Latent Diffusion Inference Spec

## Topic

This spec defines the inference worktree for standalone unconditional latent diffusion in `2d-gen`.

This worktree owns:

- `2d-gen/src/uncond_ldm/pipeline.py`
- `2d-gen/src/uncond_ldm/checkpointing.py`
- `2d-gen/src/infer/run_infer_uncond_ldm.py`
- infer YAMLs

## Goal

Implement unconditional sampling that:

- loads the exported model bundle produced by training
- samples in latent space without prompts
- decodes images through the fixed VAE
- saves outputs with filenames aligned to a reference image directory for pairwise evaluation

## Fixed Context

- inference is unconditional and must not accept prompts
- the reused VAE is fixed and referenced from the export bundle or config
- generated image filenames must match sorted filenames from the reference directory one-to-one
- the reference directory is used for counting and filename alignment only, never as conditioning input
- inference must stay outside the existing prompt-conditioned `family` abstraction

## Module Layout

The inference worktree is responsible for:

```text
2d-gen/src/uncond_ldm/
  pipeline.py
  checkpointing.py
```

and the entrypoint:

```text
2d-gen/src/infer/run_infer_uncond_ldm.py
```

## Responsibilities

`uncond_ldm/pipeline.py`

- expose a small unconditional generation helper
- sample Gaussian latent noise
- run scheduler denoising with `UNet2DModel`
- decode latents with the fixed VAE
- postprocess decoded outputs to images

`uncond_ldm/checkpointing.py`

- load the final exported model bundle
- keep training checkpoint layout separate from inference loading contract
- validate required export artifacts exist

`run_infer_uncond_ldm.py`

- load inference YAML
- resolve the exported bundle and VAE source
- iterate reference images in sorted order
- generate one unconditional sample per reference image
- save outputs and metadata

## Runtime Objects

The inference path should need only:

- `AutoencoderKL`
- `UNet2DModel`
- one inference scheduler such as `DDIMScheduler` or `DDPMScheduler`

No tokenizer, text encoder, pooled conditioning, prompt embeddings, or prompt files should exist in this path.

## Input Contract

The inference entrypoint should accept:

- trained model path or export bundle path
- VAE path if not fully resolved by the export bundle
- reference image directory
- output directory
- number of inference steps
- batch size
- seed
- image size

It must not accept prompts.

## Filename Alignment Contract

To preserve `CLIP-I` and `BiomedCLIP-I`, inference must generate one image per reference image and save it with the same filename.

Required behavior:

1. list and sort files in `reference_image_dir`
2. for each reference filename, generate one unconditional sample
3. save the sample under `output_dir/<same_filename>`
4. write metadata including:
   - reference image path
   - generated image path
   - sample index
   - seed used

This alignment contract is mandatory because evaluation depends on exact filename pairing.

## Why Reference Images Still Exist In Inference

The reference directory is not a conditioning signal. It exists only to:

- define the number of required generated samples
- define deterministic filename alignment for pairwise image metrics

This keeps the generation model unconditional while still making evaluation reproducible.

## Export Bundle Contract

Inference must consume only the final exported bundle produced by training. The stable bundle layout is:

- `unet` weights
- inference scheduler config
- metadata about the reused VAE source
- training summary JSON

Inference must not depend on optimizer state, accelerator state, or other trainer-only checkpoint internals.

## YAML Deliverables

Add dataset-specific inference YAMLs for:

- `lits`
- `m4raw_t1`
- `m4raw_t2`
- `m4raw_flair`

Each infer YAML should include:

- model export path
- VAE path
- scheduler settings
- `reference_image_dir`
- `output_dir`
- `batch_size`
- `num_inference_steps`
- `seed`

## Interfaces To Other Worktrees

- inference depends on the training export bundle contract, not trainer internals
- inference produces only generated image directories plus metadata for evaluation
- inference should not modify prompt-conditioned generation code

## Testing Requirements

At minimum, this worktree should add or cover:

- export bundle loading
- unconditional sampling smoke coverage
- filename alignment behavior
- metadata writing for generated outputs

If runtime constraints are tight, tests should use tiny fixtures and mocked lightweight modules.

## Success Criteria

This worktree is complete when the repo can:

1. load the exported unconditional model bundle
2. generate unconditional samples without prompt inputs
3. save outputs with filenames exactly aligned to a reference directory
4. run ready-to-use infer YAMLs for `lits`, `m4raw_t1`, `m4raw_t2`, and `m4raw_flair`
