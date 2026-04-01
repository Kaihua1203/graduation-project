# Diffusers Pipeline Training Mapping

## Scope

This is the top-level map for four diffusers-based training families:

- Stable Diffusion
- SDXL
- Flux
- QwenImage

Use it to answer one question quickly:

- what is shared across the families
- what must stay model-specific
- which deeper doc or reference file to read next

## Disclosure Path

1. Read this doc for the shared mental model.
2. Read `training_architecture.md` when you are designing or changing trainer code.
3. Read a model-specific doc only when you need the family-specific contract.
4. Open `reference/` only when you need to match upstream implementation details.

## Shared Skeleton

Ignoring inference-only wrappers, all four families reduce to the same high-level train step:

1. prepare text conditioning
2. prepare image latents
3. sample timestep or sigma
4. inject noise or construct flow-matching state
5. run denoiser or transformer
6. compute target and loss
7. backprop through selected trainable modules

The main divergence points are steps 1, 2, 3, and 5.

## Shared Abstraction

A reusable training framework usually needs these adapter hooks:

- `encode_text(batch) -> conditioning`
- `encode_image(batch) -> clean_latents`
- `prepare_noisy_input(clean_latents, time_state) -> model_input`
- `forward_denoiser(model_input, time_state, conditioning) -> prediction`
- `compute_target(clean_latents, noise, time_state) -> target`
- `compute_loss(prediction, target) -> loss`

The trainer should own the loop. The adapter should own the contract.

## Family Split

| Family | Denoiser | Text Path | Latent Format During Denoising | Time Interface | Main Extra Conditioning |
| --- | --- | --- | --- | --- | --- |
| Stable Diffusion | UNet | single CLIP | `[B, C, H, W]` | timestep `t` | none |
| SDXL | UNet | dual CLIP | `[B, C, H, W]` | timestep `t` | pooled text embeds + `time_ids` |
| Flux | Transformer | T5 seq + CLIP pooled | packed latent tokens | sigma/timestep with flow-matching shift | `pooled_projections`, `txt_ids`, `img_ids` |
| QwenImage | Transformer | Qwen hidden states + mask | packed latent tokens | sigma/timestep with flow-matching shift | prompt mask, `img_shapes` |

## Why The Families Differ

### Stable Diffusion

- simplest baseline
- standard latent diffusion
- best starting point for trainer scaffolding

### SDXL

- same denoiser family as Stable Diffusion
- adds pooled embeddings and `time_ids`
- first place where generic SD-style code tends to break

### Flux

- switches to a transformer denoiser
- uses packed latent tokens
- mixes sequence text features and pooled features from different encoders

### QwenImage

- also uses a transformer denoiser
- requires prompt template and mask handling
- keeps its own text conditioning semantics

## What Can Be Shared

These are good candidates for shared trainer code:

- dataset and batch loading
- VAE encode/decode wrappers
- optimizer and scaler setup
- checkpointing
- logging
- LoRA or full-finetune parameter selection
- classifier-free dropout orchestration at batch construction time

## What Should Stay Model-Specific

These usually belong behind per-family adapters:

- prompt encoding
- latent packing vs non-packing
- timestep or sigma sampling logic
- denoiser forward kwargs
- training target construction

If you collapse these into one `if model_name == ...` block, the trainer will get brittle quickly.

## Reference Directory

Use `reference/` when you need concrete upstream code patterns:

- `reference/train_text_to_image_lora_sd.py`
- `reference/train_text_to_image_lora_sdxl.py`
- `reference/train_dreambooth_lora_sd.py`
- `reference/train_image_to_image_lora_kontext.py`

Prefer the family docs first, then the reference scripts only for unresolved details.

## Next Doc

If the question shifts from "how do these families differ" to "how should I implement the trainer", read `training_architecture.md`.
