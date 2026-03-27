# Diffusers Pipeline Training Mapping

## Scope

This note maps four inference pipelines to one training-oriented mental model:

- Stable Diffusion
- SDXL
- Flux
- QwenImage

Goal:

- identify the shared training skeleton
- show where each model family diverges
- make it easier to design reusable training code

## One Unified Training Skeleton

Ignoring inference-only wrappers, all four pipelines can be reduced to:

1. prepare text conditioning
2. prepare image latents
3. sample timestep / sigma
4. inject noise or construct flow-matching state
5. run denoiser / transformer
6. compute training target and loss
7. backprop through selected trainable modules

The main differences are in steps 1, 2, 3, and 5.

## Common Abstraction

If you want one reusable training framework, the cleanest abstraction is:

- `encode_text(batch) -> conditioning`
- `encode_image(batch) -> clean_latents`
- `prepare_noisy_input(clean_latents, time_state) -> model_input`
- `forward_denoiser(model_input, time_state, conditioning) -> prediction`
- `compute_target(clean_latents, noise, time_state) -> target`
- `compute_loss(prediction, target) -> loss`

Then specialize by model family.

## Side-By-Side Summary

| Model | Denoiser | Text Path | Latent Format During Denoising | Time Interface | Main Extra Conditioning |
| --- | --- | --- | --- | --- | --- |
| Stable Diffusion | UNet | single CLIP | `[B, C, H, W]` | timestep `t` | none |
| SDXL | UNet | dual CLIP | `[B, C, H, W]` | timestep `t` | pooled text embeds + `time_ids` |
| Flux | Transformer | T5 seq + CLIP pooled | packed latent tokens | sigma/timestep with flow-matching shift | `pooled_projections`, `txt_ids`, `img_ids` |
| QwenImage | Transformer | Qwen hidden states + mask | packed latent tokens | sigma/timestep with flow-matching shift | prompt mask, `img_shapes` |

## Step 1: Text Conditioning

### Stable Diffusion

Need:

- `prompt_embeds`

Forward interface:

```python
encoder_hidden_states=prompt_embeds
```

Training note:

- simplest text path
- best baseline for building the first version of trainer code

### SDXL

Need:

- `prompt_embeds`
- `pooled_prompt_embeds`
- `add_time_ids`

Forward interface:

```python
encoder_hidden_states=prompt_embeds
added_cond_kwargs={
    "text_embeds": pooled_prompt_embeds,
    "time_ids": add_time_ids,
}
```

Training note:

- this is the first place where a generic SD trainer usually breaks
- pooled embeddings and `time_ids` are not optional if you want real SDXL behavior

### Flux

Need:

- `prompt_embeds` from T5
- `pooled_prompt_embeds` from CLIP
- `text_ids`

Forward interface:

```python
encoder_hidden_states=prompt_embeds
pooled_projections=pooled_prompt_embeds
txt_ids=text_ids
```

Training note:

- sequence text features and pooled text features come from different encoders

### QwenImage

Need:

- `prompt_embeds`
- `prompt_embeds_mask`

Forward interface:

```python
encoder_hidden_states=prompt_embeds
encoder_hidden_states_mask=prompt_embeds_mask
```

Training note:

- prompt template logic must be preserved
- mask handling is part of the conditioning interface

## Step 2: Latent Preparation

### Stable Diffusion / SDXL

Latents are standard VAE image latents:

- shape style: `[B, C, H/vae_scale, W/vae_scale]`

Training implication:

- image-to-latent and noise injection are straightforward

### Flux / QwenImage

Latents are packed into token-like representations before denoising.

Training implication:

- the denoiser does not operate on a plain 2D latent feature map
- you need to preserve pack/unpack logic in the train path

This is the strongest architectural split in the four models.

## Step 3: Time / Noise Construction

### Stable Diffusion / SDXL

Typical pattern:

- sample discrete timestep `t`
- add Gaussian noise according to scheduler convention
- predict noise, velocity, or related target

### Flux / QwenImage

Typical pattern:

- build sigma-based or flow-matching time state
- use shifted timesteps derived from sequence length
- run a flow-matching-compatible transformer forward

Training implication:

- do not reuse SD scheduler assumptions for Flux/QwenImage
- the loss target may be different even if the outer trainer loop looks similar

## Step 4: Denoiser Forward Mapping

### Stable Diffusion

```python
pred = unet(
    noisy_latents,
    t,
    encoder_hidden_states=prompt_embeds,
)[0]
```

### SDXL

```python
pred = unet(
    noisy_latents,
    t,
    encoder_hidden_states=prompt_embeds,
    added_cond_kwargs={
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids,
    },
)[0]
```

### Flux

```python
pred = transformer(
    hidden_states=packed_latents,
    timestep=timestep / 1000,
    guidance=guidance,
    pooled_projections=pooled_prompt_embeds,
    encoder_hidden_states=prompt_embeds,
    txt_ids=text_ids,
    img_ids=img_ids,
)[0]
```

### QwenImage

```python
pred = transformer(
    hidden_states=packed_latents,
    timestep=timestep / 1000,
    guidance=guidance,
    encoder_hidden_states=prompt_embeds,
    encoder_hidden_states_mask=prompt_embeds_mask,
    img_shapes=img_shapes,
)[0]
```

## What Can Be Unified In Code

These parts are good candidates for a shared trainer framework:

- dataset and batch loading
- VAE encode/decode wrappers
- optimizer and scaler setup
- checkpointing
- logging
- LoRA/full-finetune parameter selection
- classifier-free dropout orchestration at batch construction time

## What Should Usually Be Model-Specific

These parts should usually live behind per-model adapters:

- prompt encoding
- latent packing vs non-packing
- timestep / sigma sampling logic
- denoiser forward kwargs
- training target construction

If you try to force these into one flat function full of `if model_name == ...`, the trainer will get messy quickly.

## Recommended Code Structure

A practical structure is:

```text
trainers/
  base_trainer.py
adapters/
  sd_adapter.py
  sdxl_adapter.py
  flux_adapter.py
  qwenimage_adapter.py
```

Where each adapter implements the same small interface:

- `encode_text(batch)`
- `encode_latents(batch)`
- `sample_time(batch, latents)`
- `prepare_model_input(latents, time_state, noise)`
- `forward_model(model_input, time_state, conditioning)`
- `build_target(latents, noise, time_state)`

This is usually the best compromise between reuse and correctness.

## Suggested Development Order

If the end goal is to support all four models, the lowest-risk order is:

1. Stable Diffusion
2. SDXL
3. Flux
4. QwenImage

Reason:

- SD establishes the basic latent diffusion trainer
- SDXL extends conditioning without changing denoiser family
- Flux changes denoiser family and latent layout
- QwenImage adds another transformer family plus custom prompt-template logic

## Minimal Design Rule

The safest design rule is:

- share the training loop
- do not over-share the model interface

In practice, these four pipelines are similar enough to use one trainer shell, but different enough that each needs its own conditioning and denoiser adapter.
