# Flux Pipeline

Source: `diffusers/src/diffusers/pipelines/flux/pipeline_flux.py`

## Scope

Flux is not a UNet latent diffusion pipeline. It is a transformer-based latent generator built around:

- `FluxTransformer2DModel`
- flow-matching style scheduler (`FlowMatchEulerDiscreteScheduler`)
- packed latent patches
- dual text conditioning from CLIP + T5

If you are writing training code, treat Flux as a different family from SD/SDXL. The main differences are the denoiser architecture, latent layout, scheduler interface, and prompt encoding scheme.

## Registered Modules

`__init__` registers:

- `scheduler`
- `vae`
- `text_encoder` (CLIP)
- `tokenizer`
- `text_encoder_2` (T5)
- `tokenizer_2`
- `transformer`
- `image_encoder` (optional)
- `feature_extractor` (optional)

Important state:

- `vae_scale_factor`
- `image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)`
- `tokenizer_max_length`
- `default_sample_size = 128`

The `* 2` is important because Flux packs latent spatial grids into 2x2 patches.

## Prompt Encoding Logic

Flux uses two different text paths:

- CLIP path via `_get_clip_prompt_embeds(...)`
  - returns pooled prompt embeddings only
- T5 path via `_get_t5_prompt_embeds(...)`
  - returns token-level prompt embeddings

`encode_prompt(...)` combines them into:

- `prompt_embeds`
  - token-level T5 embeddings
- `pooled_prompt_embeds`
  - pooled CLIP embeddings
- `text_ids`
  - currently initialized as zeros with shape `[seq_len, 3]`

Training takeaway:

- Flux does not use CLIP token hidden states the way SD/SDXL do.
- The transformer consumes T5 sequence embeddings and CLIP pooled projections in different argument slots.

## Latent Packing

This file has explicit latent pack/unpack helpers:

- `_pack_latents(...)`
- `_unpack_latents(...)`
- `_prepare_latent_image_ids(...)`

The packed latent representation is a sequence-like tensor, not a standard `[B, C, H, W]` map during denoising.

Practical consequence:

- VAE output must be reshaped and packed before entering the transformer.
- The transformer also receives `img_ids`, which describe image patch positions.

## `__call__` Main Inference Flow

High-level order:

1. `check_inputs(...)`
2. decide whether true CFG is active via `true_cfg_scale` + negative prompt presence
3. `encode_prompt(...)` for positive branch
4. optional `encode_prompt(...)` for negative branch
5. `prepare_latents(...)`
   - returns `latents` and `latent_image_ids`
6. build sigma schedule
7. compute dynamic shift `mu = calculate_shift(...)`
8. `retrieve_timesteps(...)`
9. optional guidance embedding tensor for guidance-distilled variants
10. optional IP-Adapter image embeddings
11. run transformer denoising loop
12. unpack latents
13. rescale with VAE config and decode

## Denoising Loop

The central forward is:

```python
noise_pred = self.transformer(
    hidden_states=latents,
    timestep=timestep / 1000,
    guidance=guidance,
    pooled_projections=pooled_prompt_embeds,
    encoder_hidden_states=prompt_embeds,
    txt_ids=text_ids,
    img_ids=latent_image_ids,
    joint_attention_kwargs=self.joint_attention_kwargs,
)[0]
```

If true CFG is enabled, the file runs a second transformer pass for the negative branch and combines outputs:

```python
noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
```

Key differences from SD/SDXL:

- no latent batch concatenation for CFG
- separate conditional and unconditional transformer passes
- denoiser input is packed latent tokens
- timestep is normalized by `/ 1000`

## Scheduler And Time Logic

Flux uses `FlowMatchEulerDiscreteScheduler`, not the common DDPM/DDIM-style schedulers.

Important details:

- default `sigmas` are linearly spaced from `1.0` to `1 / num_inference_steps`
- `calculate_shift(...)` derives `mu` from image sequence length
- `retrieve_timesteps(...)` is called with `sigmas` and `mu`

Training implication:

- When you write training code, do not blindly copy SD noise scheduling assumptions.
- Check the transformer objective and scheduler expectation together.

## Minimal Training Mental Model

To train Flux-like models, your forward path is closer to:

1. encode image to latent
2. pack latent grid into patch sequence
3. sample time / sigma according to flow-matching setup
4. build T5 token embeddings and CLIP pooled embeddings
5. prepare `text_ids` and `img_ids`
6. run `FluxTransformer2DModel`
7. compute the target compatible with the flow-matching formulation

## What Matters Most For Reuse

- The core reusable object is the transformer forward signature, not the pipeline wrapper.
- Latent packing is architectural, not a cosmetic inference trick.
- Flux conditioning is split across:
  - `encoder_hidden_states`
  - `pooled_projections`
  - `txt_ids`
  - `img_ids`
- If you reuse only the VAE and prompt encoders but keep SD-style `[B, C, H, W]` UNet training, you are no longer matching Flux.
