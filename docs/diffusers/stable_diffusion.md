# Stable Diffusion Pipeline

Source: `diffusers/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`

## Scope

This file is the classic latent diffusion text-to-image pipeline:

- text side: single CLIP text encoder + tokenizer
- denoiser: `UNet2DConditionModel`
- latent codec: `AutoencoderKL`
- sampler: `KarrasDiffusionSchedulers`

If you want to write training code, this is the cleanest baseline to start from because the conditioning path is the simplest and the denoiser is still pure UNet-style latent diffusion.

## Registered Modules

`__init__` registers:

- `vae`
- `text_encoder`
- `tokenizer`
- `unet`
- `scheduler`
- `safety_checker`
- `feature_extractor`
- `image_encoder` (optional, for IP-Adapter)

Important runtime state:

- `vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)`, usually `8`
- `image_processor = VaeImageProcessor(...)`
- `_callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]`

## Core Helper Functions

The pipeline is organized around a small set of helpers:

- `encode_prompt(...)`
  - tokenizes prompt text
  - runs CLIP text encoder
  - handles textual inversion, LoRA scaling, `clip_skip`
  - builds unconditional embeddings for classifier-free guidance
- `prepare_latents(...)`
  - creates Gaussian latents of shape `[B, C, H/8, W/8]`
- `prepare_extra_step_kwargs(...)`
  - passes scheduler-specific args such as `eta`
- `prepare_ip_adapter_image_embeds(...)`
  - computes optional image conditioning for IP-Adapter
- `get_guidance_scale_embedding(...)`
  - only used if the UNet has `time_cond_proj_dim`

## Prompt Encoding Logic

`encode_prompt(...)` is the main text-conditioning entry.

Flow:

1. Normalize `prompt` to batch form.
2. Run tokenizer with truncation to `tokenizer.model_max_length`.
3. Run `text_encoder(...)`.
4. Use:
   - final hidden states by default
   - an earlier hidden state plus final LayerNorm when `clip_skip` is set
5. Repeat embeddings for `num_images_per_prompt`.
6. If CFG is enabled, build `negative_prompt_embeds` with the same sequence length.

Training takeaway:

- The train-time conditioning tensor that matters most is `prompt_embeds`, shape `[B, seq_len, hidden_dim]`.
- For vanilla SD training, the text path only contributes through `encoder_hidden_states` into the UNet cross-attention blocks.
- There is no pooled text embedding path and no extra size/time metadata path.

## `__call__` Main Inference Flow

The high-level inference order is:

1. `check_inputs(...)`
2. determine batch size
3. `encode_prompt(...)`
4. concatenate negative and positive prompt embeddings if CFG is enabled
5. optionally prepare IP-Adapter image embeddings
6. `retrieve_timesteps(...)`
7. `prepare_latents(...)`
8. build scheduler extra kwargs
9. optional `timestep_cond` for guidance-scale embedding
10. run denoising loop
11. decode latents with VAE
12. run safety checker
13. postprocess to PIL / numpy

## Denoising Loop

Inside the loop, each step does:

1. duplicate latents if CFG is enabled
2. optionally call `scheduler.scale_model_input(...)`
3. run UNet:

```python
noise_pred = self.unet(
    latent_model_input,
    t,
    encoder_hidden_states=prompt_embeds,
    timestep_cond=timestep_cond,
    cross_attention_kwargs=self.cross_attention_kwargs,
    added_cond_kwargs=added_cond_kwargs,
)[0]
```

4. if CFG:
   - split `noise_pred` into unconditional and conditional halves
   - combine with `guidance_scale`
5. optionally apply `guidance_rescale`
6. update latents with `scheduler.step(...)`

This is the exact place you would mirror during training, except training usually samples one timestep and does one denoiser forward instead of a full sampling loop.

## Minimal Training Mental Model

For training code, reduce the pipeline to this skeleton:

1. Encode image to latent with VAE encoder.
2. Sample timestep `t`.
3. Add noise to latent using scheduler/noise schedule.
4. Encode prompt into `prompt_embeds`.
5. Run UNet with:
   - `sample=noisy_latents`
   - `timestep=t`
   - `encoder_hidden_states=prompt_embeds`
6. Compute target:
   - usually noise `epsilon`
   - or velocity / other target depending on scheduler + model setup
7. Backprop only through the trainable modules you choose.

## What Matters Most For Reuse

- The training-critical API is the UNet call signature, not the full pipeline wrapper.
- CFG in inference is implemented by batch concatenation; train-time dropout for CFG is a separate concern and is not implemented here.
- `safety_checker` and `image_processor.postprocess(...)` are inference-only concerns.
- If you are writing custom training code, this file mostly teaches you:
  - what conditioning tensors must be prepared
  - what latent shape convention the model expects
  - what kwargs the UNet forward requires
