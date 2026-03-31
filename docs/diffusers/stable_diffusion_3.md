# Stable Diffusion 3 Pipeline

Source: `/home/jupyter-wenkaihua/data3_link/kaihua.wen/code/diffusers/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py`

## Scope

This file is the SD3 text-to-image pipeline built around MMDiT-style transformer denoising:

- text side: two CLIP encoders + one T5 encoder
- denoiser: `SD3Transformer2DModel`
- latent codec: `AutoencoderKL`
- sampler: `FlowMatchEulerDiscreteScheduler`

If you are writing training code, SD3 should be treated as transformer-based latent diffusion (closer to Flux than SD1.x UNet), but with SD3-specific prompt fusion and guidance behavior.

## Registered Modules

`__init__` registers:

- `vae`
- `text_encoder`, `tokenizer`
- `text_encoder_2`, `tokenizer_2`
- `text_encoder_3`, `tokenizer_3`
- `transformer`
- `scheduler`
- `image_encoder` (optional, for IP-Adapter)
- `feature_extractor` (optional)

Important runtime state:

- `vae_scale_factor`
- `image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)`
- `tokenizer_max_length` (CLIP side, usually 77)
- `default_sample_size`
- `patch_size` (from transformer config)
- `_callback_tensor_inputs = ["latents", "prompt_embeds", "pooled_prompt_embeds"]`

## Core Helper Functions

Main helper set:

- `_get_clip_prompt_embeds(...)`
  - runs CLIP encoder #1 or #2
  - returns token-level hidden states + pooled projection
- `_get_t5_prompt_embeds(...)`
  - runs T5 encoder
  - returns token-level embeddings
- `encode_prompt(...)`
  - fuses CLIP(1)+CLIP(2)+T5 into final conditioning tensors
  - handles negative branch for CFG
  - supports LoRA scale on text encoders
- `prepare_latents(...)`
  - creates Gaussian latent tensor `[B, C, H/vae_sf, W/vae_sf]`
- `prepare_ip_adapter_image_embeds(...)`
  - prepares optional vision conditioning for IP-Adapter
- `retrieve_timesteps(...)`
  - scheduler-agnostic helper for `timesteps` / `sigmas`
- `calculate_shift(...)`
  - computes `mu` for dynamic timestep shifting

## Prompt Encoding Logic

`encode_prompt(...)` is the key SD3 text-conditioning path.

Positive branch:

1. Encode `prompt` with CLIP #1 (`_get_clip_prompt_embeds(..., clip_model_index=0)`).
2. Encode `prompt_2` with CLIP #2.
3. Concatenate two CLIP token embeddings on hidden dimension (`dim=-1`).
4. Encode `prompt_3` with T5.
5. Pad CLIP concat embedding to T5 hidden width.
6. Concatenate CLIP part and T5 part on sequence dimension (`dim=-2`).
7. Build pooled embedding by concatenating pooled CLIP #1 and CLIP #2 vectors.

Negative branch (when CFG enabled):

- Repeat the same procedure for `negative_prompt`, `negative_prompt_2`, `negative_prompt_3`.
- Return:
  - `prompt_embeds`, `negative_prompt_embeds` (token-level)
  - `pooled_prompt_embeds`, `negative_pooled_prompt_embeds` (pooled-level)

Training takeaway:

- SD3 transformer consumes both token-level and pooled conditioning.
- Unlike SD1.x, pooled text representation is a first-class required input path.

## `__call__` Main Inference Flow

High-level order:

1. `check_inputs(...)`
   - validates prompt/embed exclusivity
   - validates shape consistency
   - enforces `height` and `width` divisibility by `vae_scale_factor * patch_size`
2. set runtime attributes (`guidance_scale`, `clip_skip`, `joint_attention_kwargs`, etc.)
3. `encode_prompt(...)`
4. if CFG, concatenate negative+positive text conditions in batch dimension
5. `prepare_latents(...)`
6. prepare timesteps:
   - optional dynamic shift `mu` via `calculate_shift(...)`
   - `retrieve_timesteps(...)` with optional custom `sigmas`
7. optional IP-Adapter embedding preparation
8. denoising loop
9. decode latents with VAE and postprocess image

## Denoising Loop

Per timestep:

1. If CFG, duplicate `latents` in batch (`torch.cat([latents] * 2)`).
2. Run transformer:

```python
noise_pred = self.transformer(
    hidden_states=latent_model_input,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    pooled_projections=pooled_prompt_embeds,
    joint_attention_kwargs=self.joint_attention_kwargs,
    return_dict=False,
)[0]
```

3. If CFG:
   - split unconditional/conditional predictions
   - combine with standard CFG formula
4. Optional skip-layer guidance (SD3.5-style):
   - in a configured step window, run an extra transformer pass with `skip_layers=...`
   - adjust guidance by `(noise_pred_text - noise_pred_skip_layers) * skip_layer_guidance_scale`
5. Update latents via scheduler:
   - `latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]`
6. Optional `callback_on_step_end` tensor rewrites.

## Scheduler And Time Logic

SD3 uses `FlowMatchEulerDiscreteScheduler`.

Important details:

- supports custom `sigmas` input
- supports optional dynamic shifting with `mu`
- if `use_dynamic_shifting` is enabled in scheduler config:
  - estimate `image_seq_len` from latent spatial size and transformer patch size
  - compute `mu` via `calculate_shift(...)`
  - pass `mu` to scheduler when setting timesteps

Training implication:

- do not directly copy DDPM-style scheduler assumptions from classic SD training loops
- align target/objective with flow-matching scheduler semantics

## Accelerate Multi-GPU Wrap/Unwrap

When adapting SD3 training to multi-GPU, the common pattern is:

1. Build modules/optimizer/dataloader/scheduler on a single process.
2. Call `accelerator.prepare(...)` to wrap distributed objects.
3. Use wrapped models for forward/backward/update.
4. Before saving/exporting/inference pipeline assembly, call `accelerator.unwrap_model(...)` to recover the base module.

Typical skeleton:

```python
from accelerate import Accelerator

accelerator = Accelerator()

transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    transformer, optimizer, train_dataloader, lr_scheduler
)

for batch in train_dataloader:
    with accelerator.accumulate(transformer):
        loss = ...
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

unwrapped_transformer = accelerator.unwrap_model(transformer)
```

Why this matters:

- `prepare(...)` may wrap the model with DDP/FSDP/mixed-precision/distributed dataloader behaviors.
- `unwrap_model(...)` is needed when:
  - saving LoRA/base weights
  - accessing raw `.config` / state dict consistently
  - building a `DiffusionPipeline` from trained modules without distributed wrappers

For SD3 specifically:

- the model you usually wrap/unwrap is `SD3Transformer2DModel` (and any trainable text encoder if enabled).
- VAE and frozen encoders can stay outside optimizer, but if they are passed into `prepare(...)`, they should also be unwrapped before serialization.

## Minimal Training Mental Model

For SD3-style training, the forward path is:

1. Encode image to latent via VAE.
2. Sample time/noise level according to flow-matching scheduler setup.
3. Build SD3 text conditions:
   - token embeddings from CLIP+T5 fusion
   - pooled embeddings from dual CLIP
4. Run `SD3Transformer2DModel` with both conditioning paths.
5. Compute training target/objective consistent with scheduler formulation.

## What Matters Most For Reuse

- The training-critical interface is the transformer forward signature:
  - `hidden_states`
  - `timestep`
  - `encoder_hidden_states`
  - `pooled_projections`
- CFG in inference is batch-concatenation based (negative+positive), then split/merge.
- Skip-layer guidance is an inference-time enhancement, not a required baseline training component.
- IP-Adapter handling is auxiliary; core SD3 training can ignore it unless image conditioning is required.
