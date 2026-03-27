# Diffusers Training Architecture

## Scope

This document turns the pipeline summary into an implementation-oriented trainer design for:

- Stable Diffusion
- SDXL
- Flux
- QwenImage

Goal:

- keep one shared training loop
- isolate model-specific logic in small adapters
- avoid a single monolithic trainer full of model-specific conditionals

## Design Principle

Use this split:

- one `BaseDiffusionTrainer`
- one adapter per model family
- one shared batch schema
- one shared optimization / checkpoint / logging path

The trainer should own:

- device placement
- mixed precision
- gradient accumulation
- optimizer and lr scheduler
- checkpointing
- logging
- train/eval loop structure

The adapter should own:

- text conditioning
- latent preparation
- timestep or sigma sampling
- denoiser forward kwargs
- target construction

## Recommended File Layout

```text
src/train/
  base_trainer.py
  train_loop.py
  losses.py
  batch_types.py
src/train/adapters/
  base_adapter.py
  stable_diffusion.py
  sdxl.py
  flux.py
  qwenimage.py
```

## Shared Batch Schema

At the dataloader boundary, keep one normalized batch dict.

Recommended keys:

```python
batch = {
    "pixel_values": Tensor,
    "prompt": list[str] | None,
    "prompt_2": list[str] | None,
    "negative_prompt": list[str] | None,
    "negative_prompt_2": list[str] | None,
    "original_size": Tensor | None,
    "target_size": Tensor | None,
    "crops_coords_top_left": Tensor | None,
    "metadata": dict | None,
}
```

Rules:

- keep fields present even when some models ignore them
- model adapters decide what they actually consume
- do not force Flux/QwenImage batches into SDXL metadata semantics inside the trainer

## Core Interfaces

## `BaseModelAdapter`

Each model adapter should implement a small fixed interface:

```python
class BaseModelAdapter:
    def encode_text(self, batch, device, dtype):
        ...

    def encode_latents(self, batch, device, dtype):
        ...

    def sample_time_state(self, latents, batch, device):
        ...

    def prepare_noisy_input(self, clean_latents, time_state, noise, batch, device, dtype):
        ...

    def forward_model(self, model_input, time_state, conditioning, batch):
        ...

    def build_target(self, clean_latents, noise, time_state, batch):
        ...

    def compute_loss(self, prediction, target, batch):
        ...
```

Optional extension hooks:

```python
def maybe_get_null_conditioning(self, batch, device, dtype):
    ...

def apply_cfg_dropout(self, conditioning, batch, rng):
    ...
```

## `TimeState`

Do not pass raw timestep tensors everywhere. Wrap them in a small structure.

```python
@dataclass
class TimeState:
    timesteps: Tensor | None
    sigmas: Tensor | None
    guidance: Tensor | None
    extra: dict
```

This keeps SD-like and flow-matching-like paths under one interface.

## `Conditioning`

Use one structured object instead of many loose tensors.

```python
@dataclass
class Conditioning:
    prompt_embeds: Tensor | None = None
    negative_prompt_embeds: Tensor | None = None
    pooled_prompt_embeds: Tensor | None = None
    negative_pooled_prompt_embeds: Tensor | None = None
    prompt_mask: Tensor | None = None
    negative_prompt_mask: Tensor | None = None
    add_time_ids: Tensor | None = None
    img_ids: Tensor | None = None
    text_ids: Tensor | None = None
    img_shapes: Any | None = None
    extra: dict | None = None
```

The trainer should never inspect model-specific fields directly.

## Shared Training Step

The generic training step should look like:

```python
def training_step(batch):
    batch = move_batch_to_device(batch)

    conditioning = adapter.encode_text(batch, device, dtype)
    clean_latents = adapter.encode_latents(batch, device, dtype)

    noise = torch.randn_like(clean_latents)
    time_state = adapter.sample_time_state(clean_latents, batch, device)
    model_input = adapter.prepare_noisy_input(
        clean_latents, time_state, noise, batch, device, dtype
    )

    prediction = adapter.forward_model(model_input, time_state, conditioning, batch)
    target = adapter.build_target(clean_latents, noise, time_state, batch)
    loss = adapter.compute_loss(prediction, target, batch)

    return loss
```

That should be the center of the trainer design.

## Stable Diffusion Adapter

## Responsibilities

- call CLIP text encoder
- VAE encode image latents
- sample discrete timestep
- add Gaussian noise
- run UNet with `encoder_hidden_states`
- build epsilon or velocity target

## Pseudocode

```python
class StableDiffusionAdapter(BaseModelAdapter):
    def encode_text(self, batch, device, dtype):
        prompt_embeds = pipe.encode_prompt(
            prompt=batch["prompt"],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]
        return Conditioning(prompt_embeds=prompt_embeds)

    def encode_latents(self, batch, device, dtype):
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        return latents.to(dtype)

    def sample_time_state(self, latents, batch, device):
        t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
        return TimeState(timesteps=t, sigmas=None, guidance=None, extra={})

    def prepare_noisy_input(self, clean_latents, time_state, noise, batch, device, dtype):
        return scheduler.add_noise(clean_latents, noise, time_state.timesteps)

    def forward_model(self, model_input, time_state, conditioning, batch):
        return unet(
            model_input,
            time_state.timesteps,
            encoder_hidden_states=conditioning.prompt_embeds,
        )[0]
```

## SDXL Adapter

## Responsibilities

- run both text encoders
- build pooled text embeddings
- build `add_time_ids`
- VAE encode image latents
- sample timestep and noisy latent
- run UNet with `added_cond_kwargs`

## Pseudocode

```python
class SDXLAdapter(BaseModelAdapter):
    def encode_text(self, batch, device, dtype):
        prompt_embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=batch["prompt"],
            prompt_2=batch["prompt_2"],
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        add_time_ids = build_add_time_ids_from_batch(batch, pipe, prompt_embeds.dtype, device)
        return Conditioning(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            add_time_ids=add_time_ids,
        )

    def forward_model(self, model_input, time_state, conditioning, batch):
        return unet(
            model_input,
            time_state.timesteps,
            encoder_hidden_states=conditioning.prompt_embeds,
            added_cond_kwargs={
                "text_embeds": conditioning.pooled_prompt_embeds,
                "time_ids": conditioning.add_time_ids,
            },
        )[0]
```

## Flux Adapter

## Responsibilities

- get T5 token embeddings
- get CLIP pooled embeddings
- prepare `text_ids`
- VAE encode then pack latents
- create `img_ids`
- sample flow-matching time state
- run transformer with packed latent tokens

## Pseudocode

```python
class FluxAdapter(BaseModelAdapter):
    def encode_text(self, batch, device, dtype):
        prompt_embeds, pooled, text_ids = pipe.encode_prompt(
            prompt=batch["prompt"],
            prompt_2=batch["prompt_2"],
            device=device,
            num_images_per_prompt=1,
        )
        return Conditioning(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled,
            text_ids=text_ids,
        )

    def encode_latents(self, batch, device, dtype):
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        packed = pack_flux_latents(latents)
        return packed

    def sample_time_state(self, latents, batch, device):
        ts = sample_flux_timesteps_or_sigmas(latents, scheduler, device)
        guidance = maybe_build_guidance_tensor(latents, transformer, device)
        img_ids = pipe._prepare_latent_image_ids(...)
        return TimeState(timesteps=ts, sigmas=None, guidance=guidance, extra={"img_ids": img_ids})

    def forward_model(self, model_input, time_state, conditioning, batch):
        return transformer(
            hidden_states=model_input,
            timestep=time_state.timesteps / 1000,
            guidance=time_state.guidance,
            pooled_projections=conditioning.pooled_prompt_embeds,
            encoder_hidden_states=conditioning.prompt_embeds,
            txt_ids=conditioning.text_ids,
            img_ids=time_state.extra["img_ids"],
        )[0]
```

## QwenImage Adapter

## Responsibilities

- apply Qwen prompt template
- extract hidden states and prompt mask
- VAE encode then pack latents
- build `img_shapes`
- sample flow-matching time state
- run transformer with prompt mask

## Pseudocode

```python
class QwenImageAdapter(BaseModelAdapter):
    def encode_text(self, batch, device, dtype):
        prompt_embeds, prompt_mask = pipe.encode_prompt(
            prompt=batch["prompt"],
            device=device,
            num_images_per_prompt=1,
        )
        return Conditioning(
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
        )

    def encode_latents(self, batch, device, dtype):
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents = normalize_qwen_latents(latents, vae)
        packed = pack_qwen_latents(latents)
        return packed

    def sample_time_state(self, latents, batch, device):
        ts = sample_qwen_timesteps_or_sigmas(latents, scheduler, device)
        guidance = maybe_build_guidance_tensor(latents, transformer, device)
        img_shapes = build_img_shapes_from_batch(batch, vae_scale_factor)
        return TimeState(timesteps=ts, sigmas=None, guidance=guidance, extra={"img_shapes": img_shapes})

    def forward_model(self, model_input, time_state, conditioning, batch):
        return transformer(
            hidden_states=model_input,
            timestep=time_state.timesteps / 1000,
            guidance=time_state.guidance,
            encoder_hidden_states=conditioning.prompt_embeds,
            encoder_hidden_states_mask=conditioning.prompt_mask,
            img_shapes=time_state.extra["img_shapes"],
        )[0]
```

## CFG Training Strategy

Do not copy inference CFG literally into the trainer.

Recommended training-time approach:

- do conditional dropout on text conditioning
- represent dropped conditioning as null prompt or zeroed auxiliary condition depending on model family
- keep this policy inside the adapter

Why:

- SD/SDXL null conditioning and Flux/QwenImage negative branch handling are not identical
- the trainer should not know how "empty conditioning" is represented for each model

## Loss Interface

Keep the trainer loss path generic:

```python
prediction = adapter.forward_model(...)
target = adapter.build_target(...)
loss = adapter.compute_loss(prediction, target, batch)
```

Likely variants:

- MSE on epsilon
- MSE on velocity
- flow-matching target
- masked loss if later needed

The trainer should not hardcode one target type.

## Parameter Selection

Add one more layer of configuration:

```python
train_mode:
  type: full | lora | transformer_only | unet_only | text_only
```

The adapter can expose named parameter groups:

```python
adapter.get_trainable_parameters(train_mode)
```

This matters because:

- SD/SDXL usually fine-tune UNet or LoRA layers
- Flux/QwenImage may focus on transformer blocks or LoRA on attention layers

## Validation Path

Do not validate by reusing the full pipeline object blindly.

Recommended:

- train with the adapter/model modules directly
- validate either:
  - with a lightweight generation helper that wraps current weights
  - or with the original pipeline rebuilt from updated modules

This avoids coupling the training step to inference-only pipeline conveniences.

## What To Implement First

Lowest-risk implementation order:

1. `BaseModelAdapter`
2. `StableDiffusionAdapter`
3. shared trainer loop
4. `SDXLAdapter`
5. `FluxAdapter`
6. `QwenImageAdapter`

Reason:

- SD proves the base trainer shape
- SDXL proves richer conditioning without changing denoiser family
- Flux proves packed latents + transformer path
- QwenImage proves custom text encoder template + mask path

## Practical Rule

If a piece of logic is visible in a pipeline `__call__`, ask:

- is this inference-only orchestration?
- or is this actually part of the model contract?

Only the second category belongs in training adapters.

Examples of model contract logic:

- prompt encoding shape and semantics
- latent packing
- time-state preparation
- denoiser forward kwargs
- latent normalization rules

Examples of inference-only logic:

- progress bars
- postprocess to PIL
- watermarking
- safety checker
- callback hooks

## Final Recommendation

Treat the four pipelines as two high-level families:

- UNet latent diffusion: Stable Diffusion, SDXL
- Transformer packed-latent diffusion: Flux, QwenImage

Then keep per-model adapters inside each family.

That gives you:

- enough sharing to avoid duplicated trainer code
- enough separation to preserve each model's actual forward contract
