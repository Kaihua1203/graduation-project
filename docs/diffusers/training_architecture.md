# Diffusers Training Architecture

## Scope

This is the implementation-level follow-up to `training_mapping.md`.

Use it when you are designing or changing trainer code for:

- Stable Diffusion
- SDXL
- Flux
- QwenImage

## Disclosure Path

1. Read `training_mapping.md` first to understand the family split.
2. Read this doc when you need the trainer and adapter contract.
3. Read the model-specific docs when you need family-specific behavior.
4. Read `reference/` only to verify an upstream pattern or script detail.

## Design Goal

Keep one shared training loop, but isolate model-specific behavior in small adapters.

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

### `BaseModelAdapter`

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

### `TimeState`

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

### `Conditioning`

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

## Family-Specific Adapters

### Stable Diffusion Adapter

Responsibilities:

- call CLIP text encoder
- VAE encode image latents
- sample discrete timestep
- add Gaussian noise
- run UNet with `encoder_hidden_states`
- build epsilon or velocity target

### SDXL Adapter

Responsibilities:

- run both text encoders
- build pooled text embeddings
- build `add_time_ids`
- VAE encode image latents
- sample timestep and noisy latent
- run UNet with `added_cond_kwargs`

### Flux Adapter

Responsibilities:

- get T5 token embeddings
- get CLIP pooled embeddings
- prepare `text_ids`
- VAE encode then pack latents
- create `img_ids`
- sample flow-matching time state
- run transformer with packed latent tokens

### QwenImage Adapter

Responsibilities:

- apply Qwen prompt template
- extract hidden states and prompt mask
- VAE encode then pack latents
- build `img_shapes`
- sample flow-matching time state
- run transformer with prompt mask

## CFG Training Strategy

Do not copy inference CFG literally into the trainer.

Recommended training-time approach:

- do conditional dropout on text conditioning
- represent dropped conditioning as null prompt or zeroed auxiliary condition depending on model family
- keep this policy inside the adapter

Why:

- SD/SDXL null conditioning and Flux/QwenImage negative branch handling are not identical
- the trainer should not know how empty conditioning is represented for each model

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
