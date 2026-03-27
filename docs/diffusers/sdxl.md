# SDXL Pipeline

Source: `diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py`

## Scope

SDXL keeps the same latent diffusion backbone idea as Stable Diffusion, but the conditioning path is richer:

- two text encoders
- concatenated text hidden states
- pooled text embeddings
- extra size/crop metadata packed into `time_ids`

If you are writing SDXL training code, the main jump from SD is not the denoiser loop. The real jump is the conditioning interface into the UNet.

## Registered Modules

`__init__` registers:

- `vae`
- `text_encoder`
- `text_encoder_2`
- `tokenizer`
- `tokenizer_2`
- `unet`
- `scheduler`
- `image_encoder` (optional)
- `feature_extractor` (optional)

Important state:

- `vae_scale_factor`
- `image_processor`
- `default_sample_size`
- optional watermark support
- `_callback_tensor_inputs = ["latents", "prompt_embeds", "add_text_embeds", "add_time_ids"]`

## Core Difference From Stable Diffusion

SDXL conditions UNet with three channels of information:

1. `prompt_embeds`
   - token-level hidden states from two text encoders, concatenated on the last dim
2. `add_text_embeds`
   - pooled prompt embedding, mainly from the second text encoder path
3. `add_time_ids`
   - metadata tensor encoding original size, crop offset, and target size

That means train code cannot stop at "text encoder hidden states". You also need the auxiliary conditioning path.

## Prompt Encoding Logic

`encode_prompt(...)` does the heavy lifting.

Flow:

1. Normalize `prompt` and `prompt_2`.
2. Run both tokenizers and both CLIP text encoders.
3. From each encoder output:
   - take pooled output when available
   - take penultimate hidden state by default
   - use an earlier hidden state when `clip_skip` is set
4. Concatenate the two token-level embeddings.
5. Build negative prompt embeddings for CFG.
6. Repeat prompt and pooled embeddings for `num_images_per_prompt`.

Special behavior:

- `force_zeros_for_empty_prompt=True` can zero out negative embeddings when no negative prompt is passed.
- LoRA scaling is applied to both text encoders.

## Additional Conditioning Path

`_get_add_time_ids(...)` builds the SDXL metadata vector from:

- `original_size`
- `crops_coords_top_left`
- `target_size`

During `__call__`, the pipeline prepares:

- `add_text_embeds = pooled_prompt_embeds`
- `add_time_ids`
- optional negative variants for CFG

Then they are packed into:

```python
added_cond_kwargs = {
    "text_embeds": add_text_embeds,
    "time_ids": add_time_ids,
}
```

If IP-Adapter is enabled, `"image_embeds"` is appended too.

## `__call__` Main Inference Flow

High-level order:

1. set default `height`, `width`, `original_size`, `target_size`
2. `check_inputs(...)`
3. `encode_prompt(...)`
4. `retrieve_timesteps(...)`
5. `prepare_latents(...)`
6. `prepare_extra_step_kwargs(...)`
7. build `add_text_embeds` and `add_time_ids`
8. optionally prepare IP-Adapter embeddings
9. optionally apply `denoising_end`
10. run denoising loop
11. upcast / decode VAE latents
12. optional watermark
13. postprocess

## Denoising Loop

The loop is still standard latent diffusion UNet sampling, but the forward call has richer inputs:

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

Where:

- `encoder_hidden_states` = concatenated token embeddings
- `added_cond_kwargs["text_embeds"]` = pooled prompt embeddings
- `added_cond_kwargs["time_ids"]` = size/crop metadata

CFG, `guidance_rescale`, scheduler stepping, and callback handling are conceptually the same as in SD.

## Minimal Training Mental Model

If you adapt SD training code to SDXL, the minimum extra work is:

1. compute token-level embeddings from both text encoders
2. concatenate them into `prompt_embeds`
3. compute pooled embeddings
4. build `time_ids`
5. pass both through `added_cond_kwargs`

The train-time denoiser step then becomes:

- noisy latent input
- timestep
- `encoder_hidden_states=prompt_embeds`
- `added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}`

## What Matters Most For Reuse

- SDXL training is mostly "SD + auxiliary conditioning".
- The auxiliary path is mandatory for faithful reproduction.
- If your training code ignores pooled embeddings or `time_ids`, it is not really SDXL training anymore.
- Watermarking, postprocess, and most callback logic are inference-only.
