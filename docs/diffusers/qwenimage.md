# QwenImage Pipeline

Source: `diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py`

## Scope

QwenImage is structurally closer to Flux than to SD/SDXL:

- transformer denoiser, not UNet
- flow-matching scheduler
- packed latents
- no CLIP/T5 dual-encoder split
- text conditioning comes from a Qwen2.5-VL model

Its most distinctive part is the prompt encoding path: the pipeline turns the user prompt into an instruction-style template before extracting hidden states from the Qwen multimodal language model.

## Registered Modules

`__init__` registers:

- `scheduler`
- `vae`
- `text_encoder`
- `tokenizer`
- `transformer`

Important state:

- `vae_scale_factor = 2 ** len(self.vae.temperal_downsample)`
- `image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)`
- `tokenizer_max_length = 1024`
- `prompt_template_encode`
- `prompt_template_encode_start_idx = 34`
- `default_sample_size = 128`

## Prompt Encoding Logic

The text path is the part you need to understand first.

`_get_qwen_prompt_embeds(...)` does:

1. wrap each prompt with an instruction template:
   - system asks for a detailed image description
   - user content is inserted into the template
2. tokenize with Qwen tokenizer
3. run `Qwen2.5-VLForConditionalGeneration(..., output_hidden_states=True)`
4. take the last hidden state
5. use `_extract_masked_hidden(...)` to remove padding by attention mask
6. drop the prefix tokens before `prompt_template_encode_start_idx`
7. pad the remaining hidden states back to a batch tensor
8. return:
   - `prompt_embeds`
   - `encoder_attention_mask`

Training takeaway:

- The model is not conditioned on raw user text directly.
- It is conditioned on hidden states produced from a fixed instruction prompt template.
- If training code skips this template logic, the conditioning distribution changes.

## Prompt Tensor Format

`encode_prompt(...)` returns:

- `prompt_embeds`: `[B, seq_len, hidden_dim]`
- `prompt_embeds_mask`: `[B, seq_len]` or `None` if fully valid

Unlike Flux:

- there is no separate CLIP pooled embedding
- there is no `text_ids`
- attention masking is explicit and passed into the transformer

## Latent Packing

Like Flux, QwenImage packs latents into 2x2 spatial patches:

- `_pack_latents(...)`
- `_unpack_latents(...)`
- `prepare_latents(...)`

But QwenImage uses a 5D VAE latent shape before packing:

- initial latent shape is `(batch_size, 1, num_channels_latents, height, width)`

Later, before decoding, the unpacked tensor is reshaped back to the VAE's expected format and only the first temporal slice is kept after decode.

## `__call__` Main Inference Flow

High-level order:

1. choose default `height` and `width`
2. `check_inputs(...)`
3. determine whether true CFG is active
4. `encode_prompt(...)` for positive branch
5. optional `encode_prompt(...)` for negative branch
6. `prepare_latents(...)`
7. build `img_shapes`
8. create sigma schedule and shifted timesteps
9. optional guidance embedding for guidance-distilled variants
10. run transformer denoising loop
11. unpack latents
12. restore VAE normalization stats
13. decode and postprocess

## Denoising Loop

The central forward is:

```python
noise_pred = self.transformer(
    hidden_states=latents,
    timestep=timestep / 1000,
    guidance=guidance,
    encoder_hidden_states_mask=prompt_embeds_mask,
    encoder_hidden_states=prompt_embeds,
    img_shapes=img_shapes,
    attention_kwargs=self.attention_kwargs,
)[0]
```

If true CFG is enabled, the pipeline runs a second negative branch and combines outputs:

```python
comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
noise_pred = comb_pred * (cond_norm / noise_norm)
```

That extra norm matching step is a notable difference from Flux and SD-style CFG.

## Scheduler And Guidance Logic

QwenImage also uses `FlowMatchEulerDiscreteScheduler`.

Important behaviors:

- timesteps come from `retrieve_timesteps(...)` with `sigmas` and `mu`
- `guidance_scale` is only meaningful for guidance-distilled models
- traditional CFG is controlled separately by `true_cfg_scale` plus negative prompt presence

So there are two guidance concepts in the file:

- `guidance_scale`
  - model input for guidance-distilled variants
- `true_cfg_scale`
  - classic conditional vs unconditional output mixing

## Minimal Training Mental Model

To reproduce QwenImage-style training, the forward path should preserve:

1. prompt templating
2. Qwen hidden-state extraction
3. prompt attention mask handling
4. packed latent representation
5. flow-matching timestep / sigma logic
6. transformer forward signature with `img_shapes`

## What Matters Most For Reuse

- The prompt template is part of the model interface, not just a convenience wrapper.
- The transformer expects both prompt embeddings and their mask.
- The latent representation is packed, so SD-style UNet training code is not directly reusable.
- If you later write LoRA or full fine-tuning code, the most important interfaces to mirror are:
  - `encode_prompt(...)`
  - `prepare_latents(...)`
  - `transformer(...)` forward arguments
