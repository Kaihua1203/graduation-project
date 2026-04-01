# Diffusers Training Docs

## Scope

Use these notes when you are analyzing or implementing training code for diffusers-based generation pipelines.

The docs are organized by disclosure level:

1. `training_mapping.md` - cross-model mental model and family-level differences.
2. `training_architecture.md` - trainer and adapter design for implementation work.
3. `stable_diffusion.md`, `stable_diffusion_3.md`, `sdxl.md`, `flux.md`, `qwenimage.md` - model-specific notes.
4. `reference/` - upstream training scripts for concrete implementation details.

## Reading Strategy

- Start with the highest-level doc that answers your question.
- Drill down only when you need the next layer of detail.
- Use `reference/` only when you need to confirm a specific upstream training pattern or script behavior.
- Do not treat the order as mandatory; treat it as a narrowing path.

## Reference Directory

- `reference/train_text_to_image_lora_sd.py`: Stable Diffusion text-to-image LoRA reference.
- `reference/train_text_to_image_lora_sdxl.py`: SDXL text-to-image LoRA reference.
- `reference/train_dreambooth_lora_sd.py`: Stable Diffusion DreamBooth LoRA reference.
- `reference/train_image_to_image_lora_kontext.py`: upstream image-to-image LoRA reference used for adapter and batch-shape comparisons.

## When To Stop

Stop at the first layer that gives you enough context to make a correct change.

- If you are mapping model families, stop at `training_mapping.md`.
- If you are designing trainer code, read `training_architecture.md` next.
- If you are matching upstream behavior, open the relevant file under `reference/`.
