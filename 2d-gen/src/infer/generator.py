from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.config import load_yaml_config, ensure_section
from common.diffusers_import import prepare_diffusers_import
from common.runtime import append_jsonl, ensure_dir


def read_prompts(path: str | Path) -> list[str]:
    prompts_path = Path(path).expanduser().resolve()
    with prompts_path.open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_path}")
    return prompts


def run_stable_diffusion_inference(config: dict) -> None:
    prepare_diffusers_import()
    from diffusers import StableDiffusionPipeline

    model_cfg = ensure_section(config, "model")
    infer_cfg = ensure_section(config, "infer")
    output_dir = ensure_dir(infer_cfg["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_cfg["pretrained_path"],
        local_files_only=model_cfg.get("local_files_only", True),
        safety_checker=None,
    ).to(device)
    pipe.load_lora_weights(model_cfg["lora_path"])

    generator = torch.Generator(device=device.type).manual_seed(infer_cfg.get("seed", 3407))
    prompts = read_prompts(infer_cfg["prompts_path"])
    records = []
    for index, prompt in enumerate(prompts):
        image = pipe(
            prompt=prompt,
            num_inference_steps=infer_cfg["num_inference_steps"],
            guidance_scale=infer_cfg["guidance_scale"],
            height=infer_cfg["height"],
            width=infer_cfg["width"],
            generator=generator,
        ).images[0]
        image_path = output_dir / f"sample_{index:05d}.png"
        image.save(image_path)
        records.append(
            {
                "image_path": str(image_path),
                "prompt": prompt,
                "seed": infer_cfg.get("seed", 3407),
            }
        )

    append_jsonl(records, output_dir / "metadata.jsonl")
    print(f"Generated {len(records)} images at {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2d-gen inference.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    family = ensure_section(config, "model")["family"]
    if family != "stable_diffusion":
        raise NotImplementedError(f"Inference for {family} is not implemented yet.")
    run_stable_diffusion_inference(config)


if __name__ == "__main__":
    main()
