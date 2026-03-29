from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from common.config import load_yaml_config, ensure_section
from common.diffusers_import import prepare_diffusers_import
from common.runtime import append_jsonl, ensure_dir


def collect_prompt_records(path: str | Path) -> list[dict[str, Any]]:
    prompts_path = Path(path).expanduser().resolve()
    if prompts_path.is_dir():
        return _collect_prompt_records_from_directory(prompts_path)
    if prompts_path.is_file():
        return _collect_prompt_records_from_file(prompts_path)
    raise FileNotFoundError(f"Prompt path does not exist: {prompts_path}")


def read_prompts(path: str | Path) -> list[str]:
    return [record["prompt"] for record in collect_prompt_records(path)]


def _collect_prompt_records_from_file(prompts_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with prompts_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            prompt = line.strip()
            if not prompt:
                continue
            records.append(
                {
                    "prompt": prompt,
                    "prompt_source_path": str(prompts_path),
                    "prompt_source_kind": "single_file",
                    "prompt_source_index": line_number,
                }
            )
    if not records:
        raise ValueError(f"No prompts found in {prompts_path}")
    return records


def _collect_prompt_records_from_directory(prompts_dir: Path) -> list[dict[str, Any]]:
    prompt_files = sorted(
        path for path in prompts_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )
    if not prompt_files:
        raise ValueError(f"No .txt prompt files found in {prompts_dir}")

    records: list[dict[str, Any]] = []
    for file_index, prompt_file in enumerate(prompt_files, start=1):
        prompt = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError(f"Prompt file is empty: {prompt_file}")
        records.append(
            {
                "prompt": prompt,
                "prompt_source_path": str(prompt_file),
                "prompt_source_kind": "directory_file",
                "prompt_source_index": file_index,
            }
        )
    return records


def run_stable_diffusion_inference(config: dict) -> None:
    prepare_diffusers_import()
    from diffusers import StableDiffusionPipeline

    model_cfg = ensure_section(config, "model")
    infer_cfg = ensure_section(config, "infer")
    output_dir = ensure_dir(infer_cfg["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_records = collect_prompt_records(infer_cfg["prompts_path"])

    pipe = StableDiffusionPipeline.from_pretrained(
        model_cfg["pretrained_path"],
        local_files_only=model_cfg.get("local_files_only", True),
        safety_checker=None,
    ).to(device)
    pipe.load_lora_weights(model_cfg["lora_path"])

    seed = infer_cfg.get("seed", 3407)
    generator = torch.Generator(device=device.type).manual_seed(seed)
    records = []
    for index, prompt_record in enumerate(prompt_records):
        prompt = prompt_record["prompt"]
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
                "seed": seed,
                "prompt_source_path": prompt_record["prompt_source_path"],
                "prompt_source_kind": prompt_record["prompt_source_kind"],
                "prompt_source_index": prompt_record["prompt_source_index"],
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
