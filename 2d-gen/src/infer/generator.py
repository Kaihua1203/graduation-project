from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import torch

from common.config import load_yaml_config, ensure_section
from common.diffusers_import import prepare_diffusers_import
from common.runtime import append_jsonl, ensure_dir


SAMPLE_IMAGE_PATTERN = re.compile(r"^sample_(\d+)\.png$")


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


def _normalize_gpu_ids(infer_cfg: dict[str, Any]) -> list[int]:
    gpu_ids = infer_cfg.get("gpu_ids", [])
    if gpu_ids is None:
        return []
    if not isinstance(gpu_ids, list):
        raise ValueError("infer.gpu_ids must be a list of integers.")
    normalized_gpu_ids: list[int] = []
    for gpu_id in gpu_ids:
        if not isinstance(gpu_id, int):
            raise ValueError("infer.gpu_ids must contain integers only.")
        normalized_gpu_ids.append(gpu_id)
    if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
        raise ValueError("infer.gpu_ids must not contain duplicates.")
    return normalized_gpu_ids


def _build_output_image_path(output_dir: Path, sample_index: int, prompt_record: dict[str, Any]) -> Path:
    if prompt_record["prompt_source_kind"] == "directory_file":
        prompt_source_path = Path(prompt_record["prompt_source_path"])
        return output_dir / f"{prompt_source_path.stem}.png"
    return output_dir / f"sample_{sample_index:05d}.png"


def _discover_completed_sample_indices(output_dir: Path, prompt_records: list[dict[str, Any]]) -> set[int]:
    completed_indices: set[int] = set()
    for sample_index, prompt_record in enumerate(prompt_records):
        image_path = _build_output_image_path(output_dir, sample_index, prompt_record)
        if image_path.exists():
            completed_indices.add(sample_index)
            continue

        match = SAMPLE_IMAGE_PATTERN.match(image_path.name)
        if match is not None and (output_dir / image_path.name).exists():
            completed_indices.add(int(match.group(1)))
    return completed_indices


def _write_jsonl_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _read_jsonl_records(input_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not input_path.exists():
        return records
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def run_stable_diffusion_inference(config: dict, resume: bool = False) -> None:
    prepare_diffusers_import()
    from diffusers import FluxPipeline, StableDiffusion3Pipeline, StableDiffusionPipeline
    from accelerate import PartialState

    model_cfg = ensure_section(config, "model")
    infer_cfg = ensure_section(config, "infer")
    output_dir = ensure_dir(infer_cfg["output_dir"])
    gpu_ids = _normalize_gpu_ids(infer_cfg)
    prompt_records = collect_prompt_records(infer_cfg["prompts_path"])
    distributed_state = PartialState()

    if gpu_ids and distributed_state.num_processes != len(gpu_ids):
        raise ValueError(
            "infer.gpu_ids count does not match active process count. "
            "Launch with accelerate using --num_processes equal to len(infer.gpu_ids)."
        )
    device = distributed_state.device

    family = model_cfg["family"]
    if family == "stable_diffusion":
        pipeline_cls = StableDiffusionPipeline
    elif family == "stable_diffusion_3":
        pipeline_cls = StableDiffusion3Pipeline
    elif family == "flux":
        pipeline_cls = FluxPipeline
    else:
        raise NotImplementedError(f"Inference for {family} is not implemented yet.")

    pipeline_kwargs: dict[str, Any] = {
        "local_files_only": model_cfg.get("local_files_only", True),
    }
    if family == "stable_diffusion":
        pipeline_kwargs["safety_checker"] = None
    if family == "flux":
        pipeline_kwargs["torch_dtype"] = torch.bfloat16

    pipe = pipeline_cls.from_pretrained(model_cfg["pretrained_path"], **pipeline_kwargs).to(device)
    pipe.load_lora_weights(model_cfg["lora_path"])

    seed = int(infer_cfg.get("seed", 3407))
    completed_indices = _discover_completed_sample_indices(output_dir, prompt_records) if resume else set()
    all_indices = list(range(len(prompt_records)))
    pending_indices = [index for index in all_indices if index not in completed_indices]
    if not pending_indices:
        if distributed_state.is_main_process:
            print(f"All {len(prompt_records)} prompts are already generated at {output_dir}; nothing to do.")
        return

    local_records: list[dict[str, Any]] = []
    with distributed_state.split_between_processes(pending_indices) as local_indices:
        for index in local_indices:
            prompt_record = prompt_records[index]
            prompt = prompt_record["prompt"]
            generator = torch.Generator(device=device.type).manual_seed(seed + index)
            inference_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "num_inference_steps": infer_cfg["num_inference_steps"],
                "guidance_scale": infer_cfg["guidance_scale"],
                "height": infer_cfg["height"],
                "width": infer_cfg["width"],
                "generator": generator,
            }
            image = pipe(**inference_kwargs).images[0]
            image_path = _build_output_image_path(output_dir, index, prompt_record)
            image.save(image_path)
            local_records.append(
                {
                    "sample_index": index,
                    "image_path": str(image_path),
                    "prompt": prompt,
                    "seed": seed + index,
                    "prompt_source_path": prompt_record["prompt_source_path"],
                    "prompt_source_kind": prompt_record["prompt_source_kind"],
                    "prompt_source_index": prompt_record["prompt_source_index"],
                }
            )

    metadata_shard_dir = output_dir / ".metadata_shards"
    shard_path = metadata_shard_dir / f"rank_{distributed_state.process_index}.jsonl"
    _write_jsonl_records(local_records, shard_path)

    distributed_state.wait_for_everyone()
    if distributed_state.is_main_process:
        merged_records: list[dict[str, Any]] = []
        for process_index in range(distributed_state.num_processes):
            merged_records.extend(_read_jsonl_records(metadata_shard_dir / f"rank_{process_index}.jsonl"))
        if merged_records:
            merged_records.sort(key=lambda record: int(record["sample_index"]))
            append_jsonl(merged_records, output_dir / "metadata.jsonl")

        for process_index in range(distributed_state.num_processes):
            (metadata_shard_dir / f"rank_{process_index}.jsonl").unlink(missing_ok=True)
        if metadata_shard_dir.exists():
            for stale_shard in metadata_shard_dir.glob("*.jsonl"):
                stale_shard.unlink(missing_ok=True)
            metadata_shard_dir.rmdir()
        print(f"Generated {len(merged_records)} images at {output_dir}")
    distributed_state.wait_for_everyone()


def parse_args(input_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 2d-gen inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume inference from existing outputs based on the configured prompt source.",
    )
    return parser.parse_args(input_args)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    family = ensure_section(config, "model")["family"]
    if family not in {"stable_diffusion", "stable_diffusion_3", "flux"}:
        raise NotImplementedError(f"Inference for {family} is not implemented yet.")
    run_stable_diffusion_inference(config, resume=args.resume)


if __name__ == "__main__":
    main()
