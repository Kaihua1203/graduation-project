from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from common.config import ensure_section, load_yaml_config
from common.runtime import ensure_dir, write_json
from uncond_ldm.checkpointing import load_export_bundle_paths, load_inference_components
from uncond_ldm.pipeline import UnconditionalLatentDiffusionPipeline


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize_infer_config(config: dict[str, Any]) -> dict[str, Any]:
    model = ensure_section(config, "model")
    infer = ensure_section(config, "infer")
    scheduler = infer.get("scheduler") or {}
    if not isinstance(scheduler, dict):
        raise ValueError("infer.scheduler must be a mapping.")

    normalized = {
        "model": {
            "export_bundle_path": _require_string(model, "export_bundle_path"),
            "vae_path": _optional_string(model.get("vae_path"), "vae_path"),
            "local_files_only": bool(model.get("local_files_only", True)),
        },
        "infer": {
            "reference_image_dir": _require_string(infer, "reference_image_dir"),
            "output_dir": _require_string(infer, "output_dir"),
            "batch_size": int(infer.get("batch_size", 1)),
            "num_inference_steps": int(infer.get("num_inference_steps", 50)),
            "seed": int(infer.get("seed", 3407)),
            "image_size": int(infer.get("image_size", 512)),
            "scheduler": {
                "type": _optional_string(scheduler.get("type"), "infer.scheduler.type"),
            },
        },
    }
    if normalized["infer"]["batch_size"] <= 0:
        raise ValueError("infer.batch_size must be positive.")
    if normalized["infer"]["num_inference_steps"] <= 0:
        raise ValueError("infer.num_inference_steps must be positive.")
    if normalized["infer"]["image_size"] <= 0:
        raise ValueError("infer.image_size must be positive.")
    return normalized


def collect_reference_images(path: str | Path) -> list[Path]:
    reference_dir = Path(path).expanduser().resolve()
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"Reference image directory does not exist: {reference_dir}")
    image_paths = sorted(
        candidate
        for candidate in reference_dir.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No reference images found in {reference_dir}")
    return image_paths


def load_existing_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_path = record.get("generated_image_path")
            if isinstance(image_path, str):
                records[Path(image_path).name] = record
    return records


def write_metadata_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def build_pipeline(config: dict[str, Any]) -> tuple[UnconditionalLatentDiffusionPipeline, Any]:
    model_cfg = config["model"]
    infer_cfg = config["infer"]
    bundle = load_export_bundle_paths(model_cfg["export_bundle_path"])
    unet, scheduler, vae = load_inference_components(
        bundle,
        vae_path=model_cfg["vae_path"],
        scheduler_type=infer_cfg["scheduler"]["type"],
        local_files_only=model_cfg["local_files_only"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = UnconditionalLatentDiffusionPipeline(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        device=device,
        image_size=infer_cfg["image_size"],
    )
    return pipeline, bundle


def run_unconditional_inference(config: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    normalized = normalize_infer_config(config)
    infer_cfg = normalized["infer"]
    reference_images = collect_reference_images(infer_cfg["reference_image_dir"])
    output_dir = ensure_dir(infer_cfg["output_dir"])
    metadata_path = output_dir / "metadata.jsonl"
    existing_metadata = load_existing_metadata(metadata_path) if resume else {}

    pipeline, bundle = build_pipeline(normalized)

    records: list[dict[str, Any]] = []
    pending_indices: list[int] = []
    pending_references: list[Path] = []

    for index, reference_path in enumerate(reference_images):
        output_path = output_dir / reference_path.name
        if resume and output_path.exists():
            records.append(
                _build_metadata_record(
                    reference_path=reference_path,
                    generated_path=output_path,
                    sample_index=index,
                    seed=infer_cfg["seed"] + index,
                    existing_record=existing_metadata.get(output_path.name),
                )
            )
            continue
        pending_indices.append(index)
        pending_references.append(reference_path)

    batch_size = infer_cfg["batch_size"]
    for batch_start in range(0, len(pending_indices), batch_size):
        batch_indices = pending_indices[batch_start : batch_start + batch_size]
        batch_references = pending_references[batch_start : batch_start + batch_size]
        batch_seeds = [infer_cfg["seed"] + sample_index for sample_index in batch_indices]
        images = pipeline.generate(seeds=batch_seeds, num_inference_steps=infer_cfg["num_inference_steps"])
        for sample_index, reference_path, seed, image in zip(batch_indices, batch_references, batch_seeds, images):
            output_path = output_dir / reference_path.name
            image.save(output_path)
            records.append(
                _build_metadata_record(
                    reference_path=reference_path,
                    generated_path=output_path,
                    sample_index=sample_index,
                    seed=seed,
                )
            )

    records.sort(key=lambda record: int(record["sample_index"]))
    write_metadata_jsonl(records, metadata_path)
    summary_path = write_json(
        {
            "bundle_dir": str(bundle.bundle_dir),
            "metadata_path": str(metadata_path),
            "num_generated": len(pending_indices),
            "num_total": len(reference_images),
            "output_dir": str(output_dir),
            "reference_image_dir": str(Path(infer_cfg["reference_image_dir"]).expanduser().resolve()),
            "resume": bool(resume),
            "training_summary_path": str(bundle.training_summary_path),
        },
        output_dir / "run_summary.json",
    )
    return {
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "run_summary_path": str(summary_path),
        "num_generated": len(pending_indices),
        "num_total": len(reference_images),
    }


def _build_metadata_record(
    *,
    reference_path: Path,
    generated_path: Path,
    sample_index: int,
    seed: int,
    existing_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "generated_image_path": str(generated_path.resolve()),
        "reference_image_path": str(reference_path.resolve()),
        "sample_index": int(sample_index),
        "seed": int(seed),
    }
    if existing_record:
        merged_record = dict(existing_record)
        merged_record.update(record)
        return merged_record
    return record


def _require_string(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for '{key}'.")
    return value


def _optional_string(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for '{key}'.")
    return value


def parse_args(input_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unconditional latent diffusion inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Skip output files that already exist.")
    return parser.parse_args(input_args)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    result = run_unconditional_inference(config, resume=args.resume)
    print(
        "Generated "
        f"{result['num_generated']} / {result['num_total']} unconditional samples at {result['output_dir']}"
    )


if __name__ == "__main__":
    main()
