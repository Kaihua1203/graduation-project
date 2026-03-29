from __future__ import annotations

import argparse
import hashlib
import logging
import math
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from common.diffusers_import import prepare_diffusers_import
from common.runtime import ensure_dir, write_json

prepare_diffusers_import()

import diffusers  # noqa: E402
import transformers  # noqa: E402
from diffusers import (  # noqa: E402
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from diffusers.training_utils import cast_training_params, compute_snr  # noqa: E402
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft  # noqa: E402
from diffusers.utils.import_utils import is_xformers_available  # noqa: E402


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args(input_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone SD1.5 DreamBooth LoRA training from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the DreamBooth YAML config.")
    return parser.parse_args(input_args)


def load_dreambooth_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"DreamBooth config at {config_path} must decode to a mapping.")
    normalized = normalize_dreambooth_config(config)
    normalized["_config_path"] = str(config_path)
    return normalized


def normalize_dreambooth_config(config: dict[str, Any]) -> dict[str, Any]:
    model = _ensure_mapping(config, "model")
    data = _ensure_mapping(config, "data")
    train = _ensure_mapping(config, "train")
    validation = _optional_mapping(config, "validation")
    logging_cfg = _optional_mapping(config, "logging")

    normalized = {
        "model": {
            "pretrained_model_name_or_path": _require_existing_path(
                model, "pretrained_model_name_or_path", must_be_dir=True
            ),
            "pretrained_vae_model_name_or_path": _optional_existing_path(
                model, "pretrained_vae_model_name_or_path", must_be_dir=True
            ),
            "revision": model.get("revision"),
            "variant": model.get("variant"),
            "local_files_only": bool(model.get("local_files_only", True)),
        },
        "data": {
            "instance_data_dir": _require_existing_path(data, "instance_data_dir", must_be_dir=True),
            "instance_prompt": _require_string(data, "instance_prompt"),
            "class_data_dir": _optional_existing_path(data, "class_data_dir", must_be_dir=True, create=True),
            "class_prompt": data.get("class_prompt"),
            "with_prior_preservation": bool(data.get("with_prior_preservation", False)),
            "num_class_images": int(data.get("num_class_images", 100)),
            "resolution": int(data.get("resolution", 512)),
            "center_crop": bool(data.get("center_crop", False)),
            "image_interpolation_mode": str(data.get("image_interpolation_mode", "lanczos")),
        },
        "train": {
            "output_dir": _require_string(train, "output_dir"),
            "seed": train.get("seed"),
            "train_batch_size": int(train.get("train_batch_size", 1)),
            "sample_batch_size": int(train.get("sample_batch_size", 4)),
            "num_train_epochs": int(train.get("num_train_epochs", 1)),
            "max_train_steps": train.get("max_train_steps"),
            "checkpointing_steps": int(train.get("checkpointing_steps", 500)),
            "checkpoints_total_limit": train.get("checkpoints_total_limit"),
            "resume_from_checkpoint": train.get("resume_from_checkpoint"),
            "gradient_accumulation_steps": int(train.get("gradient_accumulation_steps", 1)),
            "gradient_checkpointing": bool(train.get("gradient_checkpointing", False)),
            "learning_rate": float(train.get("learning_rate", 5.0e-4)),
            "scale_lr": bool(train.get("scale_lr", False)),
            "lr_scheduler": str(train.get("lr_scheduler", "constant")),
            "lr_warmup_steps": int(train.get("lr_warmup_steps", 0)),
            "lr_num_cycles": int(train.get("lr_num_cycles", 1)),
            "lr_power": float(train.get("lr_power", 1.0)),
            "dataloader_num_workers": int(train.get("dataloader_num_workers", 0)),
            "max_grad_norm": float(train.get("max_grad_norm", 1.0)),
            "allow_tf32": bool(train.get("allow_tf32", False)),
            "mixed_precision": _normalize_precision_value(train.get("mixed_precision", "no")),
            "prior_generation_precision": _normalize_precision_value(
                train.get("prior_generation_precision", "fp16")
            ),
            "enable_xformers_memory_efficient_attention": bool(
                train.get("enable_xformers_memory_efficient_attention", False)
            ),
            "noise_offset": float(train.get("noise_offset", 0.0)),
            "prior_loss_weight": float(train.get("prior_loss_weight", 1.0)),
            "snr_gamma": train.get("snr_gamma"),
            "optimizer": _normalize_optimizer_config(train.get("optimizer")),
            "lora": _normalize_lora_config(train.get("lora")),
        },
        "validation": {
            "validation_prompt": validation.get("validation_prompt"),
            "num_validation_images": int(validation.get("num_validation_images", 4)),
            "validation_epochs": int(validation.get("validation_epochs", 1)),
        },
        "logging": {
            "report_to": str(logging_cfg.get("report_to", "none")),
            "logging_dir": str(logging_cfg.get("logging_dir", "logs")),
            "log_every_n_steps": int(logging_cfg.get("log_every_n_steps", 10)),
            "tracker_project_name": str(logging_cfg.get("tracker_project_name", "2d-gen-dreambooth-sd15")),
        },
    }

    _validate_dreambooth_config(normalized)
    return normalized


def _ensure_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing mapping section: {key}")
    return value


def _optional_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{key}' must be a mapping.")
    return value


def _require_string(section: dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for '{key}'.")
    return value


def _require_existing_path(
    section: dict[str, Any],
    key: str,
    *,
    must_be_dir: bool,
) -> str:
    value = _require_string(section, key)
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Expected existing path for '{key}': {path}")
    if must_be_dir and not path.is_dir():
        raise ValueError(f"Expected directory for '{key}': {path}")
    return str(path)


def _optional_existing_path(
    section: dict[str, Any],
    key: str,
    *,
    must_be_dir: bool,
    create: bool = False,
) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected string or null for '{key}'.")
    path = Path(value).expanduser().resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        raise FileNotFoundError(f"Expected existing path for '{key}': {path}")
    if must_be_dir and not path.is_dir():
        raise ValueError(f"Expected directory for '{key}': {path}")
    return str(path)


def _normalize_optimizer_config(value: Any) -> dict[str, Any]:
    config = value or {}
    if not isinstance(config, dict):
        raise ValueError("train.optimizer must be a mapping.")
    return {
        "use_8bit_adam": bool(config.get("use_8bit_adam", False)),
        "adam_beta1": float(config.get("adam_beta1", 0.9)),
        "adam_beta2": float(config.get("adam_beta2", 0.999)),
        "adam_weight_decay": float(config.get("adam_weight_decay", 1.0e-2)),
        "adam_epsilon": float(config.get("adam_epsilon", 1.0e-8)),
    }


def _normalize_lora_config(value: Any) -> dict[str, Any]:
    config = value or {}
    if not isinstance(config, dict):
        raise ValueError("train.lora must be a mapping.")
    target_modules = config.get("target_modules") or ["to_k", "to_q", "to_v", "to_out.0"]
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("train.lora.target_modules must be a non-empty list.")
    return {
        "rank": int(config.get("rank", 8)),
        "alpha": int(config.get("alpha", int(config.get("rank", 8)))),
        "dropout": float(config.get("dropout", 0.0)),
        "target_modules": [str(module_name) for module_name in target_modules],
    }


def _normalize_precision_value(value: Any) -> str:
    if value is False:
        return "no"
    if isinstance(value, str):
        return value
    raise ValueError("Precision values must be encoded as 'no', 'fp16', 'bf16', or 'fp32'.")


def _validate_dreambooth_config(config: dict[str, Any]) -> None:
    train_cfg = config["train"]
    data_cfg = config["data"]
    validation_cfg = config["validation"]

    if train_cfg["train_batch_size"] <= 0:
        raise ValueError("train.train_batch_size must be positive.")
    if train_cfg["sample_batch_size"] <= 0:
        raise ValueError("train.sample_batch_size must be positive.")
    if train_cfg["num_train_epochs"] <= 0:
        raise ValueError("train.num_train_epochs must be positive.")
    if train_cfg["gradient_accumulation_steps"] <= 0:
        raise ValueError("train.gradient_accumulation_steps must be positive.")
    if train_cfg["dataloader_num_workers"] < 0:
        raise ValueError("train.dataloader_num_workers must be non-negative.")
    if train_cfg["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("train.mixed_precision must be one of: no, fp16, bf16.")
    if train_cfg["prior_generation_precision"] not in {"no", "fp32", "fp16", "bf16"}:
        raise ValueError("train.prior_generation_precision must be one of: no, fp32, fp16, bf16.")
    if validation_cfg["num_validation_images"] <= 0:
        raise ValueError("validation.num_validation_images must be positive.")
    if validation_cfg["validation_epochs"] <= 0:
        raise ValueError("validation.validation_epochs must be positive.")
    if train_cfg["max_train_steps"] is not None and int(train_cfg["max_train_steps"]) <= 0:
        raise ValueError("train.max_train_steps must be positive when provided.")
    if train_cfg["optimizer"]["use_8bit_adam"]:
        raise ValueError("train.optimizer.use_8bit_adam is not supported in the standalone DreamBooth trainer.")
    if train_cfg["prior_loss_weight"] < 0:
        raise ValueError("train.prior_loss_weight must be non-negative.")

    if data_cfg["with_prior_preservation"]:
        if data_cfg["class_data_dir"] is None:
            raise ValueError("data.class_data_dir is required when data.with_prior_preservation=true.")
        if not isinstance(data_cfg.get("class_prompt"), str) or not data_cfg["class_prompt"].strip():
            raise ValueError("data.class_prompt is required when data.with_prior_preservation=true.")
        if data_cfg["num_class_images"] <= 0:
            raise ValueError("data.num_class_images must be positive.")


def flatten_config(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else key
            flatten_config(nested_prefix, nested_value, output)
        return
    output[prefix] = value


def should_run_validation(config: dict[str, Any], epoch: int) -> bool:
    validation_cfg = config["validation"]
    prompt = validation_cfg.get("validation_prompt")
    return bool(prompt) and (epoch + 1) % validation_cfg["validation_epochs"] == 0


def resolve_prior_generation_dtype(precision: str, accelerator: Accelerator) -> torch.dtype:
    if precision in {"no", "fp32"}:
        return torch.float32
    if precision == "fp16":
        if accelerator.device.type in {"cuda", "xpu", "mps"}:
            return torch.float16
        LOGGER.warning(
            "prior_generation_precision=fp16 is not supported on device type %s; falling back to fp32.",
            accelerator.device.type,
        )
        return torch.float32
    if precision == "bf16":
        if accelerator.device.type == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            LOGGER.warning("prior_generation_precision=bf16 is not supported by the current CUDA device; falling back to fp32.")
            return torch.float32
        if accelerator.device.type == "cpu":
            return torch.bfloat16
        LOGGER.warning(
            "prior_generation_precision=bf16 is not supported on device type %s; falling back to fp32.",
            accelerator.device.type,
        )
        return torch.float32
    raise ValueError(f"Unsupported prior generation precision: {precision}")


class PromptDataset(Dataset):
    def __init__(self, prompt: str, num_samples: int) -> None:
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"prompt": self.prompt, "index": index}


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        *,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer: Any,
        class_data_root: str | None = None,
        class_prompt: str | None = None,
        class_num: int | None = None,
        size: int = 512,
        center_crop: bool = False,
        image_interpolation_mode: str = "lanczos",
    ) -> None:
        self.tokenizer = tokenizer
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        self.instance_images_path = _list_image_files(instance_data_root)
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.class_images_path: list[Path] = []
        if class_data_root is not None:
            self.class_images_path = _list_image_files(class_data_root)
            if class_num is not None:
                self.class_images_path = self.class_images_path[:class_num]
            self._length = max(self.num_instance_images, len(self.class_images_path))

        interpolation = getattr(transforms.InterpolationMode, image_interpolation_mode.upper(), None)
        if interpolation is None:
            raise ValueError(f"Unsupported interpolation mode: {image_interpolation_mode}")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=interpolation),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        example: dict[str, Any] = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)
        if instance_image.mode != "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        instance_text_inputs = tokenize_prompt(self.tokenizer, self.instance_prompt)
        example["instance_prompt_ids"] = instance_text_inputs.input_ids
        example["instance_attention_mask"] = instance_text_inputs.attention_mask

        if self.class_images_path:
            class_image = Image.open(self.class_images_path[index % len(self.class_images_path)])
            class_image = exif_transpose(class_image)
            if class_image.mode != "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            class_text_inputs = tokenize_prompt(self.tokenizer, self.class_prompt)
            example["class_prompt_ids"] = class_text_inputs.input_ids
            example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


def _list_image_files(path: str | Path) -> list[Path]:
    root = Path(path).expanduser().resolve()
    files = sorted(
        candidate
        for candidate in root.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise ValueError(f"No image files were found under {root}")
    return files


def collate_fn(examples: list[dict[str, Any]], *, with_prior_preservation: bool) -> dict[str, Any]:
    input_ids = [example["instance_prompt_ids"] for example in examples]
    attention_mask = [example["instance_attention_mask"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    if with_prior_preservation:
        input_ids.extend(example["class_prompt_ids"] for example in examples)
        attention_mask.extend(example["class_attention_mask"] for example in examples)
        pixel_values.extend(example["class_images"] for example in examples)

    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_mask, dim=0),
        "pixel_values": torch.stack(pixel_values).to(memory_format=torch.contiguous_format).float(),
    }


def tokenize_prompt(tokenizer: Any, prompt: str) -> Any:
    return tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )


def encode_prompt(text_encoder: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    prompt_embeds = text_encoder(
        input_ids.to(text_encoder.device),
        attention_mask=attention_mask.to(text_encoder.device),
        return_dict=False,
    )[0]
    return prompt_embeds


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    return accelerator.unwrap_model(model)


def prune_checkpoints(output_dir: Path, checkpoints_total_limit: int | None) -> None:
    if checkpoints_total_limit is None:
        return
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda checkpoint_path: int(checkpoint_path.name.split("-", 1)[1]),
    )
    while len(checkpoints) > checkpoints_total_limit:
        shutil.rmtree(checkpoints.pop(0), ignore_errors=True)


def resolve_resume_checkpoint(output_dir: Path, resume_from_checkpoint: str | None) -> Path | None:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        candidates = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda checkpoint_path: int(checkpoint_path.name.split("-", 1)[1]),
        )
        return candidates[-1] if candidates else None

    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = output_dir / checkpoint_path
    return checkpoint_path.resolve()


def generate_class_images(config: dict[str, Any], accelerator: Accelerator) -> None:
    data_cfg = config["data"]
    train_cfg = config["train"]
    model_cfg = config["model"]
    class_images_dir = ensure_dir(data_cfg["class_data_dir"])
    existing_images = _list_image_files(class_images_dir) if any(class_images_dir.iterdir()) else []

    if len(existing_images) >= data_cfg["num_class_images"]:
        return

    torch_dtype = resolve_prior_generation_dtype(train_cfg["prior_generation_precision"], accelerator)

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        revision=model_cfg.get("revision"),
        variant=model_cfg.get("variant"),
        local_files_only=model_cfg.get("local_files_only", True),
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    num_new_images = data_cfg["num_class_images"] - len(existing_images)
    LOGGER.info("Generating %s class images for prior preservation.", num_new_images)
    sample_dataset = PromptDataset(data_cfg["class_prompt"], num_new_images)
    sample_dataloader = DataLoader(sample_dataset, batch_size=train_cfg["sample_batch_size"], shuffle=False)
    sample_dataloader = accelerator.prepare(sample_dataloader)

    for batch in tqdm(
        sample_dataloader,
        desc="Generating class images",
        disable=not accelerator.is_local_main_process,
    ):
        generated = pipeline(list(batch["prompt"])).images
        for offset, image in enumerate(generated):
            sample_index = int(batch["index"][offset]) + len(existing_images)
            digest = hashlib.sha1(image.tobytes()).hexdigest()
            image_path = class_images_dir / f"{sample_index}-{digest}.png"
            image.save(image_path)

    del pipeline
    accelerator.wait_for_everyone()
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()


def register_checkpoint_hooks(
    accelerator: Accelerator,
    unet: torch.nn.Module,
) -> None:
    def save_model_hook(models: list[torch.nn.Module], weights: list[torch.Tensor], output_dir: str) -> None:
        if not accelerator.is_main_process:
            while weights:
                weights.pop()
            return

        unet_lora_layers_to_save = None
        for model in models:
            if not isinstance(model, type(unwrap_model(accelerator, unet))):
                raise ValueError(f"Unexpected save model: {model.__class__}")
            unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            if weights:
                weights.pop()

        StableDiffusionPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_layers_to_save,
            safe_serialization=True,
        )

    def load_model_hook(models: list[torch.nn.Module], input_dir: str) -> None:
        unet_model = None
        while models:
            model = models.pop()
            if not isinstance(model, type(unwrap_model(accelerator, unet))):
                raise ValueError(f"Unexpected load model: {model.__class__}")
            unet_model = model

        lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(input_dir)
        unet_state_dict = {
            key.replace("unet.", ""): value
            for key, value in lora_state_dict.items()
            if key.startswith("unet.")
        }
        set_peft_model_state_dict(unet_model, convert_unet_state_dict_to_peft(unet_state_dict), adapter_name="default")
        if accelerator.mixed_precision == "fp16":
            cast_training_params([unet_model], dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def log_validation(
    *,
    accelerator: Accelerator,
    config: dict[str, Any],
    unet: torch.nn.Module,
    vae: AutoencoderKL,
    tokenizer: Any,
    text_encoder: Any,
    weight_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    phase_name: str,
) -> None:
    validation_cfg = config["validation"]
    prompt = validation_cfg["validation_prompt"]
    if not prompt or validation_cfg["num_validation_images"] <= 0 or not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    LOGGER.info(
        "Running %s validation with %s image(s) for prompt: %s",
        phase_name,
        validation_cfg["num_validation_images"],
        prompt,
    )

    pipeline = StableDiffusionPipeline.from_pretrained(
        config["model"]["pretrained_model_name_or_path"],
        revision=config["model"].get("revision"),
        variant=config["model"].get("variant"),
        local_files_only=config["model"].get("local_files_only", True),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unwrap_model(accelerator, unet),
        torch_dtype=weight_dtype,
        safety_checker=None,
        feature_extractor=None,
    )
    scheduler_args: dict[str, Any] = {}
    variance_type = getattr(pipeline.scheduler.config, "variance_type", None)
    if variance_type in {"learned", "learned_range"}:
        scheduler_args["variance_type"] = "fixed_small"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = None
    if config["train"].get("seed") is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(int(config["train"]["seed"]))

    autocast_context = (
        torch.autocast(accelerator.device.type, dtype=weight_dtype)
        if accelerator.device.type != "cpu" and weight_dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    )
    with autocast_context:
        images = [
            pipeline(prompt=prompt, generator=generator, num_inference_steps=25).images[0]
            for _ in range(validation_cfg["num_validation_images"])
        ]

    if accelerator.trackers:
        accelerator.log({f"{phase_name}/epoch": epoch + 1, f"{phase_name}/global_step": global_step}, step=global_step)
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            tracker.writer.add_images(
                phase_name,
                np.stack([np.asarray(image) for image in images]),
                global_step,
                dataformats="NHWC",
            )
        elif tracker.name == "swanlab":
            tracker.log({f"{phase_name}/prompt": prompt}, step=global_step)
            tracker.log_images({f"{phase_name}/images": images}, step=global_step)

    del pipeline
    if accelerator.device.type == "cuda":
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


def main() -> None:
    args = parse_args()
    config = load_dreambooth_config(args.config)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    output_dir = ensure_dir(config["train"]["output_dir"])
    logging_dir = ensure_dir(output_dir / config["logging"]["logging_dir"])
    accelerator = Accelerator(
        gradient_accumulation_steps=config["train"]["gradient_accumulation_steps"],
        mixed_precision=config["train"]["mixed_precision"],
        log_with=None if config["logging"]["report_to"] == "none" else config["logging"]["report_to"],
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logging_dir)),
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    LOGGER.info(accelerator.state)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if config["train"].get("seed") is not None:
        set_seed(int(config["train"]["seed"]))

    if config["data"]["with_prior_preservation"]:
        generate_class_images(config, accelerator)

    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config["data"]

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_cfg["pretrained_model_name_or_path"],
        revision=model_cfg.get("revision"),
        variant=model_cfg.get("variant"),
        local_files_only=model_cfg.get("local_files_only", True),
        safety_checker=None,
    )
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet
    noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
    del pipeline

    pretrained_vae_path = model_cfg.get("pretrained_vae_model_name_or_path")
    if pretrained_vae_path is not None:
        vae = AutoencoderKL.from_pretrained(
            pretrained_vae_path,
            revision=model_cfg.get("revision"),
            variant=model_cfg.get("variant"),
            local_files_only=model_cfg.get("local_files_only", True),
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    lora_cfg = train_cfg["lora"]
    unet.add_adapter(
        LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )
    )

    if train_cfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    if train_cfg["enable_xformers_memory_efficient_attention"]:
        if not is_xformers_available():
            raise ValueError("xformers is not available but train.enable_xformers_memory_efficient_attention=true.")
        unet.enable_xformers_memory_efficient_attention()

    if train_cfg["allow_tf32"] and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if accelerator.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    train_dataset = DreamBoothDataset(
        instance_data_root=data_cfg["instance_data_dir"],
        instance_prompt=data_cfg["instance_prompt"],
        tokenizer=tokenizer,
        class_data_root=data_cfg["class_data_dir"] if data_cfg["with_prior_preservation"] else None,
        class_prompt=data_cfg["class_prompt"],
        class_num=data_cfg["num_class_images"] if data_cfg["with_prior_preservation"] else None,
        size=data_cfg["resolution"],
        center_crop=data_cfg["center_crop"],
        image_interpolation_mode=data_cfg["image_interpolation_mode"],
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg["train_batch_size"],
        shuffle=True,
        num_workers=train_cfg["dataloader_num_workers"],
        collate_fn=lambda examples: collate_fn(
            examples,
            with_prior_preservation=data_cfg["with_prior_preservation"],
        ),
    )

    learning_rate = train_cfg["learning_rate"]
    if train_cfg["scale_lr"]:
        learning_rate = (
            learning_rate
            * train_cfg["gradient_accumulation_steps"]
            * train_cfg["train_batch_size"]
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in unet.parameters() if parameter.requires_grad],
        lr=learning_rate,
        betas=(
            train_cfg["optimizer"]["adam_beta1"],
            train_cfg["optimizer"]["adam_beta2"],
        ),
        weight_decay=train_cfg["optimizer"]["adam_weight_decay"],
        eps=train_cfg["optimizer"]["adam_epsilon"],
    )

    num_warmup_steps_for_scheduler = train_cfg["lr_warmup_steps"] * accelerator.num_processes
    if train_cfg["max_train_steps"] is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / train_cfg["gradient_accumulation_steps"]
        )
        num_training_steps_for_scheduler = (
            train_cfg["num_train_epochs"] * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = int(train_cfg["max_train_steps"]) * accelerator.num_processes

    lr_scheduler = diffusers.optimization.get_scheduler(
        train_cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=train_cfg["lr_num_cycles"],
        power=train_cfg["lr_power"],
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    register_checkpoint_hooks(accelerator, unet)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_cfg["gradient_accumulation_steps"])
    max_train_steps = int(train_cfg["max_train_steps"] or (train_cfg["num_train_epochs"] * num_update_steps_per_epoch))
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process and config["logging"]["report_to"] != "none":
        tracker_config: dict[str, Any] = {}
        flatten_config("", config, tracker_config)
        accelerator.init_trackers(config["logging"]["tracker_project_name"], config=tracker_config)

    global_step = 0
    first_epoch = 0
    checkpoint_path = resolve_resume_checkpoint(output_dir, train_cfg["resume_from_checkpoint"])
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Requested checkpoint does not exist: {checkpoint_path}")
        accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
        accelerator.load_state(str(checkpoint_path))
        global_step = int(checkpoint_path.name.split("-", 1)[1])
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    loss_history: list[float] = []

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                noise = torch.randn_like(model_input)
                if train_cfg["noise_offset"]:
                    noise = noise + train_cfg["noise_offset"] * torch.randn(
                        (model_input.shape[0], model_input.shape[1], 1, 1),
                        device=model_input.device,
                        dtype=model_input.dtype,
                    )
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (model_input.shape[0],),
                    device=model_input.device,
                    dtype=torch.long,
                )
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                encoder_hidden_states = encode_prompt(
                    text_encoder,
                    batch["input_ids"],
                    batch["attention_mask"],
                ).to(device=accelerator.device, dtype=weight_dtype)

                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False,
                )[0]
                if model_pred.shape[1] == model_input.shape[1] * 2:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                if data_cfg["with_prior_preservation"]:
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    instance_timesteps, _ = torch.chunk(timesteps, 2, dim=0)
                else:
                    instance_timesteps = timesteps

                if train_cfg["snr_gamma"] is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, instance_timesteps)
                    loss_weights = torch.stack(
                        [snr, float(train_cfg["snr_gamma"]) * torch.ones_like(snr)],
                        dim=1,
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        loss_weights = loss_weights / snr
                    else:
                        loss_weights = loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * loss_weights
                    loss = loss.mean()

                if data_cfg["with_prior_preservation"]:
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    loss = loss + train_cfg["prior_loss_weight"] * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [parameter for parameter in unet.parameters() if parameter.requires_grad],
                        train_cfg["max_grad_norm"],
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.detach().item():.4f}")
            loss_history.append(loss.detach().item())
            if config["logging"]["report_to"] != "none":
                accelerator.log(
                    {"train/loss": loss.detach().item(), "train/lr": lr_scheduler.get_last_lr()[0]},
                    step=global_step,
                )

            if global_step % config["logging"]["log_every_n_steps"] == 0:
                accelerator.print(f"step={global_step} loss={loss.detach().item():.6f}")

            if global_step % train_cfg["checkpointing_steps"] == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(checkpoint_dir))
                if accelerator.is_main_process:
                    prune_checkpoints(output_dir, train_cfg["checkpoints_total_limit"])

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

        if should_run_validation(config, epoch):
            log_validation(
                accelerator=accelerator,
                config=config,
                unet=unet,
                vae=vae,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                weight_dtype=weight_dtype,
                epoch=epoch,
                global_step=global_step,
                phase_name="validation",
            )

    accelerator.wait_for_everyone()
    final_output_dir = ensure_dir(output_dir / "final_lora")
    if accelerator.is_main_process:
        final_unet = unwrap_model(accelerator, unet).to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(final_unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=final_output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
        write_json(
            {
                "config_path": config.get("_config_path"),
                "num_steps": global_step,
                "num_updates": len(loss_history),
                "mean_loss": float(sum(loss_history) / max(1, len(loss_history))),
                "min_loss": float(min(loss_history)) if loss_history else math.nan,
                "final_checkpoint": str(final_output_dir),
            },
            output_dir / "train_summary.json",
        )

    if config["validation"]["validation_prompt"] is not None:
        log_validation(
            accelerator=accelerator,
            config=config,
            unet=unet,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            weight_dtype=weight_dtype,
            epoch=max(num_train_epochs - 1, 0),
            global_step=global_step,
            phase_name="test",
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
