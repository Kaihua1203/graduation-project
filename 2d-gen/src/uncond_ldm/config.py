from __future__ import annotations

from typing import Any

from common.config import ensure_section, load_yaml_config


def load_uncond_ldm_train_config(path: str) -> dict[str, Any]:
    return normalize_uncond_ldm_train_config(load_yaml_config(path))


def normalize_uncond_ldm_train_config(config: dict[str, Any]) -> dict[str, Any]:
    model = ensure_section(config, "model")
    data = ensure_section(config, "data")
    train = ensure_section(config, "train")
    validation = config.get("validation") or {}
    logging = config.get("logging") or {}
    distributed = config.get("distributed") or {}

    for section_name, section_value in (
        ("validation", validation),
        ("logging", logging),
        ("distributed", distributed),
    ):
        if not isinstance(section_value, dict):
            raise ValueError(f"Section '{section_name}' must be a mapping.")

    unet = model.get("unet") or {}
    scheduler = model.get("scheduler") or {}
    optimizer = train.get("optimizer") or {}

    if not isinstance(unet, dict):
        raise ValueError("model.unet must be a mapping.")
    if not isinstance(scheduler, dict):
        raise ValueError("model.scheduler must be a mapping.")
    if not isinstance(optimizer, dict):
        raise ValueError("train.optimizer must be a mapping.")

    normalized = {
        "model": {
            "model_type": str(model.get("model_type", "uncond_ldm")),
            "pretrained_vae_model_name_or_path": _require_string(model, "pretrained_vae_model_name_or_path"),
            "vae_subfolder": _optional_string(model.get("vae_subfolder"), "vae_subfolder"),
            "local_files_only": bool(model.get("local_files_only", True)),
            "latent_scaling_factor": _optional_float(model.get("latent_scaling_factor"), "latent_scaling_factor"),
            "unet": {
                "sample_size": int(unet.get("sample_size", 64)),
                "in_channels": int(unet.get("in_channels", 4)),
                "out_channels": int(unet.get("out_channels", 4)),
                "layers_per_block": int(unet.get("layers_per_block", 2)),
                "block_out_channels": tuple(int(value) for value in unet.get("block_out_channels", [224, 448, 672, 896])),
                "down_block_types": tuple(
                    str(value)
                    for value in unet.get(
                        "down_block_types",
                        ["DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
                    )
                ),
                "up_block_types": tuple(
                    str(value)
                    for value in unet.get(
                        "up_block_types",
                        ["AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"],
                    )
                ),
                "norm_num_groups": int(unet.get("norm_num_groups", 32)),
                "attention_head_dim": int(unet.get("attention_head_dim", 8)),
                "dropout": float(unet.get("dropout", 0.0)),
            },
            "scheduler": {
                "num_train_timesteps": int(scheduler.get("num_train_timesteps", 1000)),
                "beta_start": float(scheduler.get("beta_start", 1.0e-4)),
                "beta_end": float(scheduler.get("beta_end", 2.0e-2)),
                "beta_schedule": str(scheduler.get("beta_schedule", "linear")),
                "prediction_type": str(scheduler.get("prediction_type", "epsilon")),
                "variance_type": str(scheduler.get("variance_type", "fixed_small")),
                "clip_sample": bool(scheduler.get("clip_sample", False)),
            },
        },
        "data": {
            "train_manifest": _optional_string(data.get("train_manifest"), "train_manifest"),
            "train_image_dir": _optional_string(data.get("train_image_dir"), "train_image_dir"),
            "image_column": str(data.get("image_column", "image_path")),
            "resolution": int(data.get("resolution", 512)),
            "center_crop": bool(data.get("center_crop", False)),
            "random_flip": bool(data.get("random_flip", False)),
            "image_interpolation_mode": str(data.get("image_interpolation_mode", "bilinear")),
            "max_train_samples": _optional_int(data.get("max_train_samples"), "max_train_samples"),
            "allowed_extensions": [str(value).lower() for value in data.get("allowed_extensions", [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])],
        },
        "train": {
            "output_dir": _require_string(train, "output_dir"),
            "seed": int(train.get("seed", 3407)),
            "train_batch_size": int(train.get("train_batch_size", 1)),
            "num_train_epochs": int(train.get("num_train_epochs", 1)),
            "max_train_steps": int(train.get("max_train_steps", 1)),
            "gradient_accumulation_steps": int(train.get("gradient_accumulation_steps", 1)),
            "learning_rate": float(train.get("learning_rate", 1.0e-4)),
            "scale_lr": bool(train.get("scale_lr", False)),
            "lr_scheduler": str(train.get("lr_scheduler", "constant")),
            "lr_warmup_steps": int(train.get("lr_warmup_steps", 0)),
            "gradient_checkpointing": bool(train.get("gradient_checkpointing", False)),
            "mixed_precision": str(train.get("mixed_precision", "no")),
            "max_grad_norm": float(train.get("max_grad_norm", 1.0)),
            "allow_tf32": bool(train.get("allow_tf32", False)),
            "dataloader_num_workers": int(train.get("dataloader_num_workers", 0)),
            "checkpointing_steps": int(train.get("checkpointing_steps", max(int(train.get("max_train_steps", 1)), 1))),
            "checkpoints_total_limit": _optional_int(train.get("checkpoints_total_limit"), "checkpoints_total_limit"),
            "resume_from_checkpoint": _optional_string(train.get("resume_from_checkpoint"), "resume_from_checkpoint"),
            "optimizer": {
                "adam_beta1": float(optimizer.get("adam_beta1", 0.9)),
                "adam_beta2": float(optimizer.get("adam_beta2", 0.999)),
                "adam_weight_decay": float(optimizer.get("adam_weight_decay", 1.0e-2)),
                "adam_epsilon": float(optimizer.get("adam_epsilon", 1.0e-8)),
            },
        },
        "validation": {
            "num_validation_images": int(validation.get("num_validation_images", 4)),
            "validation_steps": int(validation.get("validation_steps", 400)),
            "num_inference_steps": int(validation.get("num_inference_steps", 50)),
            "seed": _optional_int(validation.get("seed"), "seed"),
        },
        "logging": {
            "report_to": str(logging.get("report_to", "swanlab")),
            "logging_dir": str(logging.get("logging_dir", "logs")),
            "project_name": _normalize_project_name(logging),
            "experiment_name": _optional_string(logging.get("experiment_name"), "experiment_name"),
        },
        "distributed": {
            "find_unused_parameters": bool(distributed.get("find_unused_parameters", False)),
        },
    }

    _validate_uncond_ldm_train_config(normalized)
    return normalized


def _validate_uncond_ldm_train_config(config: dict[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    train = config["train"]
    validation = config["validation"]

    if model["model_type"] != "uncond_ldm":
        raise ValueError("model.model_type must be 'uncond_ldm'.")

    if bool(data["train_manifest"]) == bool(data["train_image_dir"]):
        raise ValueError("Exactly one of data.train_manifest or data.train_image_dir must be provided.")

    if train["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("train.mixed_precision must be one of: no, fp16, bf16.")
    if config["logging"]["report_to"] not in {"none", "swanlab"}:
        raise ValueError("logging.report_to must be either 'none' or 'swanlab'.")
    if model["scheduler"]["prediction_type"] != "epsilon":
        raise ValueError("model.scheduler.prediction_type must be 'epsilon' in the first version.")

    if train["train_batch_size"] <= 0:
        raise ValueError("train.train_batch_size must be positive.")
    if train["gradient_accumulation_steps"] <= 0:
        raise ValueError("train.gradient_accumulation_steps must be positive.")
    if train["max_train_steps"] <= 0:
        raise ValueError("train.max_train_steps must be positive.")
    if train["num_train_epochs"] <= 0:
        raise ValueError("train.num_train_epochs must be positive.")
    if train["checkpointing_steps"] <= 0:
        raise ValueError("train.checkpointing_steps must be positive.")
    if data["resolution"] <= 0:
        raise ValueError("data.resolution must be positive.")

    unet = model["unet"]
    if unet["sample_size"] <= 0:
        raise ValueError("model.unet.sample_size must be positive.")
    if data["resolution"] != unet["sample_size"] * 8:
        raise ValueError("data.resolution must equal model.unet.sample_size * 8 for the fixed 8x VAE setup.")
    if len(unet["block_out_channels"]) != len(unet["down_block_types"]):
        raise ValueError("model.unet.block_out_channels and model.unet.down_block_types must have matching lengths.")
    if len(unet["block_out_channels"]) != len(unet["up_block_types"]):
        raise ValueError("model.unet.block_out_channels and model.unet.up_block_types must have matching lengths.")
    if unet["in_channels"] != 4 or unet["out_channels"] != 4:
        raise ValueError("model.unet.in_channels and model.unet.out_channels must both be 4.")

    if validation["num_validation_images"] < 0:
        raise ValueError("validation.num_validation_images must be non-negative.")
    if validation["validation_steps"] <= 0:
        raise ValueError("validation.validation_steps must be positive.")
    if validation["num_inference_steps"] <= 0:
        raise ValueError("validation.num_inference_steps must be positive.")


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


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer for '{key}'.") from exc


def _optional_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected float for '{key}'.") from exc


def _normalize_project_name(logging: dict[str, Any]) -> str:
    project_name = logging.get("project_name", "2d-gen-uncond-ldm")
    if not isinstance(project_name, str) or not project_name.strip():
        raise ValueError("Expected non-empty string for 'project_name'.")
    return project_name
