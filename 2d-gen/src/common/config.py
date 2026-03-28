from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must decode to a mapping.")
    return data


def ensure_section(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing mapping section: {key}")
    return value


def load_train_config(path: str | Path) -> Dict[str, Any]:
    config = load_yaml_config(path)
    return normalize_train_config(config)


def normalize_train_config(config: Dict[str, Any]) -> Dict[str, Any]:
    _reject_legacy_train_schema(config)

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

    normalized = {
        "model": {
            "family": _require_string(model, "family"),
            "pretrained_model_name_or_path": _require_string(model, "pretrained_model_name_or_path"),
            "pretrained_vae_model_name_or_path": model.get("pretrained_vae_model_name_or_path"),
            "revision": model.get("revision"),
            "variant": model.get("variant"),
            "local_files_only": bool(model.get("local_files_only", True)),
        },
        "data": {
            "train_manifest": _require_string(data, "train_manifest"),
            "image_column": str(data.get("image_column", "image_path")),
            "caption_column": str(data.get("caption_column", "prompt")),
            "resolution": int(data.get("resolution", 512)),
            "center_crop": bool(data.get("center_crop", False)),
            "random_flip": bool(data.get("random_flip", False)),
            "image_interpolation_mode": str(data.get("image_interpolation_mode", "bilinear")),
            "max_train_samples": data.get("max_train_samples"),
        },
        "train": {
            "output_dir": _require_string(train, "output_dir"),
            "seed": int(train.get("seed", 3407)),
            "train_batch_size": int(train.get("train_batch_size", 1)),
            "num_train_epochs": int(train.get("num_train_epochs", 1)),
            "max_train_steps": int(train.get("max_train_steps", 0)),
            "gradient_accumulation_steps": int(train.get("gradient_accumulation_steps", 1)),
            "learning_rate": float(train.get("learning_rate", 1.0e-4)),
            "scale_lr": bool(train.get("scale_lr", False)),
            "lr_scheduler": str(train.get("lr_scheduler", "constant")),
            "lr_warmup_steps": int(train.get("lr_warmup_steps", 0)),
            "gradient_checkpointing": bool(train.get("gradient_checkpointing", False)),
            "mixed_precision": str(train.get("mixed_precision", "no")),
            "snr_gamma": train.get("snr_gamma"),
            "prediction_type": train.get("prediction_type"),
            "noise_offset": float(train.get("noise_offset", 0.0)),
            "max_grad_norm": float(train.get("max_grad_norm", 1.0)),
            "allow_tf32": bool(train.get("allow_tf32", False)),
            "dataloader_num_workers": int(train.get("dataloader_num_workers", 0)),
            "checkpointing_steps": int(train.get("checkpointing_steps", max(int(train.get("max_train_steps", 1)), 1))),
            "checkpoints_total_limit": train.get("checkpoints_total_limit"),
            "resume_from_checkpoint": train.get("resume_from_checkpoint"),
            "enable_xformers_memory_efficient_attention": bool(
                train.get("enable_xformers_memory_efficient_attention", False)
            ),
            "optimizer": _normalize_optimizer_config(train.get("optimizer")),
            "lora": _normalize_lora_config(train.get("lora")),
            "sdxl": _normalize_sdxl_config(train.get("sdxl")),
        },
        "validation": {
            "validation_prompt": validation.get("validation_prompt"),
            "num_validation_images": int(validation.get("num_validation_images", 4)),
            "validation_epochs": int(validation.get("validation_epochs", 1)),
        },
        "logging": {
            "report_to": str(logging.get("report_to", "swanlab")),
            "logging_dir": str(logging.get("logging_dir", "logs")),
            "log_every_n_steps": int(logging.get("log_every_n_steps", 10)),
            "tracker_project_name": str(logging.get("tracker_project_name", "2d-gen-train")),
        },
        "distributed": {
            "ddp_backend": str(distributed.get("ddp_backend", "nccl")),
            "find_unused_parameters": bool(distributed.get("find_unused_parameters", False)),
            "local_rank": int(distributed.get("local_rank", -1)),
        },
    }

    _validate_train_config(normalized)
    return normalized


def _normalize_optimizer_config(value: Any) -> Dict[str, Any]:
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


def _normalize_lora_config(value: Any) -> Dict[str, Any]:
    config = value or {}
    if not isinstance(config, dict):
        raise ValueError("train.lora must be a mapping.")
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        raise ValueError("train.lora.target_modules must be a non-empty list.")
    return {
        "rank": int(config.get("rank", 16)),
        "alpha": int(config.get("alpha", 16)),
        "dropout": float(config.get("dropout", 0.0)),
        "target_modules": [str(module_name) for module_name in target_modules],
    }


def _normalize_sdxl_config(value: Any) -> Dict[str, Any]:
    config = value or {}
    if not isinstance(config, dict):
        raise ValueError("train.sdxl must be a mapping.")
    return {
        "train_text_encoder": bool(config.get("train_text_encoder", False)),
        "enable_npu_flash_attention": bool(config.get("enable_npu_flash_attention", False)),
        "debug_loss": bool(config.get("debug_loss", False)),
    }


def _reject_legacy_train_schema(config: Dict[str, Any]) -> None:
    legacy_model_keys = {"pretrained_path"}
    legacy_train_keys = {
        "batch_size",
        "num_epochs",
        "image_size",
        "num_workers",
        "save_every_n_steps",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
    }
    model = config.get("model")
    train = config.get("train")
    if isinstance(model, dict) and legacy_model_keys.intersection(model.keys()):
        raise ValueError("Legacy training config schema is no longer supported. Please migrate to the new schema.")
    if isinstance(train, dict) and legacy_train_keys.intersection(train.keys()):
        raise ValueError("Legacy training config schema is no longer supported. Please migrate to the new schema.")


def _require_string(section: Dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for '{key}'.")
    return value


def _validate_train_config(config: Dict[str, Any]) -> None:
    mixed_precision = config["train"]["mixed_precision"]
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("train.mixed_precision must be one of: no, fp16, bf16.")

    report_to = config["logging"]["report_to"]
    if report_to not in {"none", "swanlab"}:
        raise ValueError("logging.report_to must be either 'none' or 'swanlab'.")

    if config["train"]["train_batch_size"] <= 0:
        raise ValueError("train.train_batch_size must be positive.")
    if config["train"]["gradient_accumulation_steps"] <= 0:
        raise ValueError("train.gradient_accumulation_steps must be positive.")
    if config["train"]["max_train_steps"] <= 0:
        raise ValueError("train.max_train_steps must be positive.")
    if config["train"]["num_train_epochs"] <= 0:
        raise ValueError("train.num_train_epochs must be positive.")
    if config["validation"]["num_validation_images"] <= 0:
        raise ValueError("validation.num_validation_images must be positive.")
    if config["validation"]["validation_epochs"] <= 0:
        raise ValueError("validation.validation_epochs must be positive.")
    if config["distributed"]["ddp_backend"] != "nccl":
        raise ValueError("distributed.ddp_backend is currently fixed to 'nccl'.")
    if config["train"]["optimizer"]["use_8bit_adam"]:
        raise ValueError("train.optimizer.use_8bit_adam is not implemented in the current trainer.")


def normalize_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_train_config(config)


def validate_train_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_train_config(config)


normalize_training_config = normalize_train_config
