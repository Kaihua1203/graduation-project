from __future__ import annotations

from copy import deepcopy
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


def _coalesce(section: Dict[str, Any], *keys: str, default: Any = None, required: bool = False) -> Any:
    for key in keys:
        if key in section and section[key] is not None:
            return section[key]
    if required:
        joined = ", ".join(keys)
        raise KeyError(f"Missing required key. Expected one of: {joined}")
    return default


def normalize_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(config)

    model = ensure_section(normalized, "model")
    data = ensure_section(normalized, "data")
    train = ensure_section(normalized, "train")
    validation = normalized.setdefault("validation", {})
    logging = normalized.setdefault("logging", {})
    distributed = normalized.setdefault("distributed", {})

    pretrained_model_name_or_path = _coalesce(
        model,
        "pretrained_model_name_or_path",
        "pretrained_path",
        required=True,
    )
    model["pretrained_model_name_or_path"] = pretrained_model_name_or_path
    model["pretrained_path"] = pretrained_model_name_or_path
    model.setdefault("pretrained_vae_model_name_or_path", None)
    model.setdefault("revision", None)
    model.setdefault("variant", None)
    model.setdefault("local_files_only", True)

    resolution = _coalesce(data, "resolution", "image_size", default=512)
    data["resolution"] = resolution
    data["image_size"] = resolution
    data.setdefault("image_column", "image_path")
    data.setdefault("caption_column", "prompt")
    data.setdefault("center_crop", False)
    data.setdefault("random_flip", False)
    data.setdefault("image_interpolation_mode", "bilinear")
    data.setdefault("max_train_samples", None)

    output_dir = _coalesce(train, "output_dir", required=True)
    train_batch_size = _coalesce(train, "train_batch_size", "batch_size", required=True)
    num_train_epochs = _coalesce(train, "num_train_epochs", "num_epochs", required=True)
    learning_rate = _coalesce(train, "learning_rate", required=True)
    max_train_steps = _coalesce(train, "max_train_steps", required=True)
    mixed_precision = _coalesce(train, "mixed_precision", required=True)
    seed = _coalesce(train, "seed", required=True)
    dataloader_num_workers = _coalesce(
        train,
        "dataloader_num_workers",
        "num_workers",
        default=0,
    )
    checkpointing_steps = _coalesce(
        train,
        "checkpointing_steps",
        "save_every_n_steps",
        default=train.get("max_train_steps"),
    )

    train["train_batch_size"] = train_batch_size
    train["batch_size"] = train_batch_size
    train["num_train_epochs"] = num_train_epochs
    train["num_epochs"] = num_train_epochs
    train["output_dir"] = output_dir
    train["learning_rate"] = learning_rate
    train["max_train_steps"] = max_train_steps
    train["mixed_precision"] = mixed_precision
    train["seed"] = seed
    train["image_size"] = resolution
    train["dataloader_num_workers"] = dataloader_num_workers
    train["num_workers"] = dataloader_num_workers
    train["checkpointing_steps"] = checkpointing_steps
    train["save_every_n_steps"] = checkpointing_steps
    train.setdefault("scale_lr", False)
    train.setdefault("lr_scheduler", "constant")
    train.setdefault("lr_warmup_steps", 0)
    train.setdefault("gradient_checkpointing", False)
    train.setdefault("snr_gamma", None)
    train.setdefault("prediction_type", None)
    train.setdefault("noise_offset", 0.0)
    train.setdefault("max_grad_norm", 1.0)
    train.setdefault("allow_tf32", False)
    train.setdefault("checkpoints_total_limit", None)
    train.setdefault("resume_from_checkpoint", None)
    train.setdefault("enable_xformers_memory_efficient_attention", False)
    train.setdefault("optimizer", {})
    train.setdefault("lora", {})
    train.setdefault("sdxl", {})

    lora = ensure_section(train, "lora")
    lora_rank = _coalesce(lora, "rank", "lora_rank", required=True)
    lora_alpha = _coalesce(lora, "alpha", "lora_alpha", required=True)
    lora_dropout = _coalesce(lora, "dropout", "lora_dropout", default=0.0)
    target_modules = _coalesce(lora, "target_modules", required=True)
    lora["rank"] = lora_rank
    lora["alpha"] = lora_alpha
    lora["dropout"] = lora_dropout
    lora["target_modules"] = target_modules
    train["lora_rank"] = lora_rank
    train["lora_alpha"] = lora_alpha
    train["lora_dropout"] = lora_dropout
    train["target_modules"] = target_modules

    optimizer = ensure_section(train, "optimizer")
    optimizer.setdefault("use_8bit_adam", False)
    optimizer.setdefault("adam_beta1", 0.9)
    optimizer.setdefault("adam_beta2", 0.999)
    optimizer.setdefault("adam_weight_decay", 1.0e-2)
    optimizer.setdefault("adam_epsilon", 1.0e-8)

    sdxl = ensure_section(train, "sdxl")
    sdxl.setdefault("train_text_encoder", False)
    sdxl.setdefault("enable_npu_flash_attention", False)
    sdxl.setdefault("debug_loss", False)

    validation.setdefault("validation_prompt", None)
    validation.setdefault("num_validation_images", 4)
    validation.setdefault("validation_epochs", 1)

    logging.setdefault("report_to", "swanlab")
    logging.setdefault("logging_dir", "logs")
    logging.setdefault("log_every_n_steps", 5)
    logging.setdefault("tracker_project_name", "2d-gen-train")

    distributed.setdefault("ddp_backend", "nccl")
    distributed.setdefault("find_unused_parameters", False)
    distributed.setdefault("local_rank", -1)

    train["log_every_n_steps"] = logging["log_every_n_steps"]
    train["report_to"] = logging["report_to"]
    train["logging_dir"] = logging["logging_dir"]
    train["tracker_project_name"] = logging["tracker_project_name"]
    train["validation_prompt"] = validation["validation_prompt"]
    train["num_validation_images"] = validation["num_validation_images"]
    train["validation_epochs"] = validation["validation_epochs"]

    return normalized
