from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.diffusers_import import prepare_diffusers_import


UNET_DIRNAME = "unet"
SCHEDULER_DIRNAME = "scheduler"
SCHEDULER_CONFIG_FILENAME = "scheduler_config.json"
METADATA_FILENAMES = ("model_metadata.json", "metadata.json")
TRAINING_SUMMARY_FILENAMES = ("training_summary.json", "train_summary.json")


@dataclass(frozen=True)
class ExportBundlePaths:
    bundle_dir: Path
    unet_dir: Path
    scheduler_dir: Path
    metadata_path: Path
    training_summary_path: Path
    metadata: dict[str, Any]
    training_summary: dict[str, Any]


def load_export_bundle_paths(path: str | Path) -> ExportBundlePaths:
    bundle_dir = Path(path).expanduser().resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Export bundle directory does not exist: {bundle_dir}")

    unet_dir = bundle_dir / UNET_DIRNAME
    scheduler_dir = bundle_dir / SCHEDULER_DIRNAME
    metadata_path = _resolve_required_file(bundle_dir, METADATA_FILENAMES)
    training_summary_path = _resolve_required_file(bundle_dir, TRAINING_SUMMARY_FILENAMES)

    missing_paths: list[str] = []
    if not unet_dir.is_dir():
        missing_paths.append(str(unet_dir))
    if not scheduler_dir.is_dir():
        missing_paths.append(str(scheduler_dir))
    if not (scheduler_dir / SCHEDULER_CONFIG_FILENAME).is_file():
        missing_paths.append(str(scheduler_dir / SCHEDULER_CONFIG_FILENAME))
    if metadata_path is None:
        missing_paths.extend(str(bundle_dir / filename) for filename in METADATA_FILENAMES)
    if training_summary_path is None:
        missing_paths.extend(str(bundle_dir / filename) for filename in TRAINING_SUMMARY_FILENAMES)
    if missing_paths:
        raise FileNotFoundError(
            "Export bundle is missing required artifacts:\n- " + "\n- ".join(missing_paths)
        )

    metadata = _load_json_object(metadata_path, "model metadata")
    training_summary = _load_json_object(training_summary_path, "training summary")
    return ExportBundlePaths(
        bundle_dir=bundle_dir,
        unet_dir=unet_dir,
        scheduler_dir=scheduler_dir,
        metadata_path=metadata_path,
        training_summary_path=training_summary_path,
        metadata=metadata,
        training_summary=training_summary,
    )


def load_inference_components(
    bundle: ExportBundlePaths,
    *,
    vae_path: str | None = None,
    scheduler_type: str | None = None,
    local_files_only: bool = True,
) -> tuple[Any, Any, Any]:
    prepare_diffusers_import()
    from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DModel

    scheduler_cls = _resolve_scheduler_class(
        scheduler_type=scheduler_type,
        exported_class_name=_read_exported_scheduler_class_name(bundle.scheduler_dir),
        ddim_cls=DDIMScheduler,
        ddpm_cls=DDPMScheduler,
    )
    resolved_vae_path, resolved_vae_subfolder = _resolve_vae_source(bundle.metadata, vae_path)

    unet = UNet2DModel.from_pretrained(bundle.unet_dir, local_files_only=local_files_only)
    scheduler = scheduler_cls.from_pretrained(bundle.scheduler_dir, local_files_only=local_files_only)
    vae_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if resolved_vae_subfolder is not None:
        vae_kwargs["subfolder"] = resolved_vae_subfolder
    vae = AutoencoderKL.from_pretrained(resolved_vae_path, **vae_kwargs)
    return unet, scheduler, vae


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{label.capitalize()} at {path} must be a JSON object.")
    return data


def _read_exported_scheduler_class_name(scheduler_dir: Path) -> str | None:
    config_path = scheduler_dir / SCHEDULER_CONFIG_FILENAME
    config = _load_json_object(config_path, "scheduler config")
    class_name = config.get("_class_name")
    if class_name is None:
        return None
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError(f"Scheduler config at {config_path} has an invalid _class_name.")
    return class_name


def _resolve_scheduler_class(
    *,
    scheduler_type: str | None,
    exported_class_name: str | None,
    ddim_cls: type[Any],
    ddpm_cls: type[Any],
) -> type[Any]:
    resolved_name = scheduler_type or exported_class_name or "DDPMScheduler"
    normalized_name = str(resolved_name).strip().lower()
    if normalized_name in {"ddim", "ddimscheduler"}:
        return ddim_cls
    if normalized_name in {"ddpm", "ddpmscheduler"}:
        return ddpm_cls
    raise ValueError(f"Unsupported scheduler type: {resolved_name}")


def _resolve_required_file(bundle_dir: Path, candidate_filenames: tuple[str, ...]) -> Path | None:
    for filename in candidate_filenames:
        candidate_path = bundle_dir / filename
        if candidate_path.is_file():
            return candidate_path
    return None


def _resolve_vae_source(metadata: dict[str, Any], override_path: str | None) -> tuple[str, str | None]:
    if override_path is not None:
        return str(Path(override_path).expanduser().resolve()), None

    vae_source = metadata.get("vae")
    if not isinstance(vae_source, dict):
        raise ValueError("Export metadata must include a 'vae' mapping or infer.model.vae_path override.")

    pretrained_path = vae_source.get("pretrained_model_name_or_path") or vae_source.get("path")
    if not isinstance(pretrained_path, str) or not pretrained_path.strip():
        raise ValueError(
            "Export metadata 'vae' mapping must contain 'pretrained_model_name_or_path' or 'path'."
        )
    subfolder = vae_source.get("subfolder")
    if subfolder is not None and (not isinstance(subfolder, str) or not subfolder.strip()):
        raise ValueError("Export metadata 'vae.subfolder' must be a non-empty string when provided.")
    return str(Path(pretrained_path).expanduser().resolve()), subfolder
