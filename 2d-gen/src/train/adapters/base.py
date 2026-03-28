from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

import torch
from accelerate import Accelerator

from common.types import Conditioning, TimeState


class BaseModelAdapter(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def setup(self, accelerator: Accelerator, weight_dtype: torch.dtype) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_models_for_accelerator_prepare(self) -> tuple[torch.nn.Module, ...]:
        raise NotImplementedError

    def set_prepared_models(self, prepared_models: tuple[torch.nn.Module, ...]) -> None:
        if len(prepared_models) == 1 and hasattr(self, "unet"):
            setattr(self, "unet", prepared_models[0])
        return None

    def get_accumulate_target(self) -> torch.nn.Module:
        prepared_models = self.get_models_for_accelerator_prepare()
        if not prepared_models:
            raise NotImplementedError("Adapter must expose at least one model to accumulate.")
        return prepared_models[0]

    @abstractmethod
    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError

    def get_grad_clip_parameters(self) -> Iterable[torch.nn.Parameter]:
        return self.get_trainable_parameters()

    @abstractmethod
    def encode_text(self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> Conditioning:
        raise NotImplementedError

    @abstractmethod
    def encode_latents(self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError

    def sample_noise(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(latents)

    @abstractmethod
    def sample_time_state(self, latents: torch.Tensor, batch: dict[str, Any], device: torch.device) -> TimeState:
        raise NotImplementedError

    @abstractmethod
    def prepare_noisy_input(
        self,
        clean_latents: torch.Tensor,
        time_state: TimeState,
        noise: torch.Tensor,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_model(
        self,
        model_input: torch.Tensor,
        time_state: TimeState,
        conditioning: Conditioning,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def build_target(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        time_state: TimeState,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        time_state: TimeState,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(prediction.float(), target.float())

    def build_validation_pipeline(self, accelerator: Accelerator) -> Any:
        raise NotImplementedError

    def save_checkpoint(self, output_dir: str | Path, accelerator: Accelerator | None = None) -> None:
        raise NotImplementedError

    def load_checkpoint(self, input_dir: str | Path, accelerator: Accelerator) -> None:
        raise NotImplementedError

    def register_checkpointing_hooks(self, accelerator: Accelerator) -> None:
        return None

    def on_checkpoint_loaded(self, accelerator: Accelerator) -> None:
        return None

    def get_validation_inference_steps(self) -> int:
        return 30

    def validate_conditioning(self, conditioning: Conditioning) -> None:
        if conditioning.prompt_embeds is None:
            raise ValueError("prompt_embeds must not be None.")
        if conditioning.prompt_embeds.ndim != 3:
            raise ValueError(
                f"prompt_embeds must be rank-3, got shape={tuple(conditioning.prompt_embeds.shape)}"
            )

    def validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 4:
            raise ValueError(f"Latents must be rank-4, got shape={tuple(latents.shape)}")
