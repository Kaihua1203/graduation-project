from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from common.types import Conditioning, TimeState


class BaseModelAdapter(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def setup(self, device: torch.device) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> Conditioning:
        raise NotImplementedError

    @abstractmethod
    def encode_latents(self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError

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
        batch: dict[str, Any],
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(prediction.float(), target.float())

    @abstractmethod
    def save_checkpoint(self, output_dir: str) -> None:
        raise NotImplementedError

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
