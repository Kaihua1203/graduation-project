from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

from common.types import Conditioning, TimeState
from train.adapters.base import BaseModelAdapter


class FluxAdapter(BaseModelAdapter):
    def setup(self, accelerator: Accelerator, weight_dtype: torch.dtype) -> None:
        raise NotImplementedError("Flux training adapter is the next milestone.")

    def get_models_for_accelerator_prepare(self) -> tuple[torch.nn.Module, ...]:
        raise NotImplementedError

    def set_prepared_models(self, prepared_models: tuple[torch.nn.Module, ...]) -> None:
        raise NotImplementedError

    def get_accumulate_target(self) -> torch.nn.Module:
        raise NotImplementedError

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        raise NotImplementedError

    def encode_text(
        self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype
    ) -> Conditioning:
        raise NotImplementedError

    def encode_latents(
        self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_time_state(
        self, latents: torch.Tensor, batch: dict[str, Any], device: torch.device
    ) -> TimeState:
        raise NotImplementedError

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

    def forward_model(
        self,
        model_input: torch.Tensor,
        time_state: TimeState,
        conditioning: Conditioning,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError

    def build_target(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        time_state: TimeState,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        raise NotImplementedError

    def build_validation_pipeline(self, accelerator: Accelerator):
        raise NotImplementedError

    def save_checkpoint(self, output_dir: str | Path, accelerator: Accelerator | None = None) -> None:
        raise NotImplementedError

    def load_checkpoint(self, input_dir: str | Path, accelerator: Accelerator) -> None:
        raise NotImplementedError

    def on_checkpoint_loaded(self, accelerator: Accelerator) -> None:
        return None

    def validate_conditioning(self, conditioning: Conditioning) -> None:
        super().validate_conditioning(conditioning)
        if conditioning.pooled_prompt_embeds is None:
            raise ValueError("Flux requires pooled_prompt_embeds.")
        if conditioning.text_ids is None:
            raise ValueError("Flux requires text_ids.")

    def validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 3:
            raise ValueError(f"Flux packed latents must be rank-3, got shape={tuple(latents.shape)}")
