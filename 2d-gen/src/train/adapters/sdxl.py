from __future__ import annotations

from typing import Any

import torch

from common.types import Conditioning, TimeState
from train.adapters.base import BaseModelAdapter


class SDXLAdapter(BaseModelAdapter):
    def setup(self, device: torch.device) -> None:
        raise NotImplementedError("SDXL training adapter is the next milestone.")

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

    def save_checkpoint(self, output_dir: str) -> None:
        raise NotImplementedError

    def validate_conditioning(self, conditioning: Conditioning) -> None:
        super().validate_conditioning(conditioning)
        if conditioning.pooled_prompt_embeds is None:
            raise ValueError("SDXL requires pooled_prompt_embeds.")
        if conditioning.add_time_ids is None:
            raise ValueError("SDXL requires add_time_ids.")
