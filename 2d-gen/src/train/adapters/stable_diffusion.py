from __future__ import annotations

from typing import Any

import torch

from common.diffusers_import import prepare_diffusers_import
from common.types import Conditioning, TimeState
from train.adapters.base import BaseModelAdapter

prepare_diffusers_import()

from peft import LoraConfig  # noqa: E402
from peft.utils import get_peft_model_state_dict  # noqa: E402
from diffusers import DDPMScheduler, StableDiffusionPipeline  # noqa: E402
from diffusers.utils import convert_state_dict_to_diffusers  # noqa: E402


class StableDiffusionAdapter(BaseModelAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.pipeline = None
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

    def setup(self, device: torch.device) -> None:
        model_cfg = self.config["model"]
        train_cfg = self.config["train"]
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_cfg["pretrained_path"],
            revision=model_cfg.get("revision"),
            local_files_only=model_cfg.get("local_files_only", True),
            safety_checker=None,
        )
        self.pipeline = self.pipeline.to(device)
        self.unet = self.pipeline.unet
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.tokenizer = self.pipeline.tokenizer
        self.scheduler = DDPMScheduler.from_config(self.pipeline.scheduler.config)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        lora_config = LoraConfig(
            r=train_cfg["lora_rank"],
            lora_alpha=train_cfg["lora_alpha"],
            lora_dropout=train_cfg["lora_dropout"],
            bias="none",
            target_modules=train_cfg["target_modules"],
        )
        self.unet.add_adapter(lora_config)

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        assert self.unet is not None
        params = [parameter for parameter in self.unet.parameters() if parameter.requires_grad]
        if not params:
            raise ValueError("No trainable LoRA parameters were found in the UNet.")
        return params

    def encode_text(
        self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype
    ) -> Conditioning:
        assert self.tokenizer is not None
        assert self.text_encoder is not None
        assert self.unet is not None
        prompts = batch["prompt"]
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        prompt_embeds = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
        )[0].to(device=device, dtype=self.unet.dtype)
        conditioning = Conditioning(prompt_embeds=prompt_embeds)
        self.validate_conditioning(conditioning)
        return conditioning

    def encode_latents(
        self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        assert self.vae is not None
        pixel_values = batch["pixel_values"].to(device=device, dtype=self.vae.dtype)
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        self.validate_latents(latents)
        return latents

    def sample_time_state(
        self, latents: torch.Tensor, batch: dict[str, Any], device: torch.device
    ) -> TimeState:
        assert self.scheduler is not None
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=device,
            dtype=torch.long,
        )
        return TimeState(timesteps=timesteps)

    def prepare_noisy_input(
        self,
        clean_latents: torch.Tensor,
        time_state: TimeState,
        noise: torch.Tensor,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert self.scheduler is not None
        assert time_state.timesteps is not None
        return self.scheduler.add_noise(clean_latents, noise, time_state.timesteps)

    def forward_model(
        self,
        model_input: torch.Tensor,
        time_state: TimeState,
        conditioning: Conditioning,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        assert self.unet is not None
        assert time_state.timesteps is not None
        return self.unet(
            model_input,
            time_state.timesteps,
            encoder_hidden_states=conditioning.prompt_embeds,
        ).sample

    def build_target(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        time_state: TimeState,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        return noise

    def save_checkpoint(self, output_dir: str) -> None:
        assert self.unet is not None
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(self.unet)
        )
        StableDiffusionPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )
