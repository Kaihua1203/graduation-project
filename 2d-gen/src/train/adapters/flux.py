from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from common.diffusers_import import prepare_diffusers_import
from common.types import Conditioning, TimeState
from train.adapters.base import BaseModelAdapter

prepare_diffusers_import()

from diffusers import FlowMatchEulerDiscreteScheduler, FluxPipeline  # noqa: E402
from diffusers.training_utils import (  # noqa: E402
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_unet_state_dict_to_peft  # noqa: E402
from peft import LoraConfig  # noqa: E402
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


class FluxAdapter(BaseModelAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.pipeline = None
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.text_encoder_2 = None
        self.tokenizer = None
        self.tokenizer_2 = None
        self.scheduler = None
        self.weight_dtype = torch.float32
        self.vae_scale_factor = 8
        self.guidance_scale = 3.5
        self.weighting_scheme = "none"
        self._checkpoint_hooks_registered = False

    def setup(self, accelerator: Any, weight_dtype: torch.dtype) -> None:
        model_cfg = self.config["model"]
        train_cfg = self.config["train"]
        pipeline = FluxPipeline.from_pretrained(
            model_cfg["pretrained_model_name_or_path"],
            revision=model_cfg.get("revision"),
            variant=model_cfg.get("variant"),
            local_files_only=model_cfg.get("local_files_only", True),
            torch_dtype=weight_dtype,
        )

        self.pipeline = pipeline
        self.transformer = pipeline.transformer
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.text_encoder_2 = pipeline.text_encoder_2
        self.tokenizer = pipeline.tokenizer
        self.tokenizer_2 = pipeline.tokenizer_2
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        self.weight_dtype = weight_dtype
        self.vae_scale_factor = pipeline.vae_scale_factor
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)

        lora_config = LoraConfig(
            r=train_cfg["lora"]["rank"],
            lora_alpha=train_cfg["lora"]["alpha"],
            lora_dropout=train_cfg["lora"]["dropout"],
            init_lora_weights="gaussian",
            target_modules=train_cfg["lora"]["target_modules"],
        )
        self.transformer.add_adapter(lora_config)

        if train_cfg.get("gradient_checkpointing", False) and hasattr(self.transformer, "enable_gradient_checkpointing"):
            self.transformer.enable_gradient_checkpointing()

        self.transformer.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        if self.text_encoder is not None:
            self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.to(accelerator.device, dtype=weight_dtype)
        if accelerator.mixed_precision == "fp16":
            cast_training_params([self.transformer], dtype=torch.float32)
        self.transformer.train()
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.text_encoder_2 is not None:
            self.text_encoder_2.eval()

    def get_models_for_accelerator_prepare(self) -> tuple[torch.nn.Module, ...]:
        assert self.transformer is not None
        return (self.transformer,)

    def set_prepared_models(self, prepared_models: tuple[torch.nn.Module, ...]) -> None:
        (self.transformer,) = prepared_models

    def get_accumulate_target(self) -> torch.nn.Module:
        assert self.transformer is not None
        return self.transformer

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        assert self.transformer is not None
        params = [parameter for parameter in self.transformer.parameters() if parameter.requires_grad]
        if not params:
            raise ValueError("No trainable LoRA parameters were found in the Flux transformer.")
        return params

    def encode_text(
        self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype
    ) -> Conditioning:
        prompts = batch["prompt"]
        prompt_2 = batch.get("prompt_2") or prompts
        assert self.pipeline is not None
        prompt_embeds, pooled_prompt_embeds, text_ids = self.pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=1,
        )
        conditioning = Conditioning(
            prompt_embeds=prompt_embeds.to(device=device, dtype=dtype),
            pooled_prompt_embeds=pooled_prompt_embeds.to(device=device, dtype=dtype),
            text_ids=text_ids.to(device=device, dtype=dtype),
        )
        self.validate_conditioning(conditioning)
        return conditioning

    def encode_latents(
        self, batch: dict[str, Any], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        assert self.vae is not None
        pixel_values = batch["pixel_values"].to(device=device, dtype=self.vae.dtype)
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        packed_latents = FluxPipeline._pack_latents(
            latents,
            batch_size=latents.shape[0],
            num_channels_latents=latents.shape[1],
            height=latents.shape[-2],
            width=latents.shape[-1],
        )
        self.validate_latents(packed_latents)
        return packed_latents.to(device=device, dtype=dtype)

    def sample_time_state(
        self, latents: torch.Tensor, batch: dict[str, Any], device: torch.device
    ) -> TimeState:
        assert self.scheduler is not None
        timestep_weights = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=latents.shape[0],
            device=device,
        )
        timestep_indices = (timestep_weights * self.scheduler.config.num_train_timesteps).long()
        scheduler_timesteps = self.scheduler.timesteps.to(device=device)
        timesteps = scheduler_timesteps[timestep_indices]
        sigmas = self._get_sigmas(timesteps, latents.ndim, latents.dtype)

        guidance = None
        if getattr(getattr(self.transformer, "config", None), "guidance_embeds", False):
            guidance = torch.full(
                (latents.shape[0],),
                self.guidance_scale,
                device=device,
                dtype=torch.float32,
            )

        latent_height = 2 * (int(batch["pixel_values"].shape[-2]) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(batch["pixel_values"].shape[-1]) // (self.vae_scale_factor * 2))
        img_ids = FluxPipeline._prepare_latent_image_ids(
            latents.shape[0],
            latent_height // 2,
            latent_width // 2,
            device,
            latents.dtype,
        )
        return TimeState(
            timesteps=timesteps,
            sigmas=sigmas,
            guidance=guidance,
            extra={"img_ids": img_ids},
        )

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
        return self.scheduler.scale_noise(clean_latents, time_state.timesteps, noise)

    def forward_model(
        self,
        model_input: torch.Tensor,
        time_state: TimeState,
        conditioning: Conditioning,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        assert self.transformer is not None
        assert time_state.timesteps is not None
        img_ids = time_state.extra["img_ids"]
        return self.transformer(
            hidden_states=model_input,
            timestep=time_state.timesteps / 1000,
            guidance=time_state.guidance,
            pooled_projections=conditioning.pooled_prompt_embeds,
            encoder_hidden_states=conditioning.prompt_embeds,
            txt_ids=conditioning.text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    def build_target(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        time_state: TimeState,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        return noise - clean_latents

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        time_state: TimeState,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=time_state.sigmas)
        loss = (weighting.float() * (prediction.float() - target.float()) ** 2).reshape(target.shape[0], -1)
        return loss.mean(dim=1).mean()

    def build_validation_pipeline(self, accelerator: Any):
        transformer = self._unwrap_transformer(accelerator)
        pipeline = FluxPipeline.from_pretrained(
            self.config["model"]["pretrained_model_name_or_path"],
            revision=self.config["model"].get("revision"),
            variant=self.config["model"].get("variant"),
            local_files_only=self.config["model"].get("local_files_only", True),
            transformer=transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            torch_dtype=self.weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def save_checkpoint(self, output_dir: str | Path, accelerator: Any | None = None) -> None:
        transformer = self._unwrap_transformer(accelerator)
        FluxPipeline.save_lora_weights(
            save_directory=str(output_dir),
            transformer_lora_layers=get_peft_model_state_dict(transformer),
            safe_serialization=True,
        )

    def load_checkpoint(self, input_dir: str | Path, accelerator: Any | None = None) -> None:
        transformer = self._unwrap_transformer(accelerator)
        lora_state_dict = FluxPipeline.lora_state_dict(str(input_dir))
        transformer_state_dict = {
            key.replace("transformer.", ""): value
            for key, value in lora_state_dict.items()
            if key.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
        if accelerator is not None and accelerator.mixed_precision == "fp16":
            cast_training_params([transformer], dtype=torch.float32)

    def register_checkpointing_hooks(self, accelerator: Any) -> None:
        if self._checkpoint_hooks_registered:
            return

        def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
            return accelerator.unwrap_model(model)

        def save_model_hook(models: list[torch.nn.Module], weights: list[torch.Tensor], output_dir: str) -> None:
            if not accelerator.is_main_process:
                while weights:
                    weights.pop()
                return

            transformer_lora_layers_to_save = None
            for model in models:
                if not isinstance(model, type(unwrap_model(self.transformer))):
                    raise ValueError(f"Unexpected save model: {model.__class__}")
                transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                if weights:
                    weights.pop()

            FluxPipeline.save_lora_weights(
                save_directory=output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                safe_serialization=True,
            )

        def load_model_hook(models: list[torch.nn.Module], input_dir: str) -> None:
            transformer = None
            while models:
                model = models.pop()
                if not isinstance(model, type(unwrap_model(self.transformer))):
                    raise ValueError(f"Unexpected load model: {model.__class__}")
                transformer = model

            lora_state_dict = FluxPipeline.lora_state_dict(input_dir)
            transformer_state_dict = {
                key.replace("transformer.", ""): value
                for key, value in lora_state_dict.items()
                if key.startswith("transformer.")
            }
            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
            if accelerator.mixed_precision == "fp16":
                cast_training_params([transformer], dtype=torch.float32)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        self._checkpoint_hooks_registered = True

    def validate_conditioning(self, conditioning: Conditioning) -> None:
        super().validate_conditioning(conditioning)
        if conditioning.pooled_prompt_embeds is None:
            raise ValueError("Flux requires pooled_prompt_embeds.")
        if conditioning.pooled_prompt_embeds.ndim != 2:
            raise ValueError(
                f"Flux pooled_prompt_embeds must be rank-2, got shape={tuple(conditioning.pooled_prompt_embeds.shape)}"
            )
        if conditioning.text_ids is None:
            raise ValueError("Flux requires text_ids.")
        if conditioning.text_ids.ndim not in {2, 3}:
            raise ValueError(f"Flux text_ids must be rank-2 or rank-3, got shape={tuple(conditioning.text_ids.shape)}")

    def validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 3:
            raise ValueError(f"Flux packed latents must be rank-3, got shape={tuple(latents.shape)}")

    def _get_sigmas(self, timesteps: torch.Tensor, ndim: int, dtype: torch.dtype) -> torch.Tensor:
        assert self.scheduler is not None
        schedule_timesteps = self.scheduler.timesteps.to(device=timesteps.device)
        sigmas = self.scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _unwrap_transformer(self, accelerator: Any | None) -> torch.nn.Module:
        assert self.transformer is not None
        if accelerator is None:
            return self.transformer
        return accelerator.unwrap_model(self.transformer)
