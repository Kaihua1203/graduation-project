from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

from common.diffusers_import import prepare_diffusers_import
from common.types import Conditioning, TimeState
from train.adapters.base import BaseModelAdapter

prepare_diffusers_import()

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline  # noqa: E402
from diffusers.training_utils import cast_training_params, compute_snr  # noqa: E402
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft  # noqa: E402
from peft import LoraConfig  # noqa: E402
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402


class StableDiffusionAdapter(BaseModelAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.weight_dtype = torch.float32
        self._checkpoint_hooks_registered = False

    def setup(self, accelerator: Any, weight_dtype: torch.dtype) -> None:
        model_cfg = self.config["model"]
        train_cfg = self.config["train"]
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_cfg["pretrained_model_name_or_path"],
            revision=model_cfg.get("revision"),
            variant=model_cfg.get("variant"),
            local_files_only=model_cfg.get("local_files_only", True),
            torch_dtype=weight_dtype,
            safety_checker=None,
        )

        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        self.weight_dtype = weight_dtype
        del pipeline

        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        pretrained_vae_model_name_or_path = model_cfg.get("pretrained_vae_model_name_or_path")
        if pretrained_vae_model_name_or_path:
            self.vae = AutoencoderKL.from_pretrained(
                pretrained_vae_model_name_or_path,
                local_files_only=model_cfg.get("local_files_only", True),
                torch_dtype=weight_dtype,
            )
            self.vae.requires_grad_(False)

        lora_config = LoraConfig(
            r=train_cfg["lora"]["rank"],
            lora_alpha=train_cfg["lora"]["alpha"],
            lora_dropout=train_cfg["lora"]["dropout"],
            bias="none",
            target_modules=train_cfg["lora"]["target_modules"],
        )
        self.unet.add_adapter(lora_config)

        if train_cfg.get("gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()
        if train_cfg.get("enable_xformers_memory_efficient_attention", False) and hasattr(
            self.unet, "enable_xformers_memory_efficient_attention"
        ):
            self.unet.enable_xformers_memory_efficient_attention()

        self.unet.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        if accelerator.mixed_precision == "fp16":
            cast_training_params(self.unet, dtype=torch.float32)
        self.unet.train()
        self.vae.eval()
        self.text_encoder.eval()

        prediction_type = train_cfg.get("prediction_type")
        if prediction_type is not None:
            self.scheduler.register_to_config(prediction_type=prediction_type)

    def get_models_for_accelerator_prepare(self) -> tuple[torch.nn.Module, ...]:
        assert self.unet is not None
        return (self.unet,)

    def set_prepared_models(self, prepared_models: tuple[torch.nn.Module, ...]) -> None:
        (self.unet,) = prepared_models

    def get_accumulate_target(self) -> torch.nn.Module:
        assert self.unet is not None
        return self.unet

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
        )[0].to(device=device, dtype=dtype)
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

    def sample_noise(self, latents: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(latents)
        noise_offset = self.config["train"].get("noise_offset", 0.0)
        if noise_offset:
            noise = noise + noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                device=latents.device,
                dtype=latents.dtype,
            )
        return noise

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
        assert self.scheduler is not None
        assert time_state.timesteps is not None
        if self.scheduler.config.prediction_type == "v_prediction":
            return self.scheduler.get_velocity(clean_latents, noise, time_state.timesteps)
        return noise

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        time_state: TimeState,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        assert self.scheduler is not None
        assert time_state.timesteps is not None
        snr_gamma = self.config["train"].get("snr_gamma")
        if snr_gamma is None:
            return torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="mean")

        snr = compute_snr(self.scheduler, time_state.timesteps)
        mse_loss_weights = torch.stack(
            [snr, snr_gamma * torch.ones_like(time_state.timesteps)],
            dim=1,
        ).min(dim=1)[0]
        if self.scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif self.scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        return loss.mean()

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

            unet_lora_layers_to_save = None
            for model in models:
                if not isinstance(model, type(unwrap_model(self.unet))):
                    raise ValueError(f"Unexpected save model: {model.__class__}")
                unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                if weights:
                    weights.pop()

            StableDiffusionPipeline.save_lora_weights(
                save_directory=output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                safe_serialization=True,
            )

        def load_model_hook(models: list[torch.nn.Module], input_dir: str) -> None:
            unet = None
            while models:
                model = models.pop()
                if not isinstance(model, type(unwrap_model(self.unet))):
                    raise ValueError(f"Unexpected load model: {model.__class__}")
                unet = model

            lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(input_dir)
            unet_state_dict = {
                key.replace("unet.", ""): value
                for key, value in lora_state_dict.items()
                if key.startswith("unet.")
            }
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")
            if accelerator.mixed_precision == "fp16":
                cast_training_params([unet], dtype=torch.float32)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        self._checkpoint_hooks_registered = True

    def _unwrap_unet(self, accelerator: Any | None) -> torch.nn.Module:
        assert self.unet is not None
        if accelerator is None:
            return self.unet
        return accelerator.unwrap_model(self.unet)

    def save_checkpoint(self, output_dir: str, accelerator: Any | None = None) -> None:
        unet = self._unwrap_unet(accelerator)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

    def load_checkpoint(self, input_dir: str, accelerator: Any | None = None) -> None:
        unet = self._unwrap_unet(accelerator)
        lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(input_dir)
        unet_state_dict = {
            key.replace("unet.", ""): value
            for key, value in lora_state_dict.items()
            if key.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")
        if accelerator is not None and accelerator.mixed_precision == "fp16":
            cast_training_params([unet], dtype=torch.float32)

    def build_validation_pipeline(self, accelerator: Any) -> Any:
        unet = self._unwrap_unet(accelerator)
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config["model"]["pretrained_model_name_or_path"],
            revision=self.config["model"].get("revision"),
            variant=self.config["model"].get("variant"),
            local_files_only=self.config["model"].get("local_files_only", True),
            safety_checker=None,
            feature_extractor=None,
            unet=unet,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            torch_dtype=self.weight_dtype,
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def generate_validation_images(
        self,
        accelerator: Any,
        prompt: str,
        num_images: int,
        seed: int | None,
    ) -> list[Any]:
        pipeline = self.build_validation_pipeline(accelerator)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=accelerator.device).manual_seed(seed)

        autocast_context = (
            torch.autocast(accelerator.device.type, dtype=self.weight_dtype)
            if accelerator.device.type != "cpu" and self.weight_dtype in {torch.float16, torch.bfloat16}
            else nullcontext()
        )
        with autocast_context:
            images = [
                pipeline(prompt=prompt, num_inference_steps=30, generator=generator).images[0]
                for _ in range(num_images)
            ]
        del pipeline
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        return images
