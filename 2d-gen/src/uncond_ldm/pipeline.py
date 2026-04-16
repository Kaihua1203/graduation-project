from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class UnconditionalLatentDiffusionPipeline:
    def __init__(self, unet, scheduler, vae, *, device: torch.device, image_size: int) -> None:
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae
        self.device = device
        self.image_size = int(image_size)
        self.latent_scale_factor = _infer_vae_scale_factor(vae)
        self.latent_scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
        self.latent_channels = int(getattr(unet.config, "in_channels", 4))

        self.unet.to(device)
        self.vae.to(device)
        self.unet.eval()
        self.vae.eval()

    def generate(self, *, seeds: Sequence[int], num_inference_steps: int) -> list[Image.Image]:
        if not seeds:
            return []
        latents = self._sample_latents(seeds=seeds, num_inference_steps=num_inference_steps)
        images = self._decode_latents(latents)
        return [_tensor_to_pil(image_tensor) for image_tensor in images]

    def _sample_latents(self, *, seeds: Sequence[int], num_inference_steps: int) -> torch.Tensor:
        latent_height = self.image_size // self.latent_scale_factor
        latent_width = self.image_size // self.latent_scale_factor
        latents = torch.cat(
            [
                torch.randn(
                    (1, self.latent_channels, latent_height, latent_width),
                    generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                    dtype=self.unet.dtype,
                )
                for seed in seeds
            ],
            dim=0,
        ).to(self.device)

        init_noise_sigma = float(getattr(self.scheduler, "init_noise_sigma", 1.0))
        latents = latents * init_noise_sigma

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        for timestep in self.scheduler.timesteps:
            model_input = latents
            if hasattr(self.scheduler, "scale_model_input"):
                model_input = self.scheduler.scale_model_input(latents, timestep)
            noise_pred = self.unet(model_input, timestep).sample
            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
        return latents

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        scaled_latents = latents / self.latent_scaling_factor
        with torch.no_grad():
            decoded = self.vae.decode(scaled_latents).sample
        images = (decoded / 2 + 0.5).clamp(0, 1)
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return images.detach().cpu()


def _infer_vae_scale_factor(vae) -> int:
    block_out_channels = getattr(vae.config, "block_out_channels", None)
    if isinstance(block_out_channels, (list, tuple)) and block_out_channels:
        return 2 ** (len(block_out_channels) - 1)
    return 8


def _tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    array = (image_tensor.mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).numpy())
    return Image.fromarray(np.asarray(array), mode="RGB")
