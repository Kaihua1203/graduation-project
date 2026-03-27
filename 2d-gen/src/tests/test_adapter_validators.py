from __future__ import annotations

import unittest

import torch

from common.types import Conditioning
from train.adapters.flux import FluxAdapter
from train.adapters.qwenimage import QwenImageAdapter
from train.adapters.sdxl import SDXLAdapter

try:
    from train.adapters.stable_diffusion import StableDiffusionAdapter
except ModuleNotFoundError:
    StableDiffusionAdapter = None


def make_config() -> dict:
    return {
        "model": {"family": "stable_diffusion", "pretrained_path": "/tmp/model"},
        "train": {
            "lora_rank": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "target_modules": ["to_q"],
        },
    }


class AdapterValidatorSmokeTest(unittest.TestCase):
    @unittest.skipIf(
        StableDiffusionAdapter is None,
        "StableDiffusionAdapter validation test requires peft/diffusers imports.",
    )
    def test_stable_diffusion_shape_validators(self) -> None:
        adapter = StableDiffusionAdapter(make_config())
        adapter.validate_conditioning(Conditioning(prompt_embeds=torch.zeros(2, 77, 768)))
        adapter.validate_latents(torch.zeros(2, 4, 64, 64))

    def test_sdxl_requires_pooled_and_time_ids(self) -> None:
        adapter = SDXLAdapter(make_config())
        with self.assertRaises(ValueError):
            adapter.validate_conditioning(Conditioning(prompt_embeds=torch.zeros(2, 77, 2048)))

    def test_flux_requires_packed_latents(self) -> None:
        adapter = FluxAdapter(make_config())
        adapter.validate_conditioning(
            Conditioning(
                prompt_embeds=torch.zeros(2, 77, 4096),
                pooled_prompt_embeds=torch.zeros(2, 768),
                text_ids=torch.zeros(2, 77, 3),
            )
        )
        adapter.validate_latents(torch.zeros(2, 4096, 64))
        with self.assertRaises(ValueError):
            adapter.validate_latents(torch.zeros(2, 4, 64, 64))

    def test_qwen_requires_prompt_mask(self) -> None:
        adapter = QwenImageAdapter(make_config())
        with self.assertRaises(ValueError):
            adapter.validate_conditioning(Conditioning(prompt_embeds=torch.zeros(2, 256, 3584)))


if __name__ == "__main__":
    unittest.main()
