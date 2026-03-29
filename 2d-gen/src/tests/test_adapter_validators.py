from __future__ import annotations

import unittest
from types import SimpleNamespace

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
        "model": {
            "family": "stable_diffusion",
            "pretrained_model_name_or_path": "/tmp/model",
            "local_files_only": True,
        },
        "train": {
            "gradient_checkpointing": False,
            "prediction_type": None,
            "enable_xformers_memory_efficient_attention": False,
            "lora": {
                "rank": 4,
                "alpha": 8,
                "dropout": 0.0,
                "target_modules": ["to_q"],
            },
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

    def test_sdxl_accepts_full_conditioning(self) -> None:
        adapter = SDXLAdapter(make_config())
        adapter.validate_conditioning(
            Conditioning(
                prompt_embeds=torch.zeros(2, 77, 2048),
                pooled_prompt_embeds=torch.zeros(2, 1280),
                add_time_ids=torch.zeros(2, 6),
            )
        )

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

    def test_flux_accepts_sequence_level_text_ids(self) -> None:
        adapter = FluxAdapter(make_config())
        adapter.validate_conditioning(
            Conditioning(
                prompt_embeds=torch.zeros(2, 77, 4096),
                pooled_prompt_embeds=torch.zeros(2, 768),
                text_ids=torch.zeros(77, 3),
            )
        )

    def test_flux_detects_guidance_from_wrapped_transformer(self) -> None:
        adapter = FluxAdapter(make_config())
        adapter.requires_guidance = True
        adapter.transformer = SimpleNamespace(module=SimpleNamespace(config=SimpleNamespace(guidance_embeds=True)))
        self.assertTrue(adapter._transformer_requires_guidance())

    def test_qwen_requires_prompt_mask(self) -> None:
        adapter = QwenImageAdapter(make_config())
        with self.assertRaises(ValueError):
            adapter.validate_conditioning(Conditioning(prompt_embeds=torch.zeros(2, 256, 3584)))

    def test_qwen_accepts_prompt_mask_and_packed_latents(self) -> None:
        adapter = QwenImageAdapter(make_config())
        adapter.validate_conditioning(
            Conditioning(
                prompt_embeds=torch.zeros(2, 256, 3584),
                prompt_mask=torch.ones(2, 256, dtype=torch.long),
            )
        )
        adapter.validate_latents(torch.zeros(2, 4096, 64))


if __name__ == "__main__":
    unittest.main()
