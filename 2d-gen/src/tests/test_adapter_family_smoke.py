from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diffusers import QwenImagePipeline

from train.adapters.flux import FluxAdapter
from train.adapters.qwenimage import QwenImageAdapter
from train.adapters.sdxl import SDXLAdapter
from train.base_trainer import BaseDiffusionTrainer, collate_manifest_batch


class _LatentDistribution:
    def __init__(self, sample: torch.Tensor) -> None:
        self._sample = sample

    def sample(self) -> torch.Tensor:
        return self._sample.clone()


class _EncodeOutput:
    def __init__(self, sample: torch.Tensor) -> None:
        self.latent_dist = _LatentDistribution(sample)


class _TensorDataset(Dataset):
    def __init__(self, resolution: int, prompt: str) -> None:
        self._resolution = resolution
        self._prompt = prompt

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, object]:
        value = 0.5 + 0.1 * index
        return {
            "pixel_values": torch.full((3, self._resolution, self._resolution), value, dtype=torch.float32),
            "prompt": self._prompt,
            "image_path": f"/tmp/sample-{index}.png",
        }


class _FakeLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer

    def step(self) -> None:
        return None

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class _FakeSDXLTokenizer:
    model_max_length = 4

    def __call__(self, prompts, padding, max_length, truncation, return_tensors):
        del padding, truncation, return_tensors
        batch_size = len(prompts)
        input_ids = torch.arange(max_length, dtype=torch.long).repeat(batch_size, 1)
        return SimpleNamespace(input_ids=input_ids)


class _FakeSDXLTextEncoder(nn.Module):
    def __init__(self, hidden_dim: int, pooled_dim: int, pooled_value: float = 1.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = SimpleNamespace(projection_dim=pooled_dim)
        self.anchor = nn.Parameter(torch.ones(1), requires_grad=False)
        self.pooled_value = pooled_value

    def forward(self, input_ids, output_hidden_states=True, return_dict=False):
        del output_hidden_states, return_dict
        batch_size, seq_len = input_ids.shape
        hidden = self.anchor * torch.ones(batch_size, seq_len, self.hidden_dim, device=input_ids.device)
        pooled = self.anchor * self.pooled_value * torch.ones(
            batch_size, self.config.projection_dim, device=input_ids.device
        )
        return (pooled, None, (hidden, hidden))


class _FakeSDXLUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.9))

    def forward(self, model_input, timesteps, encoder_hidden_states, added_cond_kwargs):
        del timesteps, encoder_hidden_states
        text_embeds = added_cond_kwargs["text_embeds"].mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        time_ids = added_cond_kwargs["time_ids"].mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
        sample = model_input * self.scale + 0.01 * (text_embeds + time_ids)
        return SimpleNamespace(sample=sample)


class _FakeFluxPipeline:
    def encode_prompt(self, prompt, prompt_2, device, num_images_per_prompt):
        del prompt_2, num_images_per_prompt
        batch_size = len(prompt)
        prompt_embeds = torch.ones(batch_size, 8, 12, device=device)
        pooled_prompt_embeds = torch.ones(batch_size, 6, device=device)
        text_ids = torch.zeros(8, 3, device=device)
        return prompt_embeds, pooled_prompt_embeds, text_ids


class _FakeQwenPipeline:
    def encode_prompt(self, prompt, device, num_images_per_prompt):
        del num_images_per_prompt
        batch_size = len(prompt)
        prompt_embeds = torch.ones(batch_size, 10, 16, device=device)
        prompt_mask = None
        return prompt_embeds, prompt_mask


class _FakeTransformer(nn.Module):
    def __init__(self, with_guidance: bool = False) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.75))
        self.config = SimpleNamespace(guidance_embeds=with_guidance)

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return (hidden_states * self.scale,)


class _FakeImageVAE(nn.Module):
    def __init__(self, scaling_factor: float = 0.5, shift_factor: float = 0.1) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1), requires_grad=False)
        self.dtype = torch.float32
        self.config = SimpleNamespace(
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            block_out_channels=(1, 1, 1, 1),
        )

    def encode(self, pixel_values: torch.Tensor) -> _EncodeOutput:
        batch_size = pixel_values.shape[0]
        height = pixel_values.shape[-2] // 8
        width = pixel_values.shape[-1] // 8
        latents = self.anchor * torch.ones(batch_size, 4, height, width, device=pixel_values.device)
        return _EncodeOutput(latents)


class _FakeQwenVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1), requires_grad=False)
        self.dtype = torch.float32
        self.config = SimpleNamespace(
            z_dim=4,
            latents_mean=[0.1, -0.2, 0.05, 0.3],
            latents_std=[1.2, 0.9, 1.1, 0.8],
        )

    def encode(self, pixel_values: torch.Tensor) -> _EncodeOutput:
        batch_size = pixel_values.shape[0]
        height = pixel_values.shape[-2] // 8
        width = pixel_values.shape[-1] // 8
        latents = self.anchor * torch.ones(batch_size, 4, 1, height, width, device=pixel_values.device)
        return _EncodeOutput(latents)


class _FakeSDXLAdapter(SDXLAdapter):
    def setup(self, accelerator, weight_dtype):
        self.unet = _FakeSDXLUNet()
        self.vae = _FakeImageVAE(scaling_factor=0.5, shift_factor=0.0)
        self.text_encoder = _FakeSDXLTextEncoder(hidden_dim=6, pooled_dim=6)
        self.text_encoder_2 = _FakeSDXLTextEncoder(hidden_dim=6, pooled_dim=6)
        self.tokenizer = _FakeSDXLTokenizer()
        self.tokenizer_2 = _FakeSDXLTokenizer()
        self.scheduler = SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=1000, prediction_type="epsilon"),
            add_noise=lambda latents, noise, timesteps: latents + 0.01 * timesteps.view(-1, 1, 1, 1).float() + noise,
            get_velocity=lambda latents, noise, timesteps: noise,
        )
        self.weight_dtype = weight_dtype
        self.unet.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    def _build_add_time_ids(self, batch, device, dtype, batch_size):
        del batch
        return torch.zeros(batch_size, 6, device=device, dtype=dtype)

    def register_checkpointing_hooks(self, accelerator):
        del accelerator
        return None

    def save_checkpoint(self, output_dir, accelerator=None):
        del accelerator
        Path(output_dir, "sdxl.bin").write_bytes(b"sdxl")

    def load_checkpoint(self, input_dir, accelerator=None):
        del input_dir, accelerator
        return None

    def build_validation_pipeline(self, accelerator):
        del accelerator
        raise AssertionError("validation should be disabled in smoke tests")


class _FakeFluxAdapter(FluxAdapter):
    def setup(self, accelerator, weight_dtype):
        self.pipeline = _FakeFluxPipeline()
        self.transformer = _FakeTransformer(with_guidance=True)
        self.vae = _FakeImageVAE()
        self.text_encoder = nn.Identity()
        self.text_encoder_2 = nn.Identity()
        self.tokenizer = object()
        self.tokenizer_2 = object()
        from diffusers import FlowMatchEulerDiscreteScheduler

        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.weight_dtype = weight_dtype
        self.vae_scale_factor = 8
        self.guidance_scale = 3.5
        self.transformer.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device)
        self.text_encoder_2.to(accelerator.device)

    def register_checkpointing_hooks(self, accelerator):
        del accelerator
        return None

    def save_checkpoint(self, output_dir, accelerator=None):
        del accelerator
        Path(output_dir, "flux.bin").write_bytes(b"flux")

    def load_checkpoint(self, input_dir, accelerator=None):
        del input_dir, accelerator
        return None

    def build_validation_pipeline(self, accelerator):
        del accelerator
        raise AssertionError("validation should be disabled in smoke tests")


class _FakeQwenAdapter(QwenImageAdapter):
    def setup(self, accelerator, weight_dtype):
        self.pipeline = _FakeQwenPipeline()
        self.transformer = _FakeTransformer(with_guidance=True)
        self.vae = _FakeQwenVAE()
        self.text_encoder = nn.Identity()
        self.tokenizer = object()
        from diffusers import FlowMatchEulerDiscreteScheduler

        self.scheduler = FlowMatchEulerDiscreteScheduler()
        self.weight_dtype = weight_dtype
        self.vae_scale_factor = 8
        self.guidance_scale = 4.0
        self.transformer.to(accelerator.device, dtype=weight_dtype)
        self.vae.to(accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(accelerator.device)

    def register_checkpointing_hooks(self, accelerator):
        del accelerator
        return None

    def save_checkpoint(self, output_dir, accelerator=None):
        del accelerator
        Path(output_dir, "qwen.bin").write_bytes(b"qwen")

    def load_checkpoint(self, input_dir, accelerator=None):
        del input_dir, accelerator
        return None

    def build_validation_pipeline(self, accelerator):
        del accelerator
        raise AssertionError("validation should be disabled in smoke tests")


def _make_config(output_dir: Path, family: str) -> dict[str, object]:
    return {
        "model": {
            "family": family,
            "pretrained_model_name_or_path": "/tmp/fake-model",
            "revision": None,
            "variant": None,
            "local_files_only": True,
            "pretrained_vae_model_name_or_path": None,
        },
        "data": {
            "train_manifest": "/tmp/fake-manifest.jsonl",
            "image_column": "image_path",
            "caption_column": "prompt",
            "resolution": 64,
            "center_crop": False,
            "random_flip": False,
            "image_interpolation_mode": "bilinear",
            "max_train_samples": None,
        },
        "train": {
            "output_dir": str(output_dir),
            "seed": 0,
            "train_batch_size": 2,
            "num_train_epochs": 1,
            "max_train_steps": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-3,
            "scale_lr": False,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "gradient_checkpointing": False,
            "mixed_precision": "no",
            "snr_gamma": None,
            "prediction_type": None,
            "noise_offset": 0.0,
            "max_grad_norm": 1.0,
            "allow_tf32": False,
            "dataloader_num_workers": 0,
            "checkpointing_steps": 100,
            "checkpoints_total_limit": None,
            "resume_from_checkpoint": None,
            "enable_xformers_memory_efficient_attention": False,
            "optimizer": {
                "use_8bit_adam": False,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 0.0,
                "adam_epsilon": 1e-8,
            },
            "lora": {
                "rank": 4,
                "alpha": 8,
                "dropout": 0.0,
                "target_modules": ["proj"],
            },
            "sdxl": {
                "train_text_encoder": False,
                "enable_npu_flash_attention": False,
                "debug_loss": False,
            },
        },
        "validation": {
            "validation_prompt": None,
            "num_validation_images": 1,
            "validation_steps": 1,
        },
        "logging": {
            "report_to": "none",
            "logging_dir": "logs",
            "log_every_n_steps": 1,
            "tracker_project_name": "adapter-smoke",
        },
        "distributed": {
            "find_unused_parameters": False,
        },
    }


class AdapterFamilyTrainerSmokeTest(unittest.TestCase):
    def _run_trainer_smoke(self, family: str, adapter_cls: type) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / family
            config = _make_config(output_dir, family)
            with mock.patch("train.base_trainer.resolve_adapter_class", return_value=adapter_cls):
                trainer = BaseDiffusionTrainer(config)
                trainer.build_dataloader = lambda: DataLoader(
                    _TensorDataset(resolution=64, prompt=f"{family} prompt"),
                    batch_size=2,
                    shuffle=False,
                    collate_fn=collate_manifest_batch,
                )
                trainer.build_lr_scheduler = lambda optimizer: _FakeLRScheduler(optimizer)
                final_checkpoint = trainer.train()

            self.assertTrue(final_checkpoint.exists())
            self.assertTrue((output_dir / "train_summary.json").exists())

    def test_sdxl_trainer_smoke(self) -> None:
        self._run_trainer_smoke("sdxl", _FakeSDXLAdapter)

    def test_flux_trainer_smoke(self) -> None:
        self._run_trainer_smoke("flux", _FakeFluxAdapter)

    def test_qwen_trainer_smoke(self) -> None:
        self._run_trainer_smoke("qwenimage", _FakeQwenAdapter)


class AdapterFamilyBehaviorTest(unittest.TestCase):
    def test_sdxl_uses_first_text_encoder_pooled_output(self) -> None:
        adapter = SDXLAdapter({"model": {}, "train": {}})
        adapter.tokenizer = _FakeSDXLTokenizer()
        adapter.tokenizer_2 = _FakeSDXLTokenizer()
        adapter.text_encoder = _FakeSDXLTextEncoder(hidden_dim=6, pooled_dim=4, pooled_value=3.0)
        adapter.text_encoder_2 = _FakeSDXLTextEncoder(hidden_dim=6, pooled_dim=4, pooled_value=7.0)

        _, _, pooled_prompt_embeds, _ = adapter._build_prompt_embeds(
            prompts=["a"],
            prompt_2=["b"],
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.allclose(pooled_prompt_embeds, torch.full((1, 4), 3.0)))

    def test_sdxl_load_checkpoint_accepts_single_state_dict(self) -> None:
        adapter = SDXLAdapter({"model": {}, "train": {}})
        adapter.unet = nn.Linear(1, 1)
        with (
            mock.patch("train.adapters.sdxl.StableDiffusionXLPipeline.lora_state_dict", return_value={"unet.to_q": torch.tensor(1.0)}),
            mock.patch("train.adapters.sdxl.convert_unet_state_dict_to_peft", side_effect=lambda state: state),
            mock.patch("train.adapters.sdxl.set_peft_model_state_dict") as set_state_dict,
        ):
            adapter.load_checkpoint("/tmp/fake")

        set_state_dict.assert_called_once()

    def test_qwen_uses_reference_latent_normalization(self) -> None:
        adapter = QwenImageAdapter({"model": {}, "train": {}})
        adapter.vae = _FakeQwenVAE()
        batch = {"pixel_values": torch.ones(2, 3, 64, 64)}

        packed_latents = adapter.encode_latents(batch, torch.device("cpu"), torch.float32)
        unpacked = QwenImagePipeline._unpack_latents(
            packed_latents,
            height=64,
            width=64,
            vae_scale_factor=8,
        )
        expected_channels = torch.tensor(
            [
                (1.0 - 0.1) / 1.2,
                (1.0 + 0.2) / 0.9,
                (1.0 - 0.05) / 1.1,
                (1.0 - 0.3) / 0.8,
            ],
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(unpacked[0, :, 0, 0, 0], expected_channels, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
