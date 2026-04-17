from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from common.runtime import set_seed
from common.diffusers_import import prepare_diffusers_import
from uncond_ldm.config import normalize_uncond_ldm_train_config
from uncond_ldm.dataset import ImageOnlyDataset
from uncond_ldm.trainer import UncondLatentDiffusionTrainer

prepare_diffusers_import()

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel


def make_config(output_dir: Path, model_dir: Path, manifest_path: str | None, image_dir: str | None) -> dict[str, object]:
    return {
        "model": {
            "model_type": "uncond_ldm",
            "pretrained_vae_model_name_or_path": str(model_dir),
            "vae_subfolder": None,
            "local_files_only": True,
            "unet": {
                "sample_size": 4,
                "in_channels": 4,
                "out_channels": 4,
                "layers_per_block": 1,
                "block_out_channels": [32, 64],
                "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
                "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
                "norm_num_groups": 8,
                "attention_head_dim": 8,
                "dropout": 0.0,
            },
            "scheduler": {
                "num_train_timesteps": 20,
                "beta_start": 1.0e-4,
                "beta_end": 2.0e-2,
                "beta_schedule": "linear",
                "prediction_type": "epsilon",
                "variance_type": "fixed_small",
                "clip_sample": False,
            },
        },
        "data": {
            "train_manifest": manifest_path,
            "train_image_dir": image_dir,
            "image_column": "image_path",
            "resolution": 32,
            "center_crop": False,
            "random_flip": False,
            "image_interpolation_mode": "bilinear",
            "max_train_samples": None,
        },
        "train": {
            "output_dir": str(output_dir),
            "seed": 7,
            "train_batch_size": 1,
            "num_train_epochs": 1,
            "max_train_steps": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1.0e-4,
            "scale_lr": False,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "gradient_checkpointing": False,
            "mixed_precision": "no",
            "max_grad_norm": 1.0,
            "allow_tf32": False,
            "dataloader_num_workers": 0,
            "checkpointing_steps": 1,
            "checkpoints_total_limit": 2,
            "resume_from_checkpoint": None,
            "optimizer": {
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 0.0,
                "adam_epsilon": 1.0e-8,
            },
        },
        "validation": {
            "num_validation_images": 0,
            "validation_steps": 1,
            "num_inference_steps": 2,
            "seed": 11,
        },
        "logging": {
            "report_to": "none",
            "logging_dir": "logs",
            "project_name": "2d-gen-uncond-ldm-test",
            "experiment_name": "unit-test",
        },
        "distributed": {
            "find_unused_parameters": False,
        },
    }


class UncondLatentDiffusionTest(unittest.TestCase):
    def _write_tiny_vae(self, directory: Path) -> None:
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            block_out_channels=(32, 32, 64, 64),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=8,
            sample_size=32,
            scaling_factor=0.18215,
        )
        vae.save_pretrained(directory)

    def test_config_validation_requires_single_data_source(self) -> None:
        raw_config = {
            "model": {
                "model_type": "uncond_ldm",
                "pretrained_vae_model_name_or_path": "/tmp/vae",
            },
            "data": {
                "train_manifest": "/tmp/train.jsonl",
                "train_image_dir": "/tmp/images",
                "resolution": 512,
            },
            "train": {
                "output_dir": "/tmp/out",
                "max_train_steps": 1,
            },
        }

        with self.assertRaisesRegex(ValueError, "Exactly one of data.train_manifest or data.train_image_dir"):
            normalize_uncond_ldm_train_config(raw_config)

    def test_image_only_dataset_supports_image_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()
            Image.new("L", (24, 24), color=128).save(image_dir / "slice_a.png")
            Image.new("L", (24, 24), color=64).save(image_dir / "slice_b.png")

            dataset = ImageOnlyDataset(image_dir=image_dir, resolution=32)
            sample = dataset[0]

            self.assertEqual(len(dataset), 2)
            self.assertEqual(tuple(sample["pixel_values"].shape), (3, 32, 32))
            self.assertTrue(sample["image_path"].endswith(".png"))

    def test_vae_encode_and_unet_forward_smoke(self) -> None:
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            block_out_channels=(32, 32, 64, 64),
            layers_per_block=1,
            latent_channels=4,
            norm_num_groups=8,
            sample_size=32,
            scaling_factor=0.18215,
        )
        unet = UNet2DModel(
            sample_size=4,
            in_channels=4,
            out_channels=4,
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            norm_num_groups=8,
            attention_head_dim=8,
        )
        scheduler = DDPMScheduler(num_train_timesteps=10, prediction_type="epsilon", clip_sample=False)

        pixel_values = torch.randn(1, 3, 32, 32)
        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), dtype=torch.long)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        prediction = unet(noisy_latents, timesteps, return_dict=False)[0]

        self.assertEqual(tuple(latents.shape), (1, 4, 4, 4))
        self.assertEqual(tuple(prediction.shape), (1, 4, 4, 4))

    def test_trainer_writes_checkpoint_and_export_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            model_dir = tmpdir_path / "tiny_vae"
            self._write_tiny_vae(model_dir)

            image_path = tmpdir_path / "sample.png"
            Image.new("L", (32, 32), color=128).save(image_path)
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(json.dumps({"image_path": str(image_path)}) + "\n", encoding="utf-8")

            config = normalize_uncond_ldm_train_config(
                make_config(
                    output_dir=tmpdir_path / "outputs",
                    model_dir=model_dir,
                    manifest_path=str(manifest_path),
                    image_dir=None,
                )
            )

            set_seed(7)
            trainer = UncondLatentDiffusionTrainer(config)
            export_dir = trainer.train()

            self.assertTrue((tmpdir_path / "outputs" / "checkpoints" / "checkpoint-1" / "checkpoint_state.json").exists())
            self.assertTrue((export_dir / "unet" / "config.json").exists())
            self.assertTrue((export_dir / "scheduler" / "scheduler_config.json").exists())
            self.assertTrue((export_dir / "metadata.json").exists())
            self.assertTrue((export_dir / "train_summary.json").exists())

    def test_trainer_can_resume_from_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            model_dir = tmpdir_path / "tiny_vae"
            self._write_tiny_vae(model_dir)

            image_path = tmpdir_path / "sample.png"
            Image.new("L", (32, 32), color=128).save(image_path)
            manifest_path = tmpdir_path / "train.jsonl"
            manifest_path.write_text(json.dumps({"image_path": str(image_path)}) + "\n", encoding="utf-8")

            base_config = normalize_uncond_ldm_train_config(
                make_config(
                    output_dir=tmpdir_path / "outputs",
                    model_dir=model_dir,
                    manifest_path=str(manifest_path),
                    image_dir=None,
                )
            )
            trainer = UncondLatentDiffusionTrainer(base_config)
            trainer.train()

            resumed_config = normalize_uncond_ldm_train_config(
                make_config(
                    output_dir=tmpdir_path / "outputs",
                    model_dir=model_dir,
                    manifest_path=str(manifest_path),
                    image_dir=None,
                )
            )
            resumed_config["train"]["max_train_steps"] = 2
            resumed_config["train"]["resume_from_checkpoint"] = "latest"

            resumed_trainer = UncondLatentDiffusionTrainer(resumed_config)
            resumed_trainer.train()

            metadata = json.loads((tmpdir_path / "outputs" / "checkpoints" / "checkpoint-2" / "checkpoint_state.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["global_step"], 2)


if __name__ == "__main__":
    unittest.main()
