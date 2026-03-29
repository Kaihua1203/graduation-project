from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from common.config import load_train_config, load_yaml_config, normalize_training_config
from data.dataset import ManifestImagePromptDataset


class ConfigAndDatasetSmokeTest(unittest.TestCase):
    def test_load_yaml_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("model:\n  family: stable_diffusion\n", encoding="utf-8")
            config = load_yaml_config(config_path)
            self.assertEqual(config["model"]["family"], "stable_diffusion")

    def test_normalize_training_config(self) -> None:
        raw_config = {
            "model": {
                "family": "stable_diffusion",
                "pretrained_model_name_or_path": "/tmp/model",
            },
            "data": {
                "train_manifest": "/tmp/train.jsonl",
                "resolution": 256,
            },
            "train": {
                "output_dir": "outputs/test",
                "seed": 3407,
                "train_batch_size": 1,
                "num_train_epochs": 1,
                "max_train_steps": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1.0e-4,
                "mixed_precision": "fp16",
                "optimizer": {},
                "lora": {
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": ["to_q"],
                },
                "sdxl": {},
            },
            "validation": {
                "validation_prompt": "a liver CT slice",
            },
            "logging": {
                "report_to": "swanlab",
                "project_name": "2d-gen-train",
                "experiment_name": "unit-test-run",
            },
        }

        normalized = normalize_training_config(raw_config)
        self.assertEqual(normalized["model"]["pretrained_model_name_or_path"], "/tmp/model")
        self.assertEqual(normalized["train"]["train_batch_size"], 1)
        self.assertEqual(normalized["data"]["resolution"], 256)
        self.assertEqual(normalized["validation"]["num_validation_images"], 4)
        self.assertEqual(normalized["logging"]["report_to"], "swanlab")
        self.assertEqual(normalized["logging"]["project_name"], "2d-gen-train")
        self.assertEqual(normalized["logging"]["experiment_name"], "unit-test-run")
        self.assertFalse(normalized["distributed"]["find_unused_parameters"])

    def test_normalize_training_config_rejects_unimplemented_knobs(self) -> None:
        raw_config = {
            "model": {
                "family": "stable_diffusion",
                "pretrained_model_name_or_path": "/tmp/model",
            },
            "data": {
                "train_manifest": "/tmp/train.jsonl",
                "resolution": 256,
            },
            "train": {
                "output_dir": "outputs/test",
                "seed": 3407,
                "train_batch_size": 1,
                "num_train_epochs": 1,
                "max_train_steps": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1.0e-4,
                "mixed_precision": "fp16",
                "optimizer": {"use_8bit_adam": True},
                "lora": {
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": ["to_q"],
                },
                "sdxl": {},
            },
        }

        with self.assertRaises(ValueError):
            normalize_training_config(raw_config)

    def test_normalize_training_config_rejects_validation_epochs(self) -> None:
        raw_config = {
            "model": {
                "family": "stable_diffusion",
                "pretrained_model_name_or_path": "/tmp/model",
            },
            "data": {
                "train_manifest": "/tmp/train.jsonl",
                "resolution": 256,
            },
            "train": {
                "output_dir": "outputs/test",
                "seed": 3407,
                "train_batch_size": 1,
                "num_train_epochs": 1,
                "max_train_steps": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1.0e-4,
                "mixed_precision": "fp16",
                "optimizer": {},
                "lora": {
                    "rank": 4,
                    "alpha": 8,
                    "dropout": 0.0,
                    "target_modules": ["to_q"],
                },
                "sdxl": {},
            },
            "validation": {
                "validation_prompt": "a liver CT slice",
                "validation_epochs": 1,
            },
        }

        with self.assertRaisesRegex(ValueError, "validation\\.validation_epochs"):
            normalize_training_config(raw_config)

    def test_manifest_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_path = tmpdir_path / "sample.png"
            Image.new("RGB", (32, 32), color=(128, 64, 32)).save(image_path)

            manifest_path = tmpdir_path / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"image_path": str(image_path), "prompt": "toy prompt"}) + "\n",
                encoding="utf-8",
            )

            dataset = ManifestImagePromptDataset(manifest_path, resolution=32)
            sample = dataset[0]
            self.assertEqual(sample["prompt"], "toy prompt")
            self.assertEqual(tuple(sample["pixel_values"].shape), (3, 32, 32))


if __name__ == "__main__":
    unittest.main()
