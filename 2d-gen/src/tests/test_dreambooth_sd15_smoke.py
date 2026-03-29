from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from train.run_dreambooth_sd15 import (
    load_dreambooth_config,
    parse_args,
    resolve_prior_generation_dtype,
    should_run_validation,
)


class DreamBoothSd15SmokeTest(unittest.TestCase):
    def test_parse_args_accepts_config_flag(self) -> None:
        args = parse_args(["--config", "configs/train_sd15_dreambooth_example.yaml"])
        self.assertEqual(args.config, "configs/train_sd15_dreambooth_example.yaml")

    def test_load_config_normalizes_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            instance_dir = root / "instance"
            class_dir = root / "class"
            model_dir.mkdir()
            instance_dir.mkdir()
            class_dir.mkdir()

            config_path = root / "dreambooth.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    model:
                      pretrained_model_name_or_path: {model_dir}
                    data:
                      instance_data_dir: {instance_dir}
                      instance_prompt: photo of tok sample
                      class_data_dir: {class_dir}
                      class_prompt: photo of a sample
                      with_prior_preservation: true
                    train:
                      output_dir: {root / "outputs"}
                      max_train_steps: 2
                      lora:
                        target_modules: [to_k, to_q]
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            config = load_dreambooth_config(config_path)
            self.assertEqual(config["model"]["pretrained_model_name_or_path"], str(model_dir.resolve()))
            self.assertEqual(config["data"]["instance_data_dir"], str(instance_dir.resolve()))
            self.assertTrue(config["data"]["with_prior_preservation"])
            self.assertEqual(config["train"]["max_train_steps"], 2)
            self.assertEqual(config["train"]["mixed_precision"], "no")

    def test_validation_schedule_ignores_tracker_choice(self) -> None:
        config = {
            "validation": {
                "validation_prompt": "photo of tok sample",
                "validation_epochs": 1,
                "num_validation_images": 1,
            },
            "logging": {"report_to": "none"},
        }
        self.assertTrue(should_run_validation(config, epoch=0))

    def test_prior_generation_fp16_falls_back_to_fp32_on_cpu(self) -> None:
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.assertEqual(resolve_prior_generation_dtype("fp16", accelerator), torch.float32)


if __name__ == "__main__":
    unittest.main()
