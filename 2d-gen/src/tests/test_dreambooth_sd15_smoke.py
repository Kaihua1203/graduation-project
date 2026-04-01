from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from train.run_dreambooth_sd15 import (
    load_dreambooth_config,
    log_validation,
    parse_args,
    resolve_prior_generation_dtype,
    should_run_validation,
)


class DreamBoothSd15SmokeTest(unittest.TestCase):
    def test_parse_args_accepts_config_flag(self) -> None:
        args = parse_args(["--config", "configs/train/train_sd15_dreambooth_example.yaml"])
        self.assertEqual(args.config, "configs/train/train_sd15_dreambooth_example.yaml")

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

    def test_load_config_rejects_validation_epochs(self) -> None:
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
                    validation:
                      validation_epochs: 1
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "validation\\.validation_epochs"):
                load_dreambooth_config(config_path)

    def test_validation_schedule_ignores_tracker_choice(self) -> None:
        config = {
            "validation": {
                "validation_prompt": "photo of tok sample",
                "validation_steps": 2,
                "num_validation_images": 1,
            },
            "logging": {"report_to": "none"},
        }
        self.assertFalse(should_run_validation(config, global_step=1))
        self.assertTrue(should_run_validation(config, global_step=2))

    def test_prior_generation_fp16_falls_back_to_fp32_on_cpu(self) -> None:
        accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.assertEqual(resolve_prior_generation_dtype("fp16", accelerator), torch.float32)

    def test_log_validation_does_not_log_prompt_to_swanlab(self) -> None:
        class DummyTracker:
            name = "swanlab"

            def __init__(self) -> None:
                self.logged_values: list[tuple[dict[str, object], int | None]] = []
                self.logged_images: list[tuple[dict[str, object], int | None]] = []

            def log(self, values: dict[str, object], step: int | None = None) -> None:
                self.logged_values.append((values, step))

            def log_images(self, values: dict[str, object], step: int | None = None) -> None:
                self.logged_images.append((values, step))

        class DummyPipeline:
            def __init__(self) -> None:
                self.scheduler = SimpleNamespace(config=SimpleNamespace(variance_type="fixed_small"))

            def to(self, device):
                del device
                return self

            def set_progress_bar_config(self, disable: bool) -> None:
                del disable

            def __call__(self, prompt: str, generator, num_inference_steps: int):
                del prompt, generator, num_inference_steps
                return SimpleNamespace(images=[np.zeros((4, 4, 3), dtype=np.uint8)])

        class DummyAccelerator:
            def __init__(self) -> None:
                self.is_main_process = True
                self.device = torch.device("cpu")
                self.trackers = [DummyTracker()]
                self.log_calls: list[tuple[dict[str, object], int | None]] = []

            def wait_for_everyone(self) -> None:
                return None

            def log(self, values: dict[str, object], step: int | None = None) -> None:
                self.log_calls.append((values, step))

            def unwrap_model(self, model):
                return model

        config = {
            "model": {
                "pretrained_model_name_or_path": "/tmp/model",
                "local_files_only": True,
            },
            "train": {
                "seed": 3407,
            },
            "validation": {
                "validation_prompt": "photo of tok sample",
                "num_validation_images": 2,
                "validation_steps": 1,
            },
        }
        accelerator = DummyAccelerator()

        with patch("train.run_dreambooth_sd15.StableDiffusionPipeline.from_pretrained", return_value=DummyPipeline()), patch(
            "train.run_dreambooth_sd15.DPMSolverMultistepScheduler.from_config",
            side_effect=lambda config, **kwargs: SimpleNamespace(config=config, kwargs=kwargs),
        ):
            log_validation(
                accelerator=accelerator,
                config=config,
                unet=torch.nn.Linear(1, 1),
                vae=object(),
                tokenizer=object(),
                text_encoder=object(),
                weight_dtype=torch.float32,
                epoch=0,
                global_step=3,
                phase_name="test",
            )

        self.assertEqual(accelerator.log_calls, [({"test/epoch": 1, "test/global_step": 3}, 3)])
        tracker = accelerator.trackers[0]
        self.assertEqual(tracker.logged_values, [])
        self.assertEqual(len(tracker.logged_images), 1)


if __name__ == "__main__":
    unittest.main()
