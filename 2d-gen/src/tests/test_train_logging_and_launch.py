from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from train import base_trainer
from train.base_trainer import BaseDiffusionTrainer


class DummyTracker:
    name = "swanlab"

    def __init__(self) -> None:
        self.logged_values: list[tuple[dict[str, object], int | None]] = []
        self.logged_images: list[tuple[dict[str, object], int | None]] = []

    def log(self, values: dict[str, object], step: int | None = None) -> None:
        self.logged_values.append((values, step))

    def log_images(self, values: dict[str, object], step: int | None = None) -> None:
        self.logged_images.append((values, step))


class FakeAccelerator:
    def __init__(self, *args, **kwargs) -> None:
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        self.mixed_precision = kwargs.get("mixed_precision", "no")
        self.num_processes = kwargs.get("num_processes", 1)
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.is_local_main_process = True
        self.sync_gradients = True
        self.trackers: list[DummyTracker] = []
        self.init_trackers_calls: list[dict[str, object]] = []
        self.log_calls: list[tuple[dict[str, object], int | None]] = []
        self.wait_calls = 0
        self.print_calls: list[str] = []

    def init_trackers(self, project_name: str, config: dict | None = None, init_kwargs: dict | None = None) -> None:
        self.init_trackers_calls.append(
            {
                "project_name": project_name,
                "config": config,
                "init_kwargs": init_kwargs,
            }
        )
        self.trackers = [DummyTracker()]

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1

    def log(self, values: dict[str, object], step: int | None = None) -> None:
        self.log_calls.append((values, step))

    def print(self, message: str) -> None:
        self.print_calls.append(message)

    def gather(self, value):  # pragma: no cover - not used in this test
        return value

    def backward(self, loss):  # pragma: no cover - not used in this test
        return None

    def clip_grad_norm_(self, *args, **kwargs):  # pragma: no cover - not used in this test
        return None

    def accumulate(self, *args, **kwargs):  # pragma: no cover - not used in this test
        class _ContextManager:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _ContextManager()

    def prepare(self, *args, **kwargs):  # pragma: no cover - not used in this test
        return args

    def save_state(self, *args, **kwargs):  # pragma: no cover - not used in this test
        return None

    def load_state(self, *args, **kwargs):  # pragma: no cover - not used in this test
        return None

    def end_training(self):  # pragma: no cover - not used in this test
        return None


class DummyAdapter:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config

    def generate_validation_images(self, accelerator, prompt: str, num_images: int, seed: int | None):
        return [f"image-{index}" for index in range(num_images)]


class DummyTrainAdapter:
    def __init__(self, config: dict[str, object]) -> None:
        self.config = config
        self.parameter = torch.nn.Parameter(torch.tensor(1.0))

    def setup(self, accelerator, weight_dtype) -> None:
        del accelerator, weight_dtype

    def get_models_for_accelerator_prepare(self):
        return (torch.nn.Linear(1, 1),)

    def set_prepared_models(self, prepared_models) -> None:
        del prepared_models

    def register_checkpointing_hooks(self, accelerator) -> None:
        del accelerator

    def get_accumulate_target(self):
        return self.parameter

    def encode_text(self, batch, device, weight_dtype):
        del batch, device, weight_dtype
        return torch.zeros(1, 1)

    def encode_latents(self, batch, device, weight_dtype):
        del device, weight_dtype
        return batch["pixel_values"]

    def sample_noise(self, clean_latents):
        return torch.zeros_like(clean_latents)

    def sample_time_state(self, clean_latents, batch, device):
        del clean_latents, batch, device
        return torch.zeros(1, dtype=torch.long)

    def prepare_noisy_input(self, clean_latents, time_state, noise, batch, device, weight_dtype):
        del time_state, noise, batch, device, weight_dtype
        return clean_latents

    def forward_model(self, model_input, time_state, conditioning, batch):
        del time_state, conditioning, batch
        return model_input

    def build_target(self, clean_latents, noise, time_state, batch):
        del noise, time_state, batch
        return clean_latents

    def compute_loss(self, prediction, target, time_state, batch):
        del target, time_state, batch
        return prediction.sum() * 0 + self.parameter * 0 + torch.tensor(1.0, requires_grad=True)

    def get_grad_clip_parameters(self):
        return [self.parameter]

    def get_trainable_parameters(self):
        return [self.parameter]

    def save_checkpoint(self, final_checkpoint, accelerator) -> None:
        del accelerator
        final_checkpoint.mkdir(parents=True, exist_ok=True)

    def on_checkpoint_loaded(self, accelerator) -> None:
        del accelerator


def make_minimal_config(output_dir: Path) -> dict[str, object]:
    return {
        "model": {
            "family": "stable_diffusion",
            "pretrained_model_name_or_path": "/tmp/model",
            "local_files_only": True,
        },
        "data": {
            "train_manifest": "/tmp/train.jsonl",
            "resolution": 256,
        },
        "train": {
            "output_dir": str(output_dir),
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
            "num_validation_images": 2,
            "validation_epochs": 1,
        },
        "logging": {
            "report_to": "swanlab",
            "logging_dir": "logs",
            "project_name": "2d-gen-train",
            "experiment_name": "unit-test-run",
        },
        "distributed": {
            "find_unused_parameters": False,
        },
    }


class TrainerLoggingSmokeTest(unittest.TestCase):
    def test_initialize_trackers_uses_project_and_experiment_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            config = make_minimal_config(output_dir)

            with patch.object(base_trainer, "Accelerator", FakeAccelerator), patch.object(
                base_trainer, "resolve_adapter_class", return_value=DummyAdapter
            ):
                trainer = BaseDiffusionTrainer(config)

            trainer.initialize_trackers()

            self.assertEqual(len(trainer.accelerator.init_trackers_calls), 1)
            call = trainer.accelerator.init_trackers_calls[0]
            self.assertEqual(call["project_name"], "2d-gen-train")
            self.assertEqual(call["init_kwargs"], {"swanlab": {"experiment_name": "unit-test-run"}})
            tracker_config = call["config"]
            self.assertIsInstance(tracker_config, dict)
            self.assertEqual(tracker_config["logging.project_name"], "2d-gen-train")
            self.assertEqual(tracker_config["logging.experiment_name"], "unit-test-run")

    def test_log_validation_does_not_log_prompt_to_swanlab(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            config = make_minimal_config(output_dir)

            with patch.object(base_trainer, "Accelerator", FakeAccelerator), patch.object(
                base_trainer, "resolve_adapter_class", return_value=DummyAdapter
            ):
                trainer = BaseDiffusionTrainer(config)

            trainer.initialize_trackers()
            trainer.log_validation("validation", epoch=0, global_step=3)

            self.assertEqual(len(trainer.accelerator.log_calls), 1)
            self.assertEqual(trainer.accelerator.log_calls[0][0], {"validation/epoch": 1, "validation/global_step": 3})
            tracker = trainer.accelerator.trackers[0]
            self.assertEqual(len(tracker.logged_values), 0)
            self.assertEqual(len(tracker.logged_images), 1)
            self.assertEqual(tracker.logged_images[0][0], {"validation/images": ["image-0", "image-1"]})

    def test_train_recomputes_epoch_count_from_max_train_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            config = make_minimal_config(output_dir)
            config["train"]["num_train_epochs"] = 1
            config["train"]["max_train_steps"] = 5
            config["train"]["train_batch_size"] = 3
            config["train"]["gradient_accumulation_steps"] = 1
            config["train"]["optimizer"] = {
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 1.0e-2,
                "adam_epsilon": 1.0e-8,
            }
            config["train"]["lr_scheduler"] = "constant"
            config["train"]["lr_warmup_steps"] = 0
            config["train"]["checkpointing_steps"] = 10
            config["train"]["max_grad_norm"] = 1.0
            config["logging"]["report_to"] = "none"
            config["validation"]["validation_prompt"] = None

            batches = [
                {"pixel_values": torch.zeros(1, 3, 8, 8), "prompt": ["a"], "image_path": ["a.png"]},
                {"pixel_values": torch.zeros(1, 3, 8, 8), "prompt": ["b"], "image_path": ["b.png"]},
            ]

            with patch.object(base_trainer, "Accelerator", FakeAccelerator), patch.object(
                base_trainer, "resolve_adapter_class", return_value=DummyTrainAdapter
            ), patch.object(BaseDiffusionTrainer, "build_dataloader", return_value=batches):
                trainer = BaseDiffusionTrainer(config)
                trainer.accelerator.num_processes = 2
                final_checkpoint = trainer.train()

            summary_path = output_dir / "train_summary.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(final_checkpoint.exists())
            self.assertIn("***** Running training *****", trainer.accelerator.print_calls)
            self.assertIn("  Num batches each epoch = 2", trainer.accelerator.print_calls)
            self.assertIn("  Num Epochs = 3", trainer.accelerator.print_calls)
            self.assertIn(
                "  Total train batch size (w. parallel, distributed & accumulation) = 6",
                trainer.accelerator.print_calls,
            )
            self.assertIn("  Total optimization steps = 5", trainer.accelerator.print_calls)

    def test_train_raises_clear_error_for_empty_dataloader(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            config = make_minimal_config(output_dir)
            config["train"]["optimizer"] = {
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_weight_decay": 1.0e-2,
                "adam_epsilon": 1.0e-8,
            }
            config["train"]["lr_scheduler"] = "constant"
            config["train"]["lr_warmup_steps"] = 0
            config["train"]["checkpointing_steps"] = 10
            config["train"]["max_grad_norm"] = 1.0
            config["logging"]["report_to"] = "none"
            config["validation"]["validation_prompt"] = None

            with patch.object(base_trainer, "Accelerator", FakeAccelerator), patch.object(
                base_trainer, "resolve_adapter_class", return_value=DummyTrainAdapter
            ), patch.object(BaseDiffusionTrainer, "build_dataloader", return_value=[]):
                trainer = BaseDiffusionTrainer(config)
                with self.assertRaisesRegex(ValueError, "Training dataloader is empty"):
                    trainer.train()


class RunTrainScriptSmokeTest(unittest.TestCase):
    def _make_env(self, tmpdir: Path) -> dict[str, str]:
        env = os.environ.copy()
        venv_dir = tmpdir / "venv"
        bin_dir = tmpdir / "bin"
        venv_bin = venv_dir / "bin"
        venv_bin.mkdir(parents=True)
        bin_dir.mkdir()

        (venv_bin / "activate").write_text(":\n", encoding="utf-8")
        accelerate_path = bin_dir / "accelerate"
        accelerate_path.write_text(
            "#!/usr/bin/env bash\n"
            "for arg in \"$@\"; do\n"
            "  printf '%s\\n' \"$arg\"\n"
            "done\n",
            encoding="utf-8",
        )
        accelerate_path.chmod(0o755)

        env["VENV_DIR"] = str(venv_dir)
        env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
        return env

    def test_single_gpu_launch_is_unchanged(self) -> None:
        repo_root = Path("/tmp/2d-gen-thread-1-train-logging/2d-gen")
        script = repo_root / "scripts" / "run_train.sh"
        config_path = repo_root / "configs" / "train_sd_lora_example.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._make_env(Path(tmpdir))
            result = subprocess.run(
                ["bash", str(script), str(config_path)],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        output_lines = result.stdout.splitlines()
        self.assertEqual(output_lines[0], "launch")
        self.assertEqual(Path(output_lines[1]).resolve(), (repo_root / "src" / "train" / "run_train.py").resolve())
        self.assertEqual(output_lines[2], "--config")
        self.assertEqual(Path(output_lines[3]).resolve(), config_path.resolve())

    def test_multi_gpu_launch_requires_visible_devices_and_matching_processes(self) -> None:
        repo_root = Path("/tmp/2d-gen-thread-1-train-logging/2d-gen")
        script = repo_root / "scripts" / "run_train.sh"
        config_path = repo_root / "configs" / "train_sd_lora_example.yaml"

        with tempfile.TemporaryDirectory() as tmpdir:
            env = self._make_env(Path(tmpdir))
            env["CUDA_VISIBLE_DEVICES"] = "0"
            result = subprocess.run(
                [
                    "bash",
                    str(script),
                    str(config_path),
                    "--multi_gpu",
                    "--num_processes",
                    "2",
                ],
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA_VISIBLE_DEVICES lists 1 GPU(s) but --num_processes is 2", result.stderr)


if __name__ == "__main__":
    unittest.main()
