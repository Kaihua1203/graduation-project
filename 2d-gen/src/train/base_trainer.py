from __future__ import annotations

import importlib
import json
import math
import re
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.runtime import ensure_dir, write_json
from data.dataset import ManifestImagePromptDataset
from train.adapters.base import BaseModelAdapter


ADAPTERS = {
    "stable_diffusion": "train.adapters.stable_diffusion:StableDiffusionAdapter",
    "sdxl": "train.adapters.sdxl:SDXLAdapter",
    "flux": "train.adapters.flux:FluxAdapter",
    "qwenimage": "train.adapters.qwenimage:QwenImageAdapter",
}


def resolve_adapter_class(family: str) -> type[BaseModelAdapter]:
    module_name, class_name = ADAPTERS[family].split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def collate_manifest_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples], dim=0)
    return {
        "pixel_values": pixel_values,
        "prompt": [example["prompt"] for example in examples],
        "image_path": [example["image_path"] for example in examples],
    }


def flatten_config(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else key
            flatten_config(nested_prefix, nested_value, output)
        return
    output[prefix] = value


class BaseDiffusionTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_config = config["model"]
        self.data_config = config["data"]
        self.train_config = config["train"]
        self.validation_config = config["validation"]
        self.logging_config = config["logging"]
        self.distributed_config = config["distributed"]

        self.output_dir = ensure_dir(self.train_config["output_dir"])
        self.logging_dir = ensure_dir(self.output_dir / self.logging_config["logging_dir"])
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.distributed_config["find_unused_parameters"]
        )
        report_to = None if self.logging_config["report_to"] == "none" else self.logging_config["report_to"]
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.train_config["gradient_accumulation_steps"],
            mixed_precision=self.train_config["mixed_precision"] if torch.cuda.is_available() else None,
            log_with=report_to,
            project_config=ProjectConfiguration(
                project_dir=str(self.output_dir),
                logging_dir=str(self.logging_dir),
            ),
            kwargs_handlers=[ddp_kwargs],
        )

        if self.train_config.get("allow_tf32", False) and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.adapter: BaseModelAdapter = resolve_adapter_class(self.model_config["family"])(config)
        self._trackers_initialized = False

    def build_dataloader(self) -> DataLoader:
        dataset = ManifestImagePromptDataset(
            manifest_path=self.data_config["train_manifest"],
            resolution=self.data_config["resolution"],
            image_column=self.data_config["image_column"],
            caption_column=self.data_config["caption_column"],
            center_crop=self.data_config["center_crop"],
            random_flip=self.data_config["random_flip"],
            max_train_samples=self.data_config["max_train_samples"],
            image_interpolation_mode=self.data_config["image_interpolation_mode"],
        )
        return DataLoader(
            dataset,
            batch_size=self.train_config["train_batch_size"],
            shuffle=True,
            num_workers=self.train_config["dataloader_num_workers"],
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_manifest_batch,
        )

    def build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_config = self.train_config["optimizer"]
        learning_rate = self.train_config["learning_rate"]
        if self.train_config.get("scale_lr", False):
            learning_rate = learning_rate * (
                self.train_config["gradient_accumulation_steps"]
                * self.train_config["train_batch_size"]
                * self.accelerator.num_processes
            )
        return torch.optim.AdamW(
            self.adapter.get_trainable_parameters(),
            lr=learning_rate,
            betas=(optimizer_config["adam_beta1"], optimizer_config["adam_beta2"]),
            weight_decay=optimizer_config["adam_weight_decay"],
            eps=optimizer_config["adam_epsilon"],
        )

    def build_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        return get_scheduler(
            self.train_config["lr_scheduler"],
            optimizer=optimizer,
            num_warmup_steps=self.train_config["lr_warmup_steps"],
            num_training_steps=self.train_config["max_train_steps"],
        )

    def initialize_trackers(self) -> None:
        if self._trackers_initialized or not self.accelerator.is_main_process:
            return
        if self.logging_config["report_to"] == "none":
            return
        tracker_config: dict[str, Any] = {}
        flatten_config("", self.config, tracker_config)
        self.accelerator.init_trackers(
            self.logging_config["tracker_project_name"],
            config=tracker_config,
        )
        self._trackers_initialized = True

    def resolve_resume_checkpoint(self) -> tuple[Path | None, dict[str, Any]]:
        resume_value = self.train_config.get("resume_from_checkpoint")
        if not resume_value:
            return None, {}

        if resume_value == "latest":
            candidates = sorted(
                self.checkpoint_dir.glob("checkpoint-*"),
                key=lambda path: int(path.name.split("-", 1)[1]),
            )
            if not candidates:
                return None, {}
            checkpoint_path = candidates[-1]
        else:
            checkpoint_path = Path(resume_value)
            if not checkpoint_path.is_absolute():
                checkpoint_path = self.checkpoint_dir / checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        metadata_path = checkpoint_path / "checkpoint_state.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return checkpoint_path, metadata

    def save_training_checkpoint(
        self,
        *,
        global_step: int,
        epoch: int,
        dataloader_step: int,
    ) -> None:
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{global_step}"
        if self.accelerator.is_main_process:
            ensure_dir(checkpoint_path)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(checkpoint_path))
        if self.accelerator.is_main_process:
            write_json(
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "dataloader_step": dataloader_step,
                },
                checkpoint_path / "checkpoint_state.json",
            )
            self.prune_checkpoints()
        self.accelerator.wait_for_everyone()

    def prune_checkpoints(self) -> None:
        checkpoints_total_limit = self.train_config.get("checkpoints_total_limit")
        if checkpoints_total_limit is None:
            return
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-", 1)[1]),
        )
        while len(checkpoints) > checkpoints_total_limit:
            old_checkpoint = checkpoints.pop(0)
            shutil.rmtree(old_checkpoint, ignore_errors=True)

    def _log_validation(self, phase_name: str, epoch: int, global_step: int) -> None:
        prompt = self.validation_config.get("validation_prompt")
        num_images = self.validation_config.get("num_validation_images", 0)
        if not prompt or num_images <= 0 or not self.accelerator.is_main_process:
            return

        if type(self.adapter).generate_validation_images is not BaseModelAdapter.generate_validation_images:
            images = self.adapter.generate_validation_images(
                self.accelerator,
                prompt=prompt,
                num_images=num_images,
                seed=self.train_config.get("seed"),
            )
        else:
            pipeline = self.adapter.build_validation_pipeline(self.accelerator)
            generator = None
            if self.train_config.get("seed") is not None:
                generator = torch.Generator(device=self.accelerator.device).manual_seed(
                    self.train_config["seed"]
                )
            autocast_context = (
                torch.autocast(self.accelerator.device.type, dtype=self.weight_dtype)
                if self.accelerator.device.type != "cpu"
                and self.weight_dtype in {torch.float16, torch.bfloat16}
                else nullcontext()
            )
            with autocast_context:
                images = [
                    pipeline(prompt=prompt, num_inference_steps=30, generator=generator).images[0]
                    for _ in range(num_images)
                ]
            del pipeline
            if self.accelerator.device.type == "cuda":
                torch.cuda.empty_cache()

        self.accelerator.log(
            {
                f"{phase_name}/epoch": epoch + 1,
                f"{phase_name}/global_step": global_step,
            },
            step=global_step,
        )
        import swanlab

        for tracker in self.accelerator.trackers:
            if tracker.name == "swanlab":
                tracker.log(
                    {
                        f"{phase_name}/prompt": swanlab.Text(prompt),
                        f"{phase_name}/images": [
                            swanlab.Image(image, caption=f"{index}: {prompt}")
                            for index, image in enumerate(images)
                        ],
                    },
                    step=global_step,
                )

    def train(self) -> Path:
        self.adapter.setup(self.accelerator, self.weight_dtype)
        train_dataloader = self.build_dataloader()
        optimizer = self.build_optimizer()
        self.adapter.register_checkpointing_hooks(self.accelerator)

        prepared_modules = self.adapter.get_models_for_accelerator_prepare()
        prepared = self.accelerator.prepare(*prepared_modules, optimizer, train_dataloader)
        prepared_models = prepared[: len(prepared_modules)]
        optimizer = prepared[len(prepared_modules)]
        train_dataloader = prepared[len(prepared_modules) + 1]
        self.adapter.set_prepared_models(tuple(prepared_models))

        steps_per_epoch = math.ceil(len(train_dataloader) / self.train_config["gradient_accumulation_steps"])
        max_train_steps = self.train_config.get("max_train_steps")
        if max_train_steps is None:
            max_train_steps = self.train_config["num_train_epochs"] * steps_per_epoch

        lr_scheduler = self.build_lr_scheduler(optimizer)
        self.initialize_trackers()

        resume_checkpoint, resume_metadata = self.resolve_resume_checkpoint()
        global_step = int(resume_metadata.get("global_step", 0))
        first_epoch = int(resume_metadata.get("epoch", 0))
        resume_dataloader_step = int(resume_metadata.get("dataloader_step", -1))
        if resume_checkpoint is not None:
            self.accelerator.print(f"Resuming from checkpoint {resume_checkpoint}")
            self.accelerator.load_state(str(resume_checkpoint))
            self.adapter.on_checkpoint_loaded(self.accelerator)

        progress_bar = tqdm(
            total=max_train_steps,
            initial=global_step,
            disable=not self.accelerator.is_local_main_process,
            desc="train",
        )

        loss_history: list[float] = []
        train_loss = 0.0
        epoch = first_epoch
        for epoch in range(first_epoch, self.train_config["num_train_epochs"]):
            progress = tqdm(
                train_dataloader,
                desc=f"epoch={epoch}",
                leave=False,
                disable=not self.accelerator.is_local_main_process,
            )
            for dataloader_step, batch in enumerate(progress):
                if epoch == first_epoch and dataloader_step <= resume_dataloader_step:
                    continue

                with self.accelerator.accumulate(self.adapter.get_accumulate_target()):
                    conditioning = self.adapter.encode_text(batch, self.accelerator.device, self.weight_dtype)
                    clean_latents = self.adapter.encode_latents(batch, self.accelerator.device, self.weight_dtype)
                    noise = self.adapter.sample_noise(clean_latents)
                    time_state = self.adapter.sample_time_state(
                        clean_latents, batch, self.accelerator.device
                    )
                    model_input = self.adapter.prepare_noisy_input(
                        clean_latents,
                        time_state,
                        noise,
                        batch,
                        self.accelerator.device,
                        self.weight_dtype,
                    )
                    prediction = self.adapter.forward_model(model_input, time_state, conditioning, batch)
                    target = self.adapter.build_target(clean_latents, noise, time_state, batch)
                    loss = self.adapter.compute_loss(prediction, target, time_state, batch)

                    avg_loss = self.accelerator.gather(loss.detach().float()).mean()
                    loss_history.append(avg_loss.item())
                    train_loss += avg_loss.item() / self.accelerator.gradient_accumulation_steps

                    self.accelerator.backward(loss / self.accelerator.gradient_accumulation_steps)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            list(self.adapter.get_grad_clip_parameters()),
                            self.train_config["max_grad_norm"],
                        )
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                if not self.accelerator.sync_gradients:
                    continue

                global_step += 1
                progress_bar.update(1)
                progress.set_postfix(loss=f"{avg_loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                if self.accelerator.is_main_process:
                    self.accelerator.log(
                        {
                            "train/loss": train_loss,
                            "train/lr": float(lr_scheduler.get_last_lr()[0]),
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )
                train_loss = 0.0

                if global_step % self.logging_config["log_every_n_steps"] == 0:
                    self.accelerator.print(f"step={global_step} loss={avg_loss.item():.6f}")

                if global_step % self.train_config["checkpointing_steps"] == 0:
                    self.save_training_checkpoint(
                        global_step=global_step,
                        epoch=epoch,
                        dataloader_step=dataloader_step,
                    )

                if global_step >= max_train_steps:
                    break

            if global_step >= max_train_steps:
                break

            if (
                self.validation_config.get("validation_prompt") is not None
                and self.validation_config["validation_epochs"] > 0
                and (epoch + 1) % self.validation_config["validation_epochs"] == 0
            ):
                self._log_validation("validation", epoch, global_step)

        self.accelerator.wait_for_everyone()
        self._log_validation("test", epoch, global_step)
        self.accelerator.wait_for_everyone()

        final_checkpoint = self.checkpoint_dir / "final_lora"
        if self.accelerator.is_main_process:
            ensure_dir(final_checkpoint)
            self.adapter.save_checkpoint(final_checkpoint, self.accelerator)
            summary = {
                "family": self.model_config["family"],
                "num_steps": global_step,
                "num_updates": len(loss_history),
                "mean_loss": float(sum(loss_history) / max(1, len(loss_history))),
                "min_loss": float(min(loss_history)) if loss_history else math.nan,
                "final_checkpoint": str(final_checkpoint),
            }
            write_json(summary, self.output_dir / "train_summary.json")

        self.accelerator.end_training()
        return final_checkpoint
