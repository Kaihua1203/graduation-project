from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.diffusers_import import prepare_diffusers_import
from common.runtime import ensure_dir, write_json
from uncond_ldm.dataset import ImageOnlyDataset, collate_image_only_batch

prepare_diffusers_import()

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler


LOGGER = logging.getLogger(__name__)

MODEL_METADATA_FILENAMES = ("model_metadata.json", "metadata.json")
TRAINING_SUMMARY_FILENAMES = ("training_summary.json", "train_summary.json")


def flatten_config(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else key
            flatten_config(nested_prefix, nested_value, output)
        return
    output[prefix] = value


class UncondLatentDiffusionTrainer:
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
        self.export_dir = ensure_dir(self.output_dir / "export")
        self.validation_dir = ensure_dir(self.output_dir / "validation")

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=self.distributed_config["find_unused_parameters"]
        )
        project_config = ProjectConfiguration(
            project_dir=str(self.output_dir),
            logging_dir=str(self.logging_dir),
        )
        report_to = None if self.logging_config["report_to"] == "none" else self.logging_config["report_to"]
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.train_config["gradient_accumulation_steps"],
            mixed_precision=self.train_config["mixed_precision"],
            log_with=report_to,
            project_config=project_config,
            kwargs_handlers=[ddp_kwargs],
        )

        if self.train_config["allow_tf32"] and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        self.vae: AutoencoderKL | None = None
        self.unet: UNet2DModel | None = None
        self.noise_scheduler: DDPMScheduler | None = None
        self.latent_scaling_factor: float | None = None
        self._trackers_initialized = False
        self._logging_enabled = self.logging_config["report_to"] != "none"

    def setup_models(self) -> None:
        vae_kwargs: dict[str, Any] = {"local_files_only": self.model_config["local_files_only"]}
        if self.model_config["vae_subfolder"] is not None:
            vae_kwargs["subfolder"] = self.model_config["vae_subfolder"]
        self.vae = AutoencoderKL.from_pretrained(
            self.model_config["pretrained_vae_model_name_or_path"],
            **vae_kwargs,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        unet_config = self.model_config["unet"]
        self.unet = UNet2DModel(
            sample_size=unet_config["sample_size"],
            in_channels=unet_config["in_channels"],
            out_channels=unet_config["out_channels"],
            layers_per_block=unet_config["layers_per_block"],
            block_out_channels=unet_config["block_out_channels"],
            down_block_types=unet_config["down_block_types"],
            up_block_types=unet_config["up_block_types"],
            norm_num_groups=unet_config["norm_num_groups"],
            attention_head_dim=unet_config["attention_head_dim"],
            dropout=unet_config["dropout"],
        )
        if self.train_config["gradient_checkpointing"] and hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_config["scheduler"]["num_train_timesteps"],
            beta_start=self.model_config["scheduler"]["beta_start"],
            beta_end=self.model_config["scheduler"]["beta_end"],
            beta_schedule=self.model_config["scheduler"]["beta_schedule"],
            prediction_type=self.model_config["scheduler"]["prediction_type"],
            variance_type=self.model_config["scheduler"]["variance_type"],
            clip_sample=self.model_config["scheduler"]["clip_sample"],
        )
        self.latent_scaling_factor = float(
            self.model_config["latent_scaling_factor"]
            if self.model_config["latent_scaling_factor"] is not None
            else getattr(self.vae.config, "scaling_factor", 1.0)
        )

    def build_dataloader(self) -> DataLoader:
        dataset = ImageOnlyDataset(
            manifest_path=self.data_config["train_manifest"],
            image_dir=self.data_config["train_image_dir"],
            image_column=self.data_config["image_column"],
            resolution=self.data_config["resolution"],
            center_crop=self.data_config["center_crop"],
            random_flip=self.data_config["random_flip"],
            image_interpolation_mode=self.data_config["image_interpolation_mode"],
            max_train_samples=self.data_config["max_train_samples"],
            allowed_extensions=self.data_config["allowed_extensions"],
        )
        return DataLoader(
            dataset,
            batch_size=self.train_config["train_batch_size"],
            shuffle=True,
            num_workers=self.train_config["dataloader_num_workers"],
            collate_fn=collate_image_only_batch,
        )

    def build_optimizer(self) -> torch.optim.Optimizer:
        optimizer_config = self.train_config["optimizer"]
        learning_rate = self.train_config["learning_rate"]
        if self.train_config["scale_lr"]:
            learning_rate = learning_rate * (
                self.train_config["gradient_accumulation_steps"]
                * self.train_config["train_batch_size"]
                * self.accelerator.num_processes
            )
        assert self.unet is not None
        return torch.optim.AdamW(
            self.unet.parameters(),
            lr=learning_rate,
            betas=(optimizer_config["adam_beta1"], optimizer_config["adam_beta2"]),
            weight_decay=optimizer_config["adam_weight_decay"],
            eps=optimizer_config["adam_epsilon"],
        )

    def build_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        return get_scheduler(
            self.train_config["lr_scheduler"],
            optimizer=optimizer,
            num_warmup_steps=self.train_config["lr_warmup_steps"] * self.accelerator.num_processes,
            num_training_steps=self.train_config["max_train_steps"] * self.accelerator.num_processes,
        )

    def initialize_trackers(self) -> None:
        if self._trackers_initialized or not self.accelerator.is_main_process or not self._logging_enabled:
            return
        tracker_config: dict[str, Any] = {}
        flatten_config("", self.config, tracker_config)
        init_kwargs: dict[str, Any] = {}
        experiment_name = self.logging_config.get("experiment_name")
        if experiment_name is not None:
            init_kwargs["swanlab"] = {"experiment_name": experiment_name}
        self.accelerator.init_trackers(
            self.logging_config["project_name"],
            config=tracker_config,
            init_kwargs=init_kwargs,
        )
        self._trackers_initialized = True

    def resolve_resume_checkpoint(self) -> tuple[Path | None, dict[str, Any]]:
        resume_value = self.train_config.get("resume_from_checkpoint")
        if not resume_value:
            return None, {}
        checkpoint_path = Path(resume_value)
        if resume_value == "latest":
            candidates = sorted(
                self.checkpoint_dir.glob("checkpoint-*"),
                key=lambda path: int(path.name.split("-", 1)[1]),
            )
            if not candidates:
                raise FileNotFoundError(
                    f"resume_from_checkpoint=latest requested, but no checkpoints exist under {self.checkpoint_dir}"
                )
            checkpoint_path = candidates[-1]
        elif not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoint_dir / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        metadata_path = checkpoint_path / "checkpoint_state.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return checkpoint_path, metadata

    def save_training_checkpoint(self, global_step: int, epoch: int) -> None:
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
                },
                checkpoint_path / "checkpoint_state.json",
            )
            self.prune_checkpoints()

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

    def encode_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert self.vae is not None
        assert self.latent_scaling_factor is not None
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(device=self.accelerator.device, dtype=self.weight_dtype)).latent_dist.sample()
            return latents * self.latent_scaling_factor

    def log_validation(self, global_step: int) -> None:
        if self.validation_config["num_validation_images"] <= 0 or not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return

        images = self.generate_validation_images(
            num_images=self.validation_config["num_validation_images"],
            num_inference_steps=self.validation_config["num_inference_steps"],
            seed=self.validation_config["seed"] if self.validation_config["seed"] is not None else self.train_config["seed"],
        )
        step_dir = ensure_dir(self.validation_dir / f"step-{global_step:08d}")
        for index, image in enumerate(images):
            image.save(step_dir / f"sample_{index:03d}.png")

        if self._logging_enabled:
            self.accelerator.log({"validation/global_step": global_step}, step=global_step)
            for tracker in self.accelerator.trackers:
                if tracker.name == "swanlab" and hasattr(tracker, "log_images"):
                    tracker.log_images({"validation/images": images}, step=global_step)
        self.accelerator.wait_for_everyone()

    def generate_validation_images(
        self,
        num_images: int,
        num_inference_steps: int,
        seed: int | None,
    ) -> list[Image.Image]:
        assert self.unet is not None
        assert self.vae is not None
        assert self.noise_scheduler is not None
        assert self.latent_scaling_factor is not None

        unwrapped_unet = self.accelerator.unwrap_model(self.unet)
        inference_scheduler = DDPMScheduler.from_config(self.noise_scheduler.config)
        inference_scheduler.set_timesteps(num_inference_steps, device=self.accelerator.device)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.accelerator.device).manual_seed(seed + num_inference_steps)

        latents = torch.randn(
            (
                num_images,
                self.model_config["unet"]["in_channels"],
                self.model_config["unet"]["sample_size"],
                self.model_config["unet"]["sample_size"],
            ),
            generator=generator,
            device=self.accelerator.device,
            dtype=self.weight_dtype,
        )

        was_training = unwrapped_unet.training
        unwrapped_unet.eval()
        with torch.no_grad():
            for timestep in inference_scheduler.timesteps:
                model_pred = unwrapped_unet(latents, timestep, return_dict=False)[0]
                latents = inference_scheduler.step(model_pred, timestep, latents).prev_sample

            decoded = self.vae.decode((latents / self.latent_scaling_factor).to(dtype=self.weight_dtype)).sample
        if was_training:
            unwrapped_unet.train()

        decoded = decoded.float().detach().cpu()
        decoded = ((decoded / 2) + 0.5).clamp(0, 1)
        decoded = decoded.permute(0, 2, 3, 1).numpy()
        return [Image.fromarray((image * 255).round().astype(np.uint8)) for image in decoded]

    def export_bundle(self, global_step: int, loss_history: list[float]) -> Path:
        assert self.unet is not None
        assert self.noise_scheduler is not None
        assert self.latent_scaling_factor is not None

        unet_dir = ensure_dir(self.export_dir / "unet")
        scheduler_dir = ensure_dir(self.export_dir / "scheduler")
        if self.accelerator.is_main_process:
            self.accelerator.unwrap_model(self.unet).save_pretrained(unet_dir)
            self.noise_scheduler.save_pretrained(scheduler_dir)
            metadata = {
                "model_type": self.model_config["model_type"],
                "image_resolution": self.data_config["resolution"],
                "latent_shape": [
                    self.model_config["unet"]["in_channels"],
                    self.model_config["unet"]["sample_size"],
                    self.model_config["unet"]["sample_size"],
                ],
                "vae": {
                    "pretrained_model_name_or_path": self.model_config["pretrained_vae_model_name_or_path"],
                    "subfolder": self.model_config["vae_subfolder"],
                    "local_files_only": self.model_config["local_files_only"],
                    "latent_scaling_factor": self.latent_scaling_factor,
                },
            }
            train_summary = {
                "model_type": self.model_config["model_type"],
                "num_steps": global_step,
                "num_updates": len(loss_history),
                "mean_loss": float(sum(loss_history) / max(1, len(loss_history))),
                "min_loss": float(min(loss_history)) if loss_history else math.nan,
                "export_dir": str(self.export_dir),
            }
            for filename in MODEL_METADATA_FILENAMES:
                write_json(metadata, self.export_dir / filename)
            for filename in TRAINING_SUMMARY_FILENAMES:
                write_json(train_summary, self.export_dir / filename)
                write_json(train_summary, self.output_dir / filename)
        return self.export_dir

    def train(self) -> Path:
        self.setup_models()
        assert self.unet is not None
        train_dataloader = self.build_dataloader()
        train_dataset_size = len(train_dataloader.dataset) if hasattr(train_dataloader, "dataset") else None
        optimizer = self.build_optimizer()
        lr_scheduler = self.build_lr_scheduler(optimizer)

        self.unet, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            self.unet, optimizer, train_dataloader, lr_scheduler
        )
        num_batches_per_epoch = len(train_dataloader)
        if num_batches_per_epoch <= 0:
            raise ValueError("Training dataloader is empty. Check data.train_manifest, data.train_image_dir, and max_train_samples.")

        num_update_steps_per_epoch = math.ceil(
            num_batches_per_epoch / self.train_config["gradient_accumulation_steps"]
        )
        max_train_steps = int(self.train_config["max_train_steps"])
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        total_batch_size = (
            self.train_config["train_batch_size"]
            * self.accelerator.num_processes
            * self.train_config["gradient_accumulation_steps"]
        )

        self.initialize_trackers()

        if self.accelerator.is_main_process:
            overview_lines = [
                "***** Running unconditional latent diffusion training *****",
                f"  Num examples = {train_dataset_size if train_dataset_size is not None else 'unknown'}",
                f"  Num batches each epoch = {num_batches_per_epoch}",
                f"  Num Epochs = {num_train_epochs}",
                f"  Instantaneous batch size per device = {self.train_config['train_batch_size']}",
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}",
                f"  Gradient Accumulation steps = {self.train_config['gradient_accumulation_steps']}",
                f"  Total optimization steps = {max_train_steps}",
            ]
            for line in overview_lines:
                LOGGER.info(line)
                self.accelerator.print(line)

        global_step = 0
        checkpoint_path, checkpoint_metadata = self.resolve_resume_checkpoint()
        if checkpoint_path is not None:
            self.accelerator.print(f"Resuming from checkpoint {checkpoint_path}")
            self.accelerator.load_state(str(checkpoint_path))
            global_step = int(checkpoint_metadata.get("global_step", 0))

        first_epoch = global_step // num_update_steps_per_epoch
        progress_bar = tqdm(
            total=max_train_steps,
            initial=global_step,
            disable=not self.accelerator.is_local_main_process,
            desc="train_uncond_ldm",
        )

        assert self.noise_scheduler is not None
        loss_history: list[float] = []
        running_loss = 0.0
        for epoch in range(first_epoch, num_train_epochs):
            self.unet.train()
            for batch in train_dataloader:
                with self.accelerator.accumulate(self.unet):
                    clean_latents = self.encode_latents(batch["pixel_values"])
                    noise = torch.randn_like(clean_latents)
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (clean_latents.shape[0],),
                        device=self.accelerator.device,
                        dtype=torch.long,
                    )
                    noisy_latents = self.noise_scheduler.add_noise(clean_latents, noise, timesteps)
                    model_pred = self.unet(noisy_latents, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                    avg_loss = self.accelerator.gather(loss.detach().repeat(clean_latents.shape[0])).mean()
                    running_loss += avg_loss.item() / self.accelerator.gradient_accumulation_steps

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), self.train_config["max_grad_norm"])
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if not self.accelerator.sync_gradients:
                    continue

                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{avg_loss.item():.4f}")
                loss_history.append(avg_loss.item())

                if self._logging_enabled:
                    self.accelerator.log(
                        {
                            "train/loss": running_loss,
                            "train/lr": float(lr_scheduler.get_last_lr()[0]),
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step,
                        },
                        step=global_step,
                    )
                running_loss = 0.0

                if global_step % self.train_config["checkpointing_steps"] == 0:
                    self.save_training_checkpoint(global_step, epoch)

                if (
                    self.validation_config["num_validation_images"] > 0
                    and global_step % self.validation_config["validation_steps"] == 0
                ):
                    self.log_validation(global_step)

                if global_step >= max_train_steps:
                    break

            if global_step >= max_train_steps:
                break

        self.accelerator.wait_for_everyone()
        export_dir = self.export_bundle(global_step, loss_history)
        self.accelerator.end_training()
        return export_dir
