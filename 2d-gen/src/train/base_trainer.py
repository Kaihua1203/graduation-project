from __future__ import annotations

from contextlib import nullcontext
import importlib
import math
from pathlib import Path
from typing import Any

import torch
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


class BaseDiffusionTrainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mixed_precision = config["train"].get("mixed_precision")
        self.autocast_enabled = self.device.type == "cuda" and mixed_precision == "fp16"
        self.dtype = torch.float16 if self.autocast_enabled else torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.autocast_enabled)
        adapter_cls = resolve_adapter_class(config["model"]["family"])
        self.adapter: BaseModelAdapter = adapter_cls(config)
        self.output_dir = ensure_dir(config["train"]["output_dir"])
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")

    def build_dataloader(self) -> DataLoader:
        dataset = ManifestImagePromptDataset(
            manifest_path=self.config["data"]["train_manifest"],
            image_size=self.config["train"]["image_size"],
        )
        return DataLoader(
            dataset,
            batch_size=self.config["train"]["batch_size"],
            shuffle=True,
            num_workers=self.config["train"].get("num_workers", 0),
            collate_fn=collate_manifest_batch,
        )

    def train(self) -> Path:
        self.adapter.setup(self.device)
        dataloader = self.build_dataloader()
        optimizer = torch.optim.AdamW(
            self.adapter.get_trainable_parameters(),
            lr=self.config["train"]["learning_rate"],
        )

        max_steps = self.config["train"]["max_train_steps"]
        epochs = self.config["train"]["num_epochs"]
        log_every_n_steps = self.config["train"].get("log_every_n_steps", 10)
        save_every_n_steps = self.config["train"].get("save_every_n_steps", max_steps)
        grad_accum = self.config["train"].get("gradient_accumulation_steps", 1)

        global_step = 0
        loss_history: list[float] = []

        for epoch in range(epochs):
            progress = tqdm(dataloader, desc=f"epoch={epoch}", leave=True)
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(progress, start=1):
                autocast_context = (
                    torch.cuda.amp.autocast(dtype=torch.float16)
                    if self.autocast_enabled
                    else nullcontext()
                )
                with autocast_context:
                    conditioning = self.adapter.encode_text(batch, self.device, self.dtype)
                    clean_latents = self.adapter.encode_latents(batch, self.device, self.dtype)
                    noise = torch.randn_like(clean_latents)
                    time_state = self.adapter.sample_time_state(clean_latents, batch, self.device)
                    model_input = self.adapter.prepare_noisy_input(
                        clean_latents, time_state, noise, batch, self.device, self.dtype
                    )
                    prediction = self.adapter.forward_model(model_input, time_state, conditioning, batch)
                    target = self.adapter.build_target(clean_latents, noise, time_state, batch)
                    loss = self.adapter.compute_loss(prediction, target, batch)
                loss_to_backprop = loss / grad_accum
                self.scaler.scale(loss_to_backprop).backward()

                if step % grad_accum == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    loss_history.append(loss.item())
                    progress.set_postfix(loss=f"{loss.item():.4f}")

                    if global_step % log_every_n_steps == 0:
                        print(f"step={global_step} loss={loss.item():.6f}")

                    if global_step % save_every_n_steps == 0:
                        checkpoint_path = self.checkpoint_dir / f"step_{global_step}"
                        ensure_dir(checkpoint_path)
                        self.adapter.save_checkpoint(str(checkpoint_path))

                    if global_step >= max_steps:
                        break

            if global_step >= max_steps:
                break

        final_checkpoint = self.checkpoint_dir / "final_lora"
        ensure_dir(final_checkpoint)
        self.adapter.save_checkpoint(str(final_checkpoint))
        summary = {
            "family": self.config["model"]["family"],
            "num_steps": global_step,
            "num_updates": len(loss_history),
            "mean_loss": float(sum(loss_history) / max(1, len(loss_history))),
            "min_loss": float(min(loss_history)) if loss_history else math.nan,
            "final_checkpoint": str(final_checkpoint),
        }
        write_json(summary, self.output_dir / "train_summary.json")
        return final_checkpoint
