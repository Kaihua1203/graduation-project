from __future__ import annotations

import argparse

from common.runtime import set_seed
from uncond_ldm.config import load_uncond_ldm_train_config
from uncond_ldm.trainer import UncondLatentDiffusionTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train unconditional latent diffusion in 2d-gen.")
    parser.add_argument("--config", required=True, help="Path to the unconditional training YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_uncond_ldm_train_config(args.config)
    set_seed(config["train"]["seed"])
    trainer = UncondLatentDiffusionTrainer(config)
    export_dir = trainer.train()
    print(f"Training finished. Export bundle: {export_dir}")


if __name__ == "__main__":
    main()
